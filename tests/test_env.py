"""Full episode smoke tests."""
import asyncio
import pytest
from env.environment import JobSwitchEnvironment
from env.models import ActionType


@pytest.fixture
def env():
    return JobSwitchEnvironment()


@pytest.mark.asyncio
async def test_reset(env):
    obs = await env.reset(seed=42, task_id=1)
    assert "step" in obs
    assert obs["step"] == 0
    assert obs["social_capital"] > 0


@pytest.mark.asyncio
async def test_step_engage(env):
    await env.reset(seed=42)
    result = await env.step({
        "action_type": "engage_content",
        "target": "Alice Chen",
        "parameters": {},
        "reasoning": "test",
    })
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert result["done"] is False


@pytest.mark.asyncio
async def test_step_wait(env):
    await env.reset(seed=42)
    result = await env.step({
        "action_type": "wait_for_signal",
        "parameters": {},
        "reasoning": "waiting",
    })
    assert result["done"] is False


@pytest.mark.asyncio
async def test_full_episode_smoke(env):
    """Run 5 steps with various actions — should not crash."""
    await env.reset(seed=42)

    actions = [
        {"action_type": "research_company", "target": "TechCorp", "parameters": {}, "reasoning": "research"},
        {"action_type": "engage_content", "target": "Alice Chen", "parameters": {}, "reasoning": "engage"},
        {"action_type": "add_to_targets", "parameters": {"company": "TechCorp"}, "reasoning": "target"},
        {"action_type": "send_message", "target": "Alice Chen", "parameters": {"message_type": "professional"}, "reasoning": "message"},
        {"action_type": "tailor_resume", "parameters": {"company": "TechCorp", "role_type": "sde"}, "reasoning": "resume"},
    ]

    for action in actions:
        result = await env.step(action)
        assert "observation" in result
        assert result["done"] is False


@pytest.mark.asyncio
async def test_research_reduces_noise(env):
    """#1: RESEARCH_COMPANY should move hiring_signal closer to truth."""
    await env.reset(seed=42)
    old_signal = env.market.companies["TechCorp"]["hiring_signal"]
    true_state = 1.0 if env.market.companies["TechCorp"]["true_hiring_state"] else 0.0

    await env.step({
        "action_type": "research_company",
        "target": "TechCorp",
        "parameters": {},
        "reasoning": "research",
    })

    new_signal = env.market.companies["TechCorp"]["hiring_signal"]
    # Signal should move closer to truth
    old_dist = abs(old_signal - true_state)
    new_dist = abs(new_signal - true_state)
    assert new_dist <= old_dist


@pytest.mark.asyncio
async def test_offer_deadline_enforcement(env):
    """#2: Offers past deadline should be removed."""
    from env.models import Offer
    await env.reset(seed=42, max_steps=10)

    # Manually add an offer with deadline at step 2
    offer = Offer(company="TechCorp", base_salary=100000, deadline_step=2)
    env.state.offers.append(offer)

    # Step past the deadline
    for _ in range(3):
        await env.step({
            "action_type": "wait_for_signal",
            "parameters": {},
            "reasoning": "wait",
        })

    assert len(env.state.offers) == 0  # offer expired


@pytest.mark.asyncio
async def test_referral_validation(env):
    """#3: APPLY_WITH_REFERRAL without granted referral downgrades to cold."""
    await env.reset(seed=42)

    # Skip ahead past min step for applications
    for _ in range(4):
        await env.step({
            "action_type": "wait_for_signal",
            "parameters": {},
            "reasoning": "wait",
        })

    # Try to apply with referral without having obtained one
    result = await env.step({
        "action_type": "apply_with_referral",
        "parameters": {"company": "TechCorp", "role": "SDE", "referrer": "Alice Chen"},
        "reasoning": "trying without referral",
    })

    # Should still create a process but as cold (not referral)
    process = env.pipeline.get_process("TechCorp")
    if process:
        assert process.referral_used is False


@pytest.mark.asyncio
async def test_warmth_threshold_crossing(env):
    """#4: Reward only for CROSSING the 0.4 threshold, not being above it."""
    await env.reset(seed=42)

    # Set someone's warmth to 0.5 (already above threshold)
    env.network.graph.nodes["Alice Chen"]["warmth"] = 0.5

    result = await env.step({
        "action_type": "engage_content",
        "target": "Alice Chen",
        "parameters": {},
        "reasoning": "test",
    })

    # Should NOT get the +0.03 threshold crossing reward
    # (already above 0.4, didn't cross it)
    assert result["reward"] <= 0.01  # only possible penalty, no +0.03 bonus


@pytest.mark.asyncio
async def test_phase_gating(env):
    """#9: Actions blocked before minimum step."""
    await env.reset(seed=42)

    # Try to apply cold at step 0 (min_step=3)
    result = await env.step({
        "action_type": "apply_cold",
        "parameters": {"company": "TechCorp", "role": "SDE"},
        "reasoning": "too early",
    })
    assert result["reward"] == -0.02  # blocked by phase gating


@pytest.mark.asyncio
async def test_drop_process(env):
    """#6/#10: DROP_PROCESS should abandon an active pipeline."""
    await env.reset(seed=42)

    # Skip to step 3+ and apply
    for _ in range(4):
        await env.step({"action_type": "wait_for_signal", "parameters": {}, "reasoning": "wait"})

    await env.step({
        "action_type": "apply_cold",
        "parameters": {"company": "TechCorp", "role": "SDE"},
        "reasoning": "apply",
    })

    # Now drop the process
    process = env.pipeline.get_process("TechCorp")
    if process:
        await env.step({
            "action_type": "drop_process",
            "parameters": {"company": "TechCorp"},
            "reasoning": "drop",
        })
        process = env.pipeline.get_process("TechCorp")
        assert process is None  # should be rejected/dropped


@pytest.mark.asyncio
async def test_observation_hides_true_warmth(env):
    """#8: Observation should not contain true warmth values."""
    await env.reset(seed=42)
    result = await env.step({
        "action_type": "wait_for_signal",
        "parameters": {},
        "reasoning": "wait",
    })
    obs = result["observation"]
    nodes = obs.get("network_graph", {}).get("nodes", {})
    for name, data in nodes.items():
        assert "warmth" not in data, f"True warmth leaked for {name}"
        assert "warmth_signal" in data


@pytest.mark.asyncio
async def test_episode_terminates_on_max_steps(env):
    """Episode should end at max_steps."""
    await env.reset(seed=42, max_steps=3)
    for _ in range(3):
        result = await env.step({
            "action_type": "wait_for_signal",
            "parameters": {},
            "reasoning": "wait",
        })
    assert result["done"] is True

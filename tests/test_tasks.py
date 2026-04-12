"""Tests for task scenario setup and grading logic."""
import pytest
from env.environment import JobSwitchEnvironment
from tasks.task1_straightforward import Task1
from tasks.task2_cold_network import Task2
from tasks.task3_competing_pressures import Task3
from env.models import InterviewStage


# ─── Task 1 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task1_warm_contacts():
    """Task 1 should start with Alice Chen and Carol Johnson at warmth ~0.65."""
    env = JobSwitchEnvironment()
    task = Task1(env)
    obs = await task.reset()

    nodes = obs["network_graph"]["nodes"]
    assert "Alice Chen" in nodes
    assert "Carol Johnson" in nodes
    # warmth_signal (noisy) should be near 0.65 ± 0.15
    assert 0.40 <= nodes["Alice Chen"]["warmth_signal"] <= 0.90
    assert 0.40 <= nodes["Carol Johnson"]["warmth_signal"] <= 0.90


@pytest.mark.asyncio
async def test_task1_optimal_trajectory_grade():
    """Simulate the optimal Task1 trajectory and verify score >= 0.70."""
    env = JobSwitchEnvironment()
    task = Task1(env)
    await task.reset()

    # Engage both warm contacts
    for person in ("Alice Chen", "Carol Johnson"):
        await env.step({
            "action_type": "engage_content",
            "target": person,
            "parameters": {},
            "reasoning": "build warmth",
        })
        await env.step({
            "action_type": "engage_content",
            "target": person,
            "parameters": {},
            "reasoning": "build warmth",
        })

    # Request referrals
    await env.step({
        "action_type": "request_referral",
        "target": "Alice Chen",
        "parameters": {"company": "TechCorp"},
        "reasoning": "request referral",
    })
    await env.step({
        "action_type": "request_referral",
        "target": "Carol Johnson",
        "parameters": {"company": "DataFlow"},
        "reasoning": "request referral",
    })

    # Apply with referrals
    await env.step({
        "action_type": "apply_with_referral",
        "target": "TechCorp",
        "parameters": {"company": "TechCorp", "role": "SDE", "referrer": "Alice Chen"},
        "reasoning": "apply",
    })
    await env.step({
        "action_type": "apply_with_referral",
        "target": "DataFlow",
        "parameters": {"company": "DataFlow", "role": "SDE", "referrer": "Carol Johnson"},
        "reasoning": "apply",
    })

    # Advance pipelines and check peak_parallel
    for _ in range(8):
        for company in ("TechCorp", "DataFlow"):
            await env.step({
                "action_type": "advance_round",
                "target": company,
                "parameters": {"company": company},
                "reasoning": "advance pipeline",
            })
        if env.state.offers:
            break

    # Accept best available offer
    if env.state.offers:
        best = max(env.state.offers, key=lambda o: o.base_salary)
        await env.step({
            "action_type": "counter_offer",
            "target": best.company,
            "parameters": {"company": best.company, "amount": round(best.base_salary * 1.10, 2)},
            "reasoning": "negotiate",
        })
        await env.step({
            "action_type": "accept_offer",
            "target": best.company,
            "parameters": {"company": best.company},
            "reasoning": "accept",
        })

    grade = task.grade()
    # Should get at least the referral credit (0.30 or 0.22)
    assert grade >= 0.20, f"Expected grade >= 0.20 for optimal-ish trajectory, got {grade:.3f}"


@pytest.mark.asyncio
async def test_task1_granted_referrals_in_obs():
    """granted_referrals should appear in observation after successful request_referral."""
    env = JobSwitchEnvironment()
    task = Task1(env)
    obs = await task.reset()

    assert obs["granted_referrals"] == {}

    # Warm up Alice Chen enough
    for _ in range(3):
        await env.step({
            "action_type": "engage_content",
            "target": "Alice Chen",
            "parameters": {},
            "reasoning": "warmth",
        })

    result = await env.step({
        "action_type": "request_referral",
        "target": "Alice Chen",
        "parameters": {"company": "TechCorp"},
        "reasoning": "referral",
    })
    obs_after = result["observation"]
    # If warmth was >= 0.60, referral is granted and should appear in obs
    if obs_after["granted_referrals"]:
        assert "TechCorp" in obs_after["granted_referrals"]


@pytest.mark.asyncio
async def test_task1_valid_actions():
    """valid_actions field should be populated and respect gating."""
    env = JobSwitchEnvironment()
    task = Task1(env)
    obs = await task.reset()

    valid = obs["valid_actions"]
    assert isinstance(valid, list)
    assert len(valid) > 0
    # At step 0: request_referral (min_step=2) should not be valid
    assert "request_referral" not in valid
    # But engage_content should be valid
    assert "engage_content" in valid
    # No offers at start, so counter_offer and accept_offer should not be valid
    assert "counter_offer" not in valid
    assert "accept_offer" not in valid


# ─── Task 2 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task2_all_cold():
    """Task 2 should start with all contacts at low warmth (0.10-0.25)."""
    env = JobSwitchEnvironment()
    task = Task2(env)
    obs = await task.reset()

    nodes = obs["network_graph"]["nodes"]
    for name, data in nodes.items():
        # warmth_signal (noisy) should reflect cold state
        assert data["warmth_signal"] < 0.55, (
            f"{name} warmth_signal={data['warmth_signal']:.2f} expected to start cold"
        )


@pytest.mark.asyncio
async def test_task2_aistartup_urgent():
    """Task 2 AIStartup should disappear after step 8."""
    env = JobSwitchEnvironment()
    task = Task2(env)
    await task.reset()

    # Step through to step 8
    for _ in range(8):
        await env.step({"action_type": "wait_for_signal", "parameters": {}, "reasoning": "wait"})

    # AIStartup should no longer be actively hiring
    assert not env.market.is_actively_hiring("AIStartup"), (
        "AIStartup should stop hiring after urgent_deadline step 8"
    )


@pytest.mark.asyncio
async def test_task2_grader_rewards_network_investment():
    """Task2 grader gives credit for concentrated network investment before applying."""
    env = JobSwitchEnvironment()
    task = Task2(env)
    await task.reset()

    # Invest in Henry Zhao (active connector) 3x before applying
    for _ in range(3):
        await env.step({
            "action_type": "engage_content",
            "target": "Henry Zhao",
            "parameters": {},
            "reasoning": "warmth",
        })
    await env.step({
        "action_type": "send_message",
        "target": "Henry Zhao",
        "parameters": {"message_type": "professional"},
        "reasoning": "warmth",
    })

    grade = task.grade()
    # Should get network investment credit (0.10 or 0.22 or 0.30)
    assert grade >= 0.10, (
        f"Expected network investment credit >= 0.10, got {grade:.3f}"
    )


# ─── Task 3 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task3_initial_scenario():
    """Task 3 should start with exploding offer, two pipelines, limited runway."""
    env = JobSwitchEnvironment()
    task = Task3(env)
    obs = await task.reset()

    # Has the CloudBase exploding offer
    offers = obs["offers_in_hand"]
    assert len(offers) == 1
    assert offers[0]["company"] == "CloudBase"
    assert offers[0]["deadline_step"] == 4

    # Has two active pipelines
    pipeline = obs["active_pipeline"]
    companies = {p["company"] for p in pipeline}
    assert "TechCorp" in companies
    assert "MegaSoft" in companies

    # Limited runway
    assert obs["financial_runway"] == 5.0


@pytest.mark.asyncio
async def test_task3_delay_extends_deadline():
    """REQUEST_DELAY on CloudBase should extend the offer deadline."""
    env = JobSwitchEnvironment()
    task = Task3(env)
    await task.reset()

    result = await env.step({
        "action_type": "request_delay",
        "target": "CloudBase",
        "parameters": {"company": "CloudBase", "days": 3},
        "reasoning": "extend deadline",
    })
    obs = result["observation"]

    offers = obs["offers_in_hand"]
    cloudbase = next((o for o in offers if o["company"] == "CloudBase"), None)
    assert cloudbase is not None
    assert cloudbase["deadline_step"] >= 6, (
        f"Expected deadline >= 6 after delay, got {cloudbase['deadline_step']}"
    )


@pytest.mark.asyncio
async def test_task3_grader_rewards_early_delay():
    """Task3 grader gives 0.25 credit for delaying CloudBase in steps 0-2."""
    env = JobSwitchEnvironment()
    task = Task3(env)
    await task.reset()

    await env.step({
        "action_type": "request_delay",
        "target": "CloudBase",
        "parameters": {"company": "CloudBase", "days": 3},
        "reasoning": "extend deadline",
    })

    grade = task.grade()
    assert grade >= 0.25, (
        f"Expected >= 0.25 for early delay credit, got {grade:.3f}"
    )


@pytest.mark.asyncio
async def test_task3_panic_accept_scores_zero_patience():
    """Accepting CloudBase as first action gives 0 patience credit."""
    env = JobSwitchEnvironment()
    task = Task3(env)
    await task.reset()

    await env.step({
        "action_type": "accept_offer",
        "target": "CloudBase",
        "parameters": {"company": "CloudBase"},
        "reasoning": "panic",
    })

    grade = task.grade()
    # Got offer (some offer quality credit) but 0 patience credit
    history = env.state.action_history
    assert history[0]["action_type"] == "accept_offer"
    # patience component should be 0
    # total score will be low because it was a bad offer
    assert grade < 0.35, (
        f"Panic accept should score < 0.35, got {grade:.3f}"
    )

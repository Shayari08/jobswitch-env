"""Test Pydantic models import and validate correctly."""
import pytest
from env.models import (
    Action,
    ActionType,
    EpisodeState,
    InterviewProcess,
    InterviewStage,
    Observation,
    Offer,
    Phase,
    Reward,
)


def test_action_type_enum():
    assert len(ActionType) == 17  # 16 original + DROP_PROCESS
    assert ActionType.RESEARCH_COMPANY.value == "research_company"
    assert ActionType.ACCEPT_OFFER.value == "accept_offer"


def test_phase_enum():
    assert len(Phase) == 5
    assert Phase.TARGETING.value == "targeting"


def test_action_valid():
    a = Action(
        action_type=ActionType.ENGAGE_CONTENT,
        target="Alice Chen",
        parameters={},
        reasoning="Building warmth",
    )
    assert a.action_type == ActionType.ENGAGE_CONTENT
    assert a.target == "Alice Chen"


def test_action_requires_reasoning():
    with pytest.raises(Exception):
        Action(action_type=ActionType.WAIT_FOR_SIGNAL, parameters={})


def test_action_validates_required_params():
    # SEND_MESSAGE requires person and message_type
    with pytest.raises(Exception):
        Action(
            action_type=ActionType.SEND_MESSAGE,
            parameters={},
            reasoning="test",
        )


def test_observation_defaults():
    obs = Observation(step=0, current_phase=1)
    assert obs.social_capital == 1.0
    assert obs.time_pressure == 0.0
    assert obs.target_companies == []


def test_reward_bounds():
    r = Reward(score=0.5)
    assert 0.0 <= r.score <= 1.0

    with pytest.raises(Exception):
        Reward(score=1.5)


def test_episode_state_defaults():
    state = EpisodeState()
    assert state.step == 0
    assert state.phase == Phase.TARGETING
    assert state.social_capital == 1.0
    assert state.bridges_burned == 0


def test_interview_stages():
    assert InterviewStage.APPLIED.value == "APPLIED"
    assert InterviewStage.OFFER.value == "OFFER"


def test_offer_creation():
    o = Offer(company="TechCorp", base_salary=100000)
    assert o.equity == 0.0
    assert o.negotiation_state == "initial"

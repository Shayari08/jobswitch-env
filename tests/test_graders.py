"""Test graders with known episode states."""
from env.models import EpisodeState, Offer
from env.reward import compute_reward
from simulation.network_graph import NetworkGraph


def test_perfect_episode_reward():
    """A perfect episode should score ~0.60+."""
    state = EpisodeState()
    state.step = 10
    state.max_steps = 20
    state.market_rate = 100000
    state.peak_parallel_pipelines = 3
    state.accepted_offer = Offer(
        company="TechCorp",
        base_salary=110000,
        equity=10000,
        culture_fit=0.8,
        role_growth_potential=0.7,
    )
    state.first_offer_received = Offer(
        company="TechCorp",
        base_salary=95000,
        equity=5000,
        culture_fit=0.8,
        role_growth_potential=0.7,
    )
    state.bridges_burned = 0
    state.desperation_signals = 0

    network = NetworkGraph(seed=42)
    state.initial_warmth_snapshot = network.snapshot_warmth()
    for person in list(network.graph.nodes)[:3]:
        network.update_warmth(person, 0.1)

    reward = compute_reward(state, network)
    assert reward.score >= 0.50, f"Perfect episode scored too low: {reward.score}"


def test_terrible_episode_reward():
    """A terrible episode should score ~0.0-0.15."""
    state = EpisodeState()
    state.step = 30
    state.max_steps = 30
    state.market_rate = 100000
    state.peak_parallel_pipelines = 0
    state.accepted_offer = None
    state.bridges_burned = 2
    state.desperation_signals = 3

    network = NetworkGraph(seed=42)
    state.initial_warmth_snapshot = network.snapshot_warmth()
    for person in network.graph.nodes:
        network.update_warmth(person, -0.3)

    reward = compute_reward(state, network)
    assert reward.score <= 0.15, f"Terrible episode scored too high: {reward.score}"


def test_reward_components():
    """Verify reward components are populated."""
    state = EpisodeState()
    state.step = 15
    state.max_steps = 30
    state.market_rate = 100000
    state.accepted_offer = Offer(
        company="X", base_salary=90000,
        culture_fit=0.6, role_growth_potential=0.5,
    )
    state.first_offer_received = state.accepted_offer

    network = NetworkGraph(seed=42)
    state.initial_warmth_snapshot = network.snapshot_warmth()

    reward = compute_reward(state, network)
    assert reward.offer_quality > 0
    assert reward.time_efficiency > 0
    assert "offer_quality_weighted" in reward.component_scores

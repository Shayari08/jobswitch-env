from env.models import EpisodeState, Reward


def offer_quality_score(state: EpisodeState) -> float:
    """final_salary / market_rate + role_growth_potential + culture_fit, normalized [0,1]."""
    if not state.accepted_offer:
        return 0.0
    offer = state.accepted_offer
    salary_ratio = min(1.0, offer.base_salary / max(state.market_rate, 1.0))
    equity_bonus = min(0.1, offer.equity / max(state.market_rate, 1.0))
    # Use actual role_growth_potential and culture_fit from the offer
    growth = offer.role_growth_potential  # [0, 1]
    culture = offer.culture_fit  # [0, 1]
    raw = salary_ratio * 0.5 + growth * 0.2 + culture * 0.2 + equity_bonus
    return min(1.0, max(0.0, raw))


def process_efficiency_score(state: EpisodeState) -> float:
    """peak_parallel_pipelines / 3, capped at 1.0."""
    return min(1.0, state.peak_parallel_pipelines / 3.0)


def network_health_score(state: EpisodeState, network) -> float:
    """avg_warmth_end / avg_warmth_start."""
    if not state.initial_warmth_snapshot:
        return 0.5
    start_values = list(state.initial_warmth_snapshot.values())
    if not start_values:
        return 0.5
    avg_start = sum(start_values) / len(start_values)
    if avg_start < 0.01:
        return 0.5

    current_warmth = network.snapshot_warmth()
    end_values = list(current_warmth.values())
    avg_end = sum(end_values) / len(end_values) if end_values else 0.0
    return min(1.0, avg_end / avg_start)


def negotiation_score(state: EpisodeState) -> float:
    """final_comp / first_offer_comp."""
    if not state.accepted_offer or not state.first_offer_received:
        return 0.0 if not state.accepted_offer else 0.5
    first_comp = (
        state.first_offer_received.base_salary
        + state.first_offer_received.equity
        + state.first_offer_received.signing_bonus
    )
    final_comp = (
        state.accepted_offer.base_salary
        + state.accepted_offer.equity
        + state.accepted_offer.signing_bonus
    )
    if first_comp <= 0:
        return 0.5
    return min(1.0, final_comp / first_comp)


def compute_reward(state: EpisodeState, network=None) -> Reward:
    """Compute full reward at episode end. Returns Reward with score in [0, 1]."""
    oq = offer_quality_score(state)
    pe = process_efficiency_score(state)
    nh = network_health_score(state, network) if network else 0.5
    nq = negotiation_score(state)
    te = max(0.0, 1.0 - state.step / max(state.max_steps, 1))

    R = 0.0
    R += 0.35 * oq
    R += 0.20 * pe
    R += 0.20 * nh
    R += 0.15 * nq
    R += 0.10 * te

    # Hard penalties
    penalties = {}
    if not state.accepted_offer:
        R -= 0.50
        penalties["no_offer"] = -0.50
    if state.bridges_burned > 0:
        pen = 0.15 * state.bridges_burned
        R -= pen
        penalties["bridges_burned"] = -pen
    if state.desperation_signals > 0:
        pen = 0.10 * min(state.desperation_signals, 3)
        R -= pen
        penalties["desperation_signals"] = -pen

    score = max(0.0, min(1.0, R))

    return Reward(
        score=score,
        offer_quality=oq,
        process_efficiency=pe,
        network_health=nh,
        negotiation_quality=nq,
        time_efficiency=te,
        penalties=penalties,
        component_scores={
            "offer_quality_weighted": 0.35 * oq,
            "process_efficiency_weighted": 0.20 * pe,
            "network_health_weighted": 0.20 * nh,
            "negotiation_quality_weighted": 0.15 * nq,
            "time_efficiency_weighted": 0.10 * te,
        },
    )

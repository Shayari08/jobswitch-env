from env.models import EpisodeState


def grade_negotiation(state: EpisodeState) -> dict:
    """Grade negotiation quality: final_comp / first_offer_comp."""
    if not state.accepted_offer:
        return {"score": 0.0, "detail": "No offer accepted"}

    if not state.first_offer_received:
        return {"score": 0.5, "detail": "No first offer recorded"}

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
        return {"score": 0.5, "detail": "First offer comp is zero"}

    ratio = final_comp / first_comp

    return {
        "score": min(1.0, max(0.0, ratio)),
        "first_comp": round(first_comp, 2),
        "final_comp": round(final_comp, 2),
        "improvement_ratio": round(ratio, 3),
        "counter_offers_made": state.accepted_offer.counter_offers,
    }

from env.models import EpisodeState


def grade_offer_quality(state: EpisodeState) -> dict:
    """Grade the quality of the accepted offer relative to market rate."""
    if not state.accepted_offer:
        return {"score": 0.0, "detail": "No offer accepted"}

    salary_ratio = state.accepted_offer.base_salary / max(state.market_rate, 1)
    equity_bonus = min(0.1, state.accepted_offer.equity / max(state.market_rate, 1))
    raw = salary_ratio * 0.8 + equity_bonus + 0.1

    return {
        "score": min(1.0, max(0.0, raw)),
        "salary_ratio": round(salary_ratio, 3),
        "equity_bonus": round(equity_bonus, 3),
        "company": state.accepted_offer.company,
    }

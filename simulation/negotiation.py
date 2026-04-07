"""Offer and counter-offer mechanics.

Handles the negotiation logic: counter-offer evaluation,
deal collapse thresholds, and component-level negotiation.
"""

from env.models import Offer
import random


class NegotiationEngine:
    """Evaluates counter-offers and determines company responses."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def evaluate_counter(self, offer: Offer, ask_salary: float,
                         ask_equity: float = 0,
                         ask_signing: float = 0) -> dict:
        """Evaluate a counter-offer. Returns result dict.

        If ask is > 130% of current, deal collapses.
        Otherwise company meets partway (30-70% of the gap).
        """
        ratio = ask_salary / max(offer.base_salary, 1)

        if ratio > 1.3:
            return {"accepted": False, "collapsed": True, "reason": "Too aggressive"}

        result = {"accepted": True, "collapsed": False}

        if ratio > 1.0:
            gap = ask_salary - offer.base_salary
            bump = gap * self.rng.uniform(0.3, 0.7)
            result["new_salary"] = round(offer.base_salary + bump, 2)
        else:
            result["new_salary"] = offer.base_salary

        if ask_equity > offer.equity:
            eq_gap = ask_equity - offer.equity
            if eq_gap / max(offer.equity + 1, 1) <= 1.5:
                eq_bump = eq_gap * self.rng.uniform(0.2, 0.6)
                result["new_equity"] = round(offer.equity + eq_bump, 2)
            else:
                result["new_equity"] = offer.equity
        else:
            result["new_equity"] = offer.equity

        if ask_signing > 0:
            sb_bump = ask_signing * self.rng.uniform(0.3, 0.7)
            result["new_signing"] = round(offer.signing_bonus + sb_bump, 2)
        else:
            result["new_signing"] = offer.signing_bonus

        return result

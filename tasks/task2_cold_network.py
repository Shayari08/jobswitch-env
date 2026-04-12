from tasks.base import BaseTask
from graders.network_health_grader import grade_network_health


class Task2(BaseTask):
    """The Cold Network Problem — MEDIUM.

    All connections start cold. One urgent company (role expires step 8).
    Social capital starts at 0.50, regenerates at +0.04/step.
    Target score: 0.40-0.60.
    """
    SEED = 202
    TASK_ID = 2
    MAX_STEPS = 25

    def _configure_scenario(self):
        env = self.env
        state = env.state

        state.financial_runway = 20.0
        state.max_steps = 25
        state.social_capital = 0.50
        state.social_capital_regen = 0.04
        env.network.social_capital = 0.50
        state.market_rate = env.market.get_market_rate()

        state.profile = {
            "skills": {"ml": 0.80, "backend": 0.65, "frontend": 0.25},
            "experience_months": 6,
            "seniority": "junior",
        }

        # All connections start cold (warmth 0.10-0.25)
        for name, data in env.network.graph.nodes(data=True):
            cold_warmth = env.network.rng.uniform(0.10, 0.25)
            data["warmth"] = cold_warmth
            data["referral_willingness_signal"] = max(
                0, min(1, cold_warmth + env.network.rng.gauss(0, 0.08))
            )

        # One urgent company: AIStartup role expires at step 8
        if "AIStartup" in env.market.companies:
            env.market.companies["AIStartup"]["true_hiring_state"] = True
            env.market.companies["AIStartup"]["hiring_signal"] = min(
                1.0, 0.90 + env.market.rng.gauss(0, 0.05)
            )
            env.market.companies["AIStartup"]["urgent_deadline"] = 8

        # Two non-urgent companies actively hiring
        for name in ("TechCorp", "DataFlow"):
            if name in env.market.companies:
                env.market.companies[name]["true_hiring_state"] = True
                env.market.companies[name]["hiring_signal"] = min(
                    1.0, 0.80 + env.market.rng.gauss(0, 0.05)
                )

        state.initial_warmth_snapshot = env.network.snapshot_warmth()

    def grade(self) -> float:
        env = self.env
        state = env.state
        history = state.action_history

        score = 0.0

        # (0.30) Did agent DELIBERATELY invest in network before applying?
        # Requires concentrated effort (>= 2 actions) on ONE person before step 7
        early_network_targets: dict[str, int] = {}
        first_apply_step = None
        for i, a in enumerate(history):
            at = a.get("action_type", "")
            target = a.get("target") or a.get("parameters", {}).get("person")
            if at in ("engage_content", "send_message", "request_intro") and i < 8 and target:
                early_network_targets[target] = early_network_targets.get(target, 0) + 1
            if at in ("apply_cold", "apply_with_referral") and first_apply_step is None:
                first_apply_step = i

        max_on_one = max(early_network_targets.values()) if early_network_targets else 0
        invested_before_applying = (
            max_on_one >= 2
            and (first_apply_step is None or first_apply_step >= 4)
        )
        if invested_before_applying and max_on_one >= 3:
            score += 0.30  # strong investment
        elif invested_before_applying:
            score += 0.22  # solid investment
        elif max_on_one >= 2:
            score += 0.10  # some concentration but applied too early
        elif max_on_one >= 1:
            score += 0.03  # minimal network effort

        # (0.25) Did agent successfully obtain AND use a referral?
        referral_companies = set(state.granted_referrals.keys())
        applied_with_valid_ref = any(
            a.get("action_type") == "apply_with_referral"
            and (a.get("parameters", {}).get("company") or a.get("target")) in referral_companies
            for a in history
        )
        if applied_with_valid_ref:
            score += 0.25
        elif len(referral_companies) >= 1:
            score += 0.12  # got referral but didn't apply with it
        elif any(a.get("action_type") == "request_referral" for a in history):
            score += 0.04  # attempted referral

        # (0.25) Did it accept an offer?
        if state.accepted_offer:
            ratio = state.accepted_offer.base_salary / max(state.market_rate, 1)
            if ratio >= 0.85:
                score += 0.25
            elif ratio >= 0.70:
                score += 0.18
            elif ratio >= 0.55:
                score += 0.10
            else:
                score += 0.05  # got an offer, at least
        elif any(a.get("action_type") == "advance_round" for a in history):
            score += 0.04  # made pipeline progress

        # (0.20) Network health — improved warmth from cold start is especially valuable
        nh_result = grade_network_health(state, env.network)
        nh = nh_result["score"]
        if nh >= 1.3:
            score += 0.20  # significant warmth improvement from cold start
        elif nh >= 1.1:
            score += 0.15
        elif nh >= 0.9:
            score += 0.08
        elif nh >= 0.7:
            score += 0.03

        return min(1.0, score)

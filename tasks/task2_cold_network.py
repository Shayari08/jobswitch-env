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
        # Enforced in environment.py step() — hiring flips to False at step 8.
        if "AIStartup" in env.market.companies:
            env.market.companies["AIStartup"]["true_hiring_state"] = True
            env.market.companies["AIStartup"]["hiring_signal"] = min(
                1.0, 0.90 + env.market.rng.gauss(0, 0.05)
            )
            env.market.companies["AIStartup"]["urgent_deadline"] = 8

        # Two non-urgent companies
        for name in ("TechCorp", "DataFlow"):
            if name in env.market.companies:
                env.market.companies[name]["true_hiring_state"] = True

        state.initial_warmth_snapshot = env.network.snapshot_warmth()

    def grade(self) -> float:
        env = self.env
        state = env.state
        history = state.action_history

        score = 0.0

        # (0.30) Did agent DELIBERATELY invest in network before applying?
        # Requires: >= 3 targeted network actions on the SAME person in steps 0-7
        # before any application action. Random agents scatter actions across people.
        early_network_targets = {}
        first_apply_step = None
        for i, a in enumerate(history):
            at = a.get("action_type", "")
            target = a.get("target") or a.get("parameters", {}).get("person")
            if at in ("engage_content", "send_message") and i < 8 and target:
                early_network_targets[target] = early_network_targets.get(target, 0) + 1
            if at in ("apply_cold", "apply_with_referral") and first_apply_step is None:
                first_apply_step = i

        # Need concentrated effort on at least one person
        max_on_one = max(early_network_targets.values()) if early_network_targets else 0
        invested_before_applying = (
            max_on_one >= 2
            and (first_apply_step is None or first_apply_step >= 4)
        )
        if invested_before_applying:
            score += 0.30
        elif max_on_one >= 2:
            score += 0.10

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
            score += 0.08

        # (0.25) Did it accept an offer with decent salary?
        if state.accepted_offer:
            ratio = state.accepted_offer.base_salary / max(state.market_rate, 1)
            if ratio >= 0.85:
                score += 0.25
            elif ratio >= 0.70:
                score += 0.12

        # (0.20) Network health — did agent preserve warmth despite starting cold?
        nh_result = grade_network_health(state, env.network)
        nh = nh_result["score"]
        if nh >= 1.2:
            score += 0.20  # improved the network
        elif nh >= 0.9:
            score += 0.10

        return min(1.0, score)

from tasks.base import BaseTask
from env.models import InterviewStage
from env.reward import compute_reward
from graders.network_health_grader import grade_network_health


class Task1(BaseTask):
    """The Straightforward Pivot — EASY.

    Fresher SDE, same-domain switch. Two warm connections at target companies.
    Market actively hiring. Ample financial runway.
    Target score: 0.70-0.85.
    """
    SEED = 101
    TASK_ID = 1
    MAX_STEPS = 20

    def _configure_scenario(self):
        env = self.env
        state = env.state

        state.financial_runway = 15.0
        state.max_steps = 20
        state.market_rate = env.market.get_market_rate()

        state.profile = {
            "skills": {"ml": 0.85, "backend": 0.70, "frontend": 0.30},
            "experience_months": 6,
            "seniority": "junior",
        }

        # Pre-warm two connections at target companies
        warm_targets = [
            ("Alice Chen", "TechCorp", 0.65),
            ("Carol Johnson", "DataFlow", 0.65),
        ]
        for person, company, warmth in warm_targets:
            if person in env.network.graph:
                env.network.graph.nodes[person]["warmth"] = warmth
                env.network.graph.nodes[person]["referral_willingness_signal"] = (
                    warmth + env.network.rng.gauss(0, 0.05)
                )

        # Ensure both target companies have strong hiring signals
        for name in ("TechCorp", "DataFlow"):
            if name in env.market.companies:
                env.market.companies[name]["true_hiring_state"] = True
                env.market.companies[name]["hiring_signal"] = min(
                    1.0, 0.85 + env.market.rng.gauss(0, 0.05)
                )

        state.initial_warmth_snapshot = env.network.snapshot_warmth()

    def grade(self) -> float:
        env = self.env
        state = env.state
        history = state.action_history

        score = 0.0

        # (0.30) Did agent get referrals THEN apply with them?
        # Must have: request_referral succeeded, then apply_with_referral at same company
        referral_companies = set(state.granted_referrals.keys())
        applied_with_ref = [
            a for a in history
            if a.get("action_type") == "apply_with_referral"
            and (a.get("parameters", {}).get("company") or a.get("target")) in referral_companies
        ]
        if len(applied_with_ref) >= 1:
            # Full credit: obtained referral AND used it properly
            score += 0.30
        elif len(referral_companies) >= 1:
            # Partial: got referral but didn't apply with it
            score += 0.10

        # (0.25) Did it run >= 2 CONCURRENT pipelines at actively hiring companies?
        # Must be at companies that are actually hiring (not random spam)
        hiring_companies = {
            name for name, d in env.market.companies.items()
            if d["true_hiring_state"]
        }
        pipeline_at_hiring = sum(
            1 for p in env.pipeline.processes
            if p.company in hiring_companies
            and p.stage not in (InterviewStage.REJECTED,)
        )
        if state.peak_parallel_pipelines >= 3 and pipeline_at_hiring >= 2:
            score += 0.25
        elif state.peak_parallel_pipelines >= 2 and pipeline_at_hiring >= 2:
            score += 0.20

        # (0.25) Did it accept an offer with salary >= 85% of market rate?
        if state.accepted_offer:
            ratio = state.accepted_offer.base_salary / max(state.market_rate, 1)
            if ratio >= 0.9:
                score += 0.25
            elif ratio >= 0.8:
                score += 0.15

        # (0.20) Network warmth: must have IMPROVED warmth, not just preserved
        nh_result = grade_network_health(state, env.network)
        nh = nh_result["score"]
        if nh >= 1.1:
            score += 0.20  # genuinely improved the network
        elif nh >= 0.95:
            score += 0.10

        return min(1.0, score)

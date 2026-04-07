from tasks.base import BaseTask
from env.models import InterviewProcess, InterviewStage, Offer
from env.reward import compute_reward


class Task3(BaseTask):
    """The Competing Pressures — HARD.

    One exploding offer at 70% market rate (deadline step 4).
    Two mid-stage processes at strong companies.
    One key contact went cold. Financial runway: only 5 steps.
    Target score: 0.15-0.35.
    """
    SEED = 303
    TASK_ID = 3
    MAX_STEPS = 30

    def _configure_scenario(self):
        env = self.env
        state = env.state

        state.financial_runway = 5.0
        state.max_steps = 30
        state.market_rate = env.market.get_market_rate()

        state.profile = {
            "skills": {"ml": 0.80, "backend": 0.70, "frontend": 0.30},
            "experience_months": 6,
            "seniority": "junior",
        }

        # Pre-loaded: one OFFER at mediocre company (70% market rate)
        mediocre_salary = state.market_rate * 0.70
        mediocre_offer = Offer(
            company="CloudBase",
            base_salary=round(mediocre_salary, 2),
            equity=round(mediocre_salary * 0.02, 2),
            deadline_step=4,
            negotiation_state="initial",
            culture_fit=0.4,
            role_growth_potential=0.3,
        )
        state.offers.append(mediocre_offer)
        state.first_offer_received = mediocre_offer

        if "CloudBase" in env.market.companies:
            env.market.companies["CloudBase"]["true_hiring_state"] = True
            env.market.companies["CloudBase"]["hiring_signal"] = 0.65

        # Two mid-stage processes at strong companies
        tech_process = InterviewProcess(
            company="TechCorp",
            stage=InterviewStage.TECHNICAL,
            created_step=0,
            expiry_step=8,
            referral_used=False,
        )
        env.pipeline.processes.append(tech_process)

        final_process = InterviewProcess(
            company="MegaSoft",
            stage=InterviewStage.FINAL,
            created_step=0,
            expiry_step=7,
            referral_used=True,
        )
        env.pipeline.processes.append(final_process)

        for name in ("TechCorp", "MegaSoft"):
            if name in env.market.companies:
                env.market.companies[name]["true_hiring_state"] = True

        # Key contact went cold
        if "Jack Brown" in env.network.graph:
            env.network.graph.nodes["Jack Brown"]["warmth"] = 0.30
            env.network.graph.nodes["Jack Brown"]["referral_willingness_signal"] = 0.28

        for name in ("Alice Chen", "Bob Martinez"):
            if name in env.network.graph:
                env.network.graph.nodes[name]["warmth"] = 0.45

        state.initial_warmth_snapshot = env.network.snapshot_warmth()

    def grade(self) -> float:
        env = self.env
        state = env.state
        history = state.action_history

        score = 0.0

        # (0.25) Strategic delay: was REQUEST_DELAY used on CloudBase in steps 0-2?
        # This is the key non-obvious move — buy time on the exploding offer.
        early_delays_on_cloudbase = sum(
            1 for i, a in enumerate(history)
            if a.get("action_type") == "request_delay"
            and (a.get("parameters", {}).get("company") or a.get("target")) == "CloudBase"
            and i < 3
        )
        if early_delays_on_cloudbase >= 1:
            score += 0.25

        # (0.25) Acceleration: did agent advance BOTH promising pipelines?
        advanced_companies = set()
        for a in history:
            at = a.get("action_type", "")
            if at in ("accelerate_process", "advance_round"):
                company = a.get("parameters", {}).get("company") or a.get("target")
                if company in ("TechCorp", "MegaSoft"):
                    advanced_companies.add(company)
        if len(advanced_companies) >= 2:
            score += 0.25
        elif len(advanced_companies) >= 1:
            score += 0.10

        # (0.30) Final offer quality: did agent end up with a BETTER offer than
        # the mediocre 70% one? Must beat the initial CloudBase offer.
        if state.accepted_offer:
            mediocre_salary = state.market_rate * 0.70
            if state.accepted_offer.base_salary > mediocre_salary * 1.15:
                # Accepted a significantly better offer
                score += 0.30
            elif state.accepted_offer.base_salary > mediocre_salary * 1.0:
                # Accepted a slightly better offer (or negotiated the mediocre one up)
                score += 0.15
            elif state.accepted_offer.company != "CloudBase":
                # At least didn't just accept the bad one
                score += 0.10

        # (0.20) Did NOT panic-accept the mediocre offer as very first action
        if history:
            first = history[0]
            if first.get("action_type") == "accept_offer":
                # Panic accept — zero points for this component
                pass
            elif state.accepted_offer:
                # Survived and accepted something
                score += 0.20
            else:
                # Ran out of steps without accepting — partial credit for trying
                score += 0.05

        return min(1.0, score)

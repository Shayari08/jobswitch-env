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
                env.market.companies[name]["hiring_signal"] = min(
                    1.0, 0.85 + env.market.rng.gauss(0, 0.05)
                )

        # Key contact went cold — Jack Brown at MegaSoft
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
        else:
            # Partial: any delay on CloudBase (even slightly late)
            any_cloudbase_delay = any(
                a.get("action_type") == "request_delay"
                and (a.get("parameters", {}).get("company") or a.get("target")) == "CloudBase"
                for a in history
            )
            if any_cloudbase_delay:
                score += 0.10

        # (0.25) Pipeline management: did agent advance BOTH promising pipelines?
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
            score += 0.12
        elif any(a.get("action_type") == "advance_round" for a in history):
            score += 0.05  # advanced some pipeline

        # (0.30) Final offer quality: accepted a better offer than the mediocre 70% one?
        if state.accepted_offer:
            mediocre_salary = state.market_rate * 0.70
            accepted_sal = state.accepted_offer.base_salary
            if state.accepted_offer.company != "CloudBase":
                # Accepted from a different, likely better company
                if accepted_sal > mediocre_salary * 1.15:
                    score += 0.30  # significantly better
                else:
                    score += 0.20  # different company, even if similar salary
            elif accepted_sal > mediocre_salary * 1.08:
                # Negotiated the CloudBase offer up meaningfully
                score += 0.15
            elif accepted_sal > mediocre_salary:
                # Slightly improved CloudBase offer
                score += 0.08
            else:
                score += 0.03  # at least got an offer

        # (0.20) Strategic patience: did NOT panic-accept as first action?
        if history:
            first = history[0]
            if first.get("action_type") == "accept_offer":
                # Panic accept — zero points for this component
                pass
            elif state.accepted_offer:
                # Survived and accepted something
                score += 0.20
            else:
                # Ran out of steps — partial credit for strategic attempts
                n_strategic = sum(
                    1 for a in history
                    if a.get("action_type") in (
                        "request_delay", "advance_round", "accelerate_process"
                    )
                )
                score += min(0.10, n_strategic * 0.02)

        return min(1.0, score)

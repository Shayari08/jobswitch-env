import random
import numpy as np

from env.models import (
    Action,
    ActionType,
    EpisodeState,
    InterviewStage,
    Observation,
    Offer,
    Phase,
)
from env.reward import compute_reward
from env.state import PhaseManager
from simulation.network_graph import NetworkGraph
from simulation.job_market import JobMarket
from simulation.pipeline import PipelineManager
from simulation.negotiation import NegotiationEngine
from simulation.profile import CandidateProfile


# Phase gating: minimum step required for each action type
ACTION_MIN_STEP = {
    ActionType.RESEARCH_COMPANY: 0,
    ActionType.ADD_TO_TARGETS: 0,
    ActionType.REMOVE_TARGET: 0,
    ActionType.WAIT_FOR_SIGNAL: 0,
    ActionType.ENGAGE_CONTENT: 0,
    ActionType.SEND_MESSAGE: 0,
    ActionType.REQUEST_INTRO: 0,
    ActionType.REQUEST_REFERRAL: 2,
    ActionType.TAILOR_RESUME: 2,
    ActionType.APPLY_COLD: 3,
    ActionType.APPLY_WITH_REFERRAL: 3,
    ActionType.ADVANCE_ROUND: 0,  # gated by having active pipeline
    ActionType.REQUEST_DELAY: 0,  # gated by having active pipeline/offer
    ActionType.ACCELERATE_PROCESS: 0,  # gated by having active pipeline
    ActionType.DROP_PROCESS: 0,  # gated by having active pipeline
    ActionType.COUNTER_OFFER: 0,  # gated by having offers
    ActionType.ACCEPT_OFFER: 0,  # gated by having offers
}

# Actions that require an active pipeline at the target company
PIPELINE_REQUIRED_ACTIONS = {
    ActionType.ADVANCE_ROUND,
    ActionType.ACCELERATE_PROCESS,
    ActionType.DROP_PROCESS,
}

# Actions that require an offer from the target company
OFFER_REQUIRED_ACTIONS = {
    ActionType.COUNTER_OFFER,
    ActionType.ACCEPT_OFFER,
}


class JobSwitchEnvironment:
    """Server-side environment. Manages simulation state and processes actions."""

    def __init__(self):
        self.state: EpisodeState = EpisodeState()
        self.network: NetworkGraph | None = None
        self.market: JobMarket | None = None
        self.pipeline: PipelineManager | None = None
        self.negotiation: NegotiationEngine | None = None
        self.profile: CandidateProfile | None = None
        self.rng: random.Random = random.Random(42)
        self.task_id: int = 1

    async def reset(self, seed: int = 42, task_id: int = 1, **kwargs) -> dict:
        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.task_id = task_id
        self.state = EpisodeState()
        self.state.max_steps = kwargs.get("max_steps", 30)
        self.network = NetworkGraph(seed=seed)
        self.market = JobMarket(seed=seed)
        self.pipeline = PipelineManager(seed=seed)
        self.negotiation = NegotiationEngine(seed=seed)
        self.profile = CandidateProfile(seed=seed)

        self.state.market_rate = self.market.get_market_rate()
        self.state.initial_warmth_snapshot = self.network.snapshot_warmth()
        self.state.social_capital = self.network.social_capital

        return self._build_observation().model_dump()

    async def step(self, action: dict) -> dict:
        act = Action(**action)

        # Execute the action — handlers return (step_reward, done)
        step_reward, done = self._execute_action(act)

        self.state.step += 1
        self.state.action_history.append(action)

        # Cap accumulated intermediate rewards at 30% of (terminal reward budget)
        self.state.cumulative_intermediate_reward += step_reward
        max_intermediate = 0.3
        if self.state.cumulative_intermediate_reward > max_intermediate:
            step_reward -= (self.state.cumulative_intermediate_reward - max_intermediate)
            self.state.cumulative_intermediate_reward = max_intermediate

        # Regenerate social capital each step
        self.state.social_capital = min(
            1.0, self.state.social_capital + self.state.social_capital_regen
        )
        self.network.social_capital = self.state.social_capital

        # Check pipeline expirations
        expired = self.pipeline.check_expirations(self.state.step)
        for company in expired:
            step_reward -= 0.03

        # Check offer deadline expirations (#2)
        expired_offers = []
        for offer in self.state.offers:
            if self.state.step >= offer.deadline_step:
                expired_offers.append(offer)
        for offer in expired_offers:
            self.state.offers.remove(offer)
            step_reward -= 0.03

        # Check urgent company deadlines (#15 — used by Task 2)
        if self.market:
            for name, data in self.market.companies.items():
                deadline = data.get("urgent_deadline")
                if deadline and self.state.step >= deadline:
                    data["true_hiring_state"] = False
                    data["hiring_signal"] = max(0.0, data["hiring_signal"] - 0.5)
                    data.pop("urgent_deadline", None)

        # Update peak parallel pipelines
        active_count = len(self.pipeline.get_active_processes())
        self.state.peak_parallel_pipelines = max(
            self.state.peak_parallel_pipelines, active_count
        )

        self._update_phase()

        # Financial runway decreases each step
        self.state.financial_runway = max(0, self.state.financial_runway - 1)

        # Financial runway exhausted — force episode end
        if self.state.financial_runway <= 0 and not self.state.accepted_offer:
            done = True

        # Check if done
        if self.state.step >= self.state.max_steps:
            done = True
        if self.state.done:
            done = True

        # Compute terminal reward if done
        terminal_reward = 0.0
        if done:
            reward_result = compute_reward(self.state, self.network)
            terminal_reward = reward_result.score

        # Build observation once (#13)
        obs = self._build_observation()

        return {
            "observation": obs.model_dump(),
            "reward": terminal_reward if done else step_reward,
            "done": done,
            "info": {
                "step": self.state.step,
                "task": self.task_id,
                "step_reward": step_reward,
            },
        }

    async def get_state(self) -> dict:
        return {"step": self.state.step, "task_id": self.task_id}

    # ─── Observation builder ─────────────────────────────────────

    def _compute_valid_actions(self) -> list:
        """Compute which action types are currently executable given state."""
        has_active_pipeline = bool(self.pipeline.get_active_processes()) if self.pipeline else False
        has_offer = bool(self.state.offers)

        valid = []
        for at in ActionType:
            min_step = ACTION_MIN_STEP.get(at, 0)
            if self.state.step < min_step:
                continue
            if at in PIPELINE_REQUIRED_ACTIONS and not has_active_pipeline:
                continue
            if at in OFFER_REQUIRED_ACTIONS and not has_offer:
                continue
            valid.append(at.value)
        return valid

    def _build_observation(self) -> Observation:
        return Observation(
            step=self.state.step,
            current_phase=self._phase_to_int(self.state.phase),
            target_companies=self.market.get_observable_companies()
            if self.market else [],
            network_graph=self.network.get_observable_graph()
            if self.network else {},
            social_capital=round(self.state.social_capital, 2),
            active_pipeline=self.pipeline.get_observable_pipeline()
            if self.pipeline else [],
            offers_in_hand=[
                {
                    "company": o.company,
                    "base_salary": o.base_salary,
                    "equity": o.equity,
                    "signing_bonus": o.signing_bonus,
                    "deadline_step": o.deadline_step,
                    "negotiation_state": o.negotiation_state,
                }
                for o in self.state.offers
            ],
            profile_strength=self.state.profile,
            financial_runway=self.state.financial_runway,
            market_signals=self.market.get_market_signals()
            if self.market else {},
            time_pressure=round(
                self.state.step / max(self.state.max_steps, 1), 2
            ),
            action_history=[
                a if isinstance(a, dict) else a
                for a in self.state.action_history[-10:]
            ],
            granted_referrals=dict(self.state.granted_referrals),
            valid_actions=self._compute_valid_actions(),
        )

    def _phase_to_int(self, phase: Phase) -> int:
        mapping = {
            Phase.TARGETING: 1, Phase.NETWORK: 2,
            Phase.APPLICATION: 3, Phase.PIPELINE: 4,
            Phase.NEGOTIATION: 5,
        }
        return mapping.get(phase, 1)

    def _update_phase(self):
        # Sync active_processes state for PhaseManager
        self.state.phase = PhaseManager.get_phase(self.state, self.pipeline)

    # ─── Action router ───────────────────────────────────────────

    def _execute_action(self, action: Action) -> tuple[float, bool]:
        """Route action to handler. Returns (step_reward, done).

        Applies phase gating (#9) before dispatching.
        """
        at = action.action_type

        # Phase gating: check minimum step (#9)
        min_step = ACTION_MIN_STEP.get(at, 0)
        if self.state.step < min_step:
            return -0.02, False  # action not yet available

        # Pipeline-required actions: must have active pipeline at company
        if at in PIPELINE_REQUIRED_ACTIONS:
            company = self._get_company(action)
            if not company or not self.pipeline.get_process(company):
                return -0.02, False

        # Offer-required actions: must have offer from company
        if at in OFFER_REQUIRED_ACTIONS:
            company = self._get_company(action)
            if not company or not any(
                o.company == company for o in self.state.offers
            ):
                return -0.02, False

        handlers = {
            ActionType.RESEARCH_COMPANY: self._handle_research_company,
            ActionType.ADD_TO_TARGETS: self._handle_add_to_targets,
            ActionType.REMOVE_TARGET: self._handle_remove_target,
            ActionType.WAIT_FOR_SIGNAL: self._handle_wait_for_signal,
            ActionType.ENGAGE_CONTENT: self._handle_engage_content,
            ActionType.SEND_MESSAGE: self._handle_send_message,
            ActionType.REQUEST_INTRO: self._handle_request_intro,
            ActionType.REQUEST_REFERRAL: self._handle_request_referral,
            ActionType.TAILOR_RESUME: self._handle_tailor_resume,
            ActionType.APPLY_COLD: self._handle_apply_cold,
            ActionType.APPLY_WITH_REFERRAL: self._handle_apply_with_referral,
            ActionType.ADVANCE_ROUND: self._handle_advance_round,
            ActionType.REQUEST_DELAY: self._handle_request_delay,
            ActionType.ACCELERATE_PROCESS: self._handle_accelerate_process,
            ActionType.DROP_PROCESS: self._handle_drop_process,
            ActionType.COUNTER_OFFER: self._handle_counter_offer,
            ActionType.ACCEPT_OFFER: self._handle_accept_offer,
        }
        handler = handlers.get(at)
        if not handler:
            return -0.02, False
        return handler(action)

    # ─── Helpers ─────────────────────────────────────────────────

    def _get_company(self, action: Action) -> str | None:
        return action.parameters.get("company") or action.target

    def _get_person(self, action: Action) -> str | None:
        return action.parameters.get("person") or action.target

    def _check_social_capital_cost(self, cost: float) -> float:
        """Deduct social capital. Return penalty if overdrafted."""
        self.state.social_capital -= cost
        self.network.social_capital = self.state.social_capital
        if self.state.social_capital < 0.1:
            self.state.desperation_signals += 1
            return -0.05
        return 0.0

    def _warmth_crossed_threshold(self, person: str, old_warmth: float,
                                   threshold: float = 0.4) -> bool:
        """Check if warmth crossed a threshold upward after an update (#4)."""
        new_warmth = self.network.get_warmth(person)
        return old_warmth <= threshold < new_warmth

    def _create_offer_from_company(self, company: str,
                                    deadline_offset: int = 5) -> Offer:
        """Create an offer with culture_fit and role_growth from market data."""
        salary_band = self.market.get_salary_band(company)
        salary = self.rng.uniform(salary_band[0], salary_band[1])
        equity = self.rng.uniform(0, salary * 0.15)
        signing = self.rng.uniform(0, salary * 0.05)

        company_data = self.market.get_company(company) or {}
        culture = company_data.get("culture_match", 0.5)
        # Derive growth potential from growth_stage
        stage_growth = {
            "early": 0.9, "growth": 0.7, "mature": 0.4, "enterprise": 0.3,
        }
        growth = stage_growth.get(company_data.get("growth_stage", "growth"), 0.5)

        return Offer(
            company=company,
            base_salary=round(salary, 2),
            equity=round(equity, 2),
            signing_bonus=round(signing, 2),
            deadline_step=self.state.step + deadline_offset,
            culture_fit=culture,
            role_growth_potential=growth,
        )

    # ─── Action Handlers ─────────────────────────────────────────
    # All handlers return (step_reward: float, done: bool).
    # Observation is built once in step(), not in handlers (#13).

    def _handle_research_company(self, action: Action) -> tuple[float, bool]:
        """#1: Research reveals a more accurate hiring signal."""
        company = action.target or action.parameters.get("company")
        if not company or not self.market.get_company(company):
            return -0.02, False

        data = self.market.companies[company]
        true_state = 1.0 if data["true_hiring_state"] else 0.0
        # Move hiring_signal closer to truth (reduce noise)
        old_signal = data["hiring_signal"]
        data["hiring_signal"] = old_signal * 0.4 + true_state * 0.6
        return 0.0, False

    def _handle_add_to_targets(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        if not company or not self.market.get_company(company):
            return -0.02, False
        if company not in self.state.target_companies:
            self.state.target_companies.append(company)
        return 0.0, False

    def _handle_remove_target(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        if company and company in self.state.target_companies:
            self.state.target_companies.remove(company)
        return 0.0, False

    def _handle_wait_for_signal(self, action: Action) -> tuple[float, bool]:
        return 0.0, False

    def _handle_engage_content(self, action: Action) -> tuple[float, bool]:
        person = self._get_person(action)
        if not person or person not in self.network.nodes:
            return -0.02, False

        # #4: Record warmth BEFORE update
        old_warmth = self.network.get_warmth(person)
        self.network.update_warmth(person, 0.05)
        penalty = self._check_social_capital_cost(0.02)

        step_reward = penalty
        if self._warmth_crossed_threshold(person, old_warmth):
            step_reward += 0.03
        return step_reward, False

    def _handle_send_message(self, action: Action) -> tuple[float, bool]:
        person = self._get_person(action)
        if not person or person not in self.network.nodes:
            return -0.02, False

        old_warmth = self.network.get_warmth(person)
        self.network.update_warmth(person, 0.15)
        penalty = self._check_social_capital_cost(0.10)

        step_reward = penalty
        if self._warmth_crossed_threshold(person, old_warmth):
            step_reward += 0.03
        return step_reward, False

    def _handle_request_intro(self, action: Action) -> tuple[float, bool]:
        from_person = action.parameters.get("from_person") or action.target
        to_person = action.parameters.get("to_person")
        if (
            not from_person or not to_person
            or from_person not in self.network.nodes
            or to_person not in self.network.nodes
        ):
            return -0.02, False

        from_warmth = self.network.get_warmth(from_person)
        if from_warmth < 0.3:
            self.state.bridges_burned += 1
            self.network.update_warmth(from_person, -0.15)
            return -0.02, False

        self.network.update_warmth(from_person, -0.05)

        old_warmth = self.network.get_warmth(to_person)
        self.network.update_warmth(to_person, 0.10)
        penalty = self._check_social_capital_cost(0.08)

        step_reward = penalty
        if self._warmth_crossed_threshold(to_person, old_warmth):
            step_reward += 0.03
        return step_reward, False

    def _handle_request_referral(self, action: Action) -> tuple[float, bool]:
        person = self._get_person(action)
        company = action.parameters.get("company") or action.target
        if not person or person not in self.network.nodes:
            return -0.02, False

        warmth = self.network.get_warmth(person)

        if warmth < 0.4:
            self.state.bridges_burned += 1
            self.network.update_warmth(person, -0.20)
            return -0.02, False

        if warmth < 0.6:
            self.network.update_warmth(person, -0.05)
            penalty = self._check_social_capital_cost(0.05)
            return penalty, False

        # Success — referral granted (#3: track it)
        if company:
            self.state.granted_referrals[company] = person
        self.network.update_warmth(person, -0.05)
        penalty = self._check_social_capital_cost(0.10)
        return 0.05 + penalty, False

    def _handle_tailor_resume(self, action: Action) -> tuple[float, bool]:
        company = action.parameters.get("company") or action.target
        role_type = action.parameters.get("role_type", "general")
        if not company:
            return -0.02, False

        improvement = self.profile.tailor_resume(company, role_type)
        # Sync profile state back to episode state
        self.state.ats_scores[company] = self.profile.ats_scores.get(company, 50)
        self.state.resume_versions[company] = role_type

        step_reward = 0.02 if improvement > 10 else 0.0
        return step_reward, False

    def _handle_apply_cold(self, action: Action) -> tuple[float, bool]:
        company = action.parameters.get("company") or action.target
        if not company or not self.market.get_company(company):
            return -0.02, False

        existing = self.pipeline.get_process(company)
        if existing:
            return -0.01, False

        self.pipeline.add_process(company, self.state.step, referral_used=False)
        success, msg = self.pipeline.advance_process(company, self.state.step)
        return (0.04 if success else 0.0), False

    def _handle_apply_with_referral(self, action: Action) -> tuple[float, bool]:
        """#3: Validates that a referral was actually granted for this company."""
        company = action.parameters.get("company") or action.target
        referrer = action.parameters.get("referrer")
        if not company or not self.market.get_company(company):
            return -0.02, False

        existing = self.pipeline.get_process(company)
        if existing:
            return -0.01, False

        # #3: Check if agent actually obtained a referral for this company
        granted_referrer = self.state.granted_referrals.get(company)
        if not granted_referrer:
            # No referral was granted — downgrade to cold application
            self.pipeline.add_process(
                company, self.state.step, referral_used=False
            )
            success, msg = self.pipeline.advance_process(
                company, self.state.step
            )
            return (0.04 if success else 0.0) - 0.01, False

        self.pipeline.add_process(company, self.state.step, referral_used=True)
        success, msg = self.pipeline.advance_process(company, self.state.step)
        return (0.04 if success else 0.0), False

    def _handle_advance_round(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        # Pipeline existence already validated in phase gating
        success, msg = self.pipeline.advance_process(company, self.state.step)
        step_reward = 0.0

        if success:
            step_reward += 0.04
            process = self.pipeline.get_process(company)
            if process and process.stage == InterviewStage.OFFER:
                offer = self._create_offer_from_company(company, deadline_offset=5)
                self.state.offers.append(offer)
                if not self.state.first_offer_received:
                    self.state.first_offer_received = offer
                step_reward += 0.03
        return step_reward, False

    def _handle_request_delay(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        days = action.parameters.get("days", 2)
        if not company:
            return -0.02, False

        # Track delays per company (max 2)
        if not isinstance(self.state.delay_counts, dict):
            self.state.delay_counts = {}
        current_delays = self.state.delay_counts.get(company, 0)
        if current_delays >= 2:
            return -0.02, False  # Already delayed twice

        # Extend deadline on offer
        for offer in self.state.offers:
            if offer.company == company:
                offer.deadline_step += min(days, 3)
                self.state.delay_counts[company] = current_delays + 1
                people = self.network.get_people_at_company(company)
                for p in people:
                    self.network.update_warmth(p, -0.05)
                return 0.0, False

        # Or extend pipeline expiry
        process = self.pipeline.get_process(company)
        if process:
            process.expiry_step += min(days, 3)
            self.state.delay_counts[company] = current_delays + 1
            people = self.network.get_people_at_company(company)
            for p in people:
                self.network.update_warmth(p, -0.05)
            return 0.0, False

        return -0.02, False

    def _handle_accelerate_process(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        # Pipeline existence already validated in phase gating

        accel_count = sum(
            1 for a in self.state.action_history
            if isinstance(a, dict) and a.get("action_type") == "accelerate_process"
        )
        if accel_count >= 2:
            self.state.desperation_signals += 1

        success, msg = self.pipeline.advance_process(company, self.state.step)
        step_reward = 0.0
        if success:
            step_reward += 0.04
            process = self.pipeline.get_process(company)
            if process and process.stage == InterviewStage.OFFER:
                offer = self._create_offer_from_company(company, deadline_offset=4)
                self.state.offers.append(offer)
                if not self.state.first_offer_received:
                    self.state.first_offer_received = offer
                step_reward += 0.03
        return step_reward, False

    def _handle_drop_process(self, action: Action) -> tuple[float, bool]:
        """#6/#10: Abandon an active interview pipeline at a company."""
        company = self._get_company(action)
        # Pipeline existence already validated in phase gating
        process = self.pipeline.get_process(company)
        if process:
            process.stage = InterviewStage.REJECTED
            process.feedback_signals.append("Dropped by candidate")
        return 0.0, False

    def _handle_counter_offer(self, action: Action) -> tuple[float, bool]:
        """#14: Supports negotiating salary, equity, and signing bonus."""
        company = self._get_company(action)
        amount = action.parameters.get("amount", 0)
        components = action.parameters.get("components", {})
        # Offer existence already validated in phase gating

        target_offer = None
        for offer in self.state.offers:
            if offer.company == company:
                target_offer = offer
                break
        if not target_offer:
            return -0.02, False

        # Delegate to NegotiationEngine
        result = self.negotiation.evaluate_counter(
            target_offer,
            ask_salary=amount,
            ask_equity=components.get("equity", 0),
            ask_signing=components.get("signing_bonus", 0),
        )

        if result["collapsed"]:
            self.state.offers.remove(target_offer)
            self.state.desperation_signals += 1
            return -0.05, False

        # Apply negotiation results
        target_offer.base_salary = result["new_salary"]
        target_offer.equity = result["new_equity"]
        target_offer.signing_bonus = result["new_signing"]
        target_offer.counter_offers += 1
        target_offer.negotiation_state = "countered"

        return 0.0, False

    def _handle_accept_offer(self, action: Action) -> tuple[float, bool]:
        company = self._get_company(action)
        # Offer existence already validated in phase gating

        target_offer = None
        for offer in self.state.offers:
            if offer.company == company:
                target_offer = offer
                break
        if not target_offer:
            return -0.02, False

        target_offer.negotiation_state = "accepted"
        self.state.accepted_offer = target_offer
        self.state.done = True

        process = self.pipeline.get_process(company)
        if process:
            process.stage = InterviewStage.ACCEPTED

        return 0.0, True

from pydantic import BaseModel, model_validator, Field
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field as dc_field


class Phase(str, Enum):
    TARGETING = "targeting"
    NETWORK = "network"
    APPLICATION = "application"
    PIPELINE = "pipeline"
    NEGOTIATION = "negotiation"


class ActionType(str, Enum):
    RESEARCH_COMPANY = "research_company"
    ADD_TO_TARGETS = "add_to_targets"
    REMOVE_TARGET = "remove_target"
    WAIT_FOR_SIGNAL = "wait_for_signal"
    ENGAGE_CONTENT = "engage_content"
    SEND_MESSAGE = "send_message"
    REQUEST_INTRO = "request_intro"
    REQUEST_REFERRAL = "request_referral"
    TAILOR_RESUME = "tailor_resume"
    APPLY_COLD = "apply_cold"
    APPLY_WITH_REFERRAL = "apply_with_referral"
    ADVANCE_ROUND = "advance_round"
    REQUEST_DELAY = "request_delay"
    ACCELERATE_PROCESS = "accelerate_process"
    DROP_PROCESS = "drop_process"
    COUNTER_OFFER = "counter_offer"
    ACCEPT_OFFER = "accept_offer"


# Required parameters for each action type
REQUIRED_PARAMS = {
    ActionType.RESEARCH_COMPANY: ["target"],
    ActionType.ADD_TO_TARGETS: ["company"],
    ActionType.REMOVE_TARGET: ["company"],
    ActionType.WAIT_FOR_SIGNAL: [],
    ActionType.ENGAGE_CONTENT: ["person"],
    ActionType.SEND_MESSAGE: ["person", "message_type"],
    ActionType.REQUEST_INTRO: ["from_person", "to_person"],
    ActionType.REQUEST_REFERRAL: ["person", "company"],
    ActionType.TAILOR_RESUME: ["company", "role_type"],
    ActionType.APPLY_COLD: ["company", "role"],
    ActionType.APPLY_WITH_REFERRAL: ["company", "role", "referrer"],
    ActionType.ADVANCE_ROUND: ["company"],
    ActionType.REQUEST_DELAY: ["company", "days"],
    ActionType.ACCELERATE_PROCESS: ["company"],
    ActionType.DROP_PROCESS: ["company"],
    ActionType.COUNTER_OFFER: ["company", "amount"],
    ActionType.ACCEPT_OFFER: ["company"],
}


class Action(BaseModel):
    action_type: ActionType
    target: Optional[str] = None
    parameters: dict = {}
    reasoning: str

    @model_validator(mode="after")
    def validate_params(self):
        required = REQUIRED_PARAMS.get(self.action_type, [])
        # target-based actions use the top-level target field
        for param in required:
            if param == "target":
                if not self.target:
                    raise ValueError(
                        f"{self.action_type.value} requires 'target' to be set"
                    )
            elif param == "person":
                if not self.target and param not in self.parameters:
                    raise ValueError(
                        f"{self.action_type.value} requires 'person' in target or parameters"
                    )
            elif param == "company":
                if not self.target and param not in self.parameters:
                    raise ValueError(
                        f"{self.action_type.value} requires 'company' in target or parameters"
                    )
            else:
                if param not in self.parameters:
                    raise ValueError(
                        f"{self.action_type.value} requires '{param}' in parameters"
                    )
        return self


class Observation(BaseModel):
    step: int
    current_phase: int
    target_companies: list[dict] = []
    network_graph: dict = {}
    social_capital: float = 1.0
    active_pipeline: list[dict] = []
    offers_in_hand: list[dict] = []
    profile_strength: dict = {}
    financial_runway: float = 15.0
    market_signals: dict = {}
    time_pressure: float = 0.0
    action_history: list[dict] = []


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    offer_quality: float = 0.0
    process_efficiency: float = 0.0
    network_health: float = 0.0
    negotiation_quality: float = 0.0
    time_efficiency: float = 0.0
    penalties: dict = {}
    component_scores: dict = {}


class InterviewStage(str, Enum):
    APPLIED = "APPLIED"
    SCREENING = "SCREENING"
    TECHNICAL = "TECHNICAL"
    FINAL = "FINAL"
    OFFER = "OFFER"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass
class CompanyState:
    name: str
    true_hiring_state: bool
    hiring_signal: float
    role_fit_score: float
    salary_band: tuple = (80000, 120000)
    growth_stage: str = "growth"
    culture_match: float = 0.5


@dataclass
class InterviewProcess:
    company: str
    stage: InterviewStage = InterviewStage.APPLIED
    created_step: int = 0
    expiry_step: int = 10
    feedback_signals: list = dc_field(default_factory=list)
    referral_used: bool = False


@dataclass
class Offer:
    company: str
    base_salary: float
    equity: float = 0.0
    signing_bonus: float = 0.0
    deadline_step: int = 30
    negotiation_state: str = "initial"
    counter_offers: int = 0
    culture_fit: float = 0.5
    role_growth_potential: float = 0.5


@dataclass
class EpisodeState:
    """Ground truth state — not shown to agent."""
    step: int = 0
    max_steps: int = 30
    phase: Phase = Phase.TARGETING
    social_capital: float = 1.0
    social_capital_regen: float = 0.05
    financial_runway: float = 15.0
    market_rate: float = 100000.0
    # Network state
    initial_warmth_snapshot: dict = dc_field(default_factory=dict)
    # Pipeline tracking
    peak_parallel_pipelines: int = 0
    # Offers
    offers: list = dc_field(default_factory=list)
    accepted_offer: Optional[Offer] = None
    first_offer_received: Optional[Offer] = None
    # Tracking
    action_history: list = dc_field(default_factory=list)
    bridges_burned: int = 0
    desperation_signals: int = 0
    target_companies: list = dc_field(default_factory=list)
    # Referrals granted: maps "company" -> "referrer_person"
    granted_referrals: dict = dc_field(default_factory=dict)
    # Resume
    resume_versions: dict = dc_field(default_factory=dict)
    ats_scores: dict = dc_field(default_factory=dict)
    # Profile
    profile: dict = dc_field(default_factory=lambda: {
        "skills": {"ml": 0.7, "backend": 0.6, "frontend": 0.3},
        "experience_months": 6,
        "seniority": "junior",
    })
    # Intermediate reward tracking
    cumulative_intermediate_reward: float = 0.0
    # Delay tracking per company
    delay_counts: dict = dc_field(default_factory=dict)
    # Done flag
    done: bool = False

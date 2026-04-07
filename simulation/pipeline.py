import random
from env.models import InterviewProcess, InterviewStage, Offer


PASS_RATES = {
    "APPLIED_cold": {"SCREENING": 0.10},
    "APPLIED_referral": {"SCREENING": 0.60},
    "SCREENING": {"TECHNICAL": 0.55},
    "TECHNICAL": {"FINAL": 0.45},
    "FINAL": {"OFFER": 0.50},
}

EXPIRY_STEPS = {
    "APPLIED": 5,
    "SCREENING": 4,
    "TECHNICAL": 5,
    "FINAL": 4,
}


class PipelineManager:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.processes: list[InterviewProcess] = []

    def add_process(
        self, company: str, current_step: int, referral_used: bool = False
    ) -> InterviewProcess:
        expiry = current_step + EXPIRY_STEPS["APPLIED"]
        process = InterviewProcess(
            company=company,
            stage=InterviewStage.APPLIED,
            created_step=current_step,
            expiry_step=expiry,
            referral_used=referral_used,
        )
        self.processes.append(process)
        return process

    def get_process(self, company: str) -> InterviewProcess | None:
        for p in self.processes:
            if (
                p.company == company
                and p.stage
                not in (InterviewStage.REJECTED, InterviewStage.ACCEPTED)
            ):
                return p
        return None

    def advance_process(self, company: str, current_step: int) -> tuple[bool, str]:
        """Advance interview to next stage. Returns (success, message)."""
        process = self.get_process(company)
        if not process:
            return False, "No active process at this company"

        current_stage = process.stage.value

        if current_stage == "APPLIED":
            key = (
                "APPLIED_referral" if process.referral_used else "APPLIED_cold"
            )
            next_stages = PASS_RATES.get(key, {})
        elif current_stage in ("SCREENING", "TECHNICAL", "FINAL"):
            next_stages = PASS_RATES.get(current_stage, {})
        elif current_stage == "OFFER":
            return False, "Already at OFFER stage — accept or counter"
        else:
            return False, f"Cannot advance from {current_stage}"

        if not next_stages:
            return False, f"No transition from {current_stage}"

        next_stage_name = list(next_stages.keys())[0]
        pass_rate = next_stages[next_stage_name]

        if self.rng.random() < pass_rate:
            process.stage = InterviewStage(next_stage_name)
            # Update expiry
            if next_stage_name in EXPIRY_STEPS:
                process.expiry_step = current_step + EXPIRY_STEPS[next_stage_name]
            else:
                process.expiry_step = current_step + 5
            feedback = f"Advanced to {next_stage_name}"
            process.feedback_signals.append(feedback)
            return True, feedback
        else:
            process.stage = InterviewStage.REJECTED
            feedback = f"Rejected at {current_stage} stage"
            process.feedback_signals.append(feedback)
            return False, feedback

    def create_offer(
        self, company: str, salary: float, equity: float, deadline_step: int
    ) -> Offer:
        """Create an offer for a company that reached OFFER stage."""
        process = self.get_process(company)
        if process:
            process.stage = InterviewStage.OFFER
        return Offer(
            company=company,
            base_salary=salary,
            equity=equity,
            deadline_step=deadline_step,
        )

    def get_active_processes(self) -> list[InterviewProcess]:
        return [
            p
            for p in self.processes
            if p.stage
            not in (
                InterviewStage.REJECTED,
                InterviewStage.ACCEPTED,
                InterviewStage.OFFER,
            )
        ]

    def get_all_non_terminal(self) -> list[InterviewProcess]:
        return [
            p
            for p in self.processes
            if p.stage not in (InterviewStage.REJECTED, InterviewStage.ACCEPTED)
        ]

    def check_expirations(self, current_step: int) -> list[str]:
        """Check and expire processes that have timed out."""
        expired = []
        for process in self.processes:
            if (
                process.stage
                not in (
                    InterviewStage.REJECTED,
                    InterviewStage.ACCEPTED,
                    InterviewStage.OFFER,
                )
                and current_step >= process.expiry_step
            ):
                process.stage = InterviewStage.REJECTED
                process.feedback_signals.append("Expired — no response")
                expired.append(process.company)
        return expired

    def get_observable_pipeline(self) -> list[dict]:
        """Return pipeline info visible to agent."""
        result = []
        for p in self.get_all_non_terminal():
            result.append({
                "company": p.company,
                "current_round": p.stage.value,
                "created_step": p.created_step,
                "expiry_step": p.expiry_step,
                "feedback": p.feedback_signals[-1] if p.feedback_signals else "",
            })
        return result

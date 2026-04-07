"""Episode state management and phase tracking."""

from env.models import EpisodeState, Phase


class PhaseManager:
    """Tracks which phase the episode is in based on state."""

    PHASE_STEP_RANGES = {
        Phase.TARGETING: (0, 5),
        Phase.NETWORK: (3, 12),
        Phase.APPLICATION: (5, 15),
        Phase.PIPELINE: (10, 25),
        Phase.NEGOTIATION: (20, 30),
    }

    @staticmethod
    def get_phase(state: EpisodeState, pipeline=None) -> Phase:
        if state.offers:
            return Phase.NEGOTIATION
        if pipeline and len(pipeline.get_active_processes()) > 0:
            return Phase.PIPELINE
        if state.target_companies and state.step >= 5:
            return Phase.APPLICATION
        if state.step >= 3:
            return Phase.NETWORK
        return Phase.TARGETING

    @staticmethod
    def get_available_phases(step: int) -> list[Phase]:
        """Phases overlap — return all phases active at this step."""
        available = []
        for phase, (start, end) in PhaseManager.PHASE_STEP_RANGES.items():
            if start <= step <= end:
                available.append(phase)
        return available or [Phase.TARGETING]

from env.models import EpisodeState


def grade_process_efficiency(state: EpisodeState) -> dict:
    """Grade how efficiently the agent ran parallel pipelines."""
    peak = state.peak_parallel_pipelines
    score = min(1.0, peak / 3.0)

    return {
        "score": score,
        "peak_parallel_pipelines": peak,
        "total_applications": sum(
            1
            for a in state.action_history
            if isinstance(a, dict)
            and a.get("action_type") in ("apply_cold", "apply_with_referral")
        ),
        "steps_used": state.step,
        "max_steps": state.max_steps,
    }

from env.models import EpisodeState


def grade_network_health(state: EpisodeState, network) -> dict:
    """Grade how well the agent preserved/improved network warmth."""
    if not state.initial_warmth_snapshot:
        return {"score": 0.5, "detail": "No warmth snapshot"}

    start_values = list(state.initial_warmth_snapshot.values())
    avg_start = sum(start_values) / len(start_values) if start_values else 0.01

    current = network.snapshot_warmth()
    end_values = list(current.values())
    avg_end = sum(end_values) / len(end_values) if end_values else 0.0

    ratio = avg_end / max(avg_start, 0.01)

    return {
        "score": min(1.0, max(0.0, ratio)),
        "avg_warmth_start": round(avg_start, 3),
        "avg_warmth_end": round(avg_end, 3),
        "bridges_burned": state.bridges_burned,
    }

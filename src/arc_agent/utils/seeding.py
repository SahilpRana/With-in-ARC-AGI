"""Small helpers for deterministic episode seed schedules."""


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Return the stable seed assigned to a zero-based episode index."""
    if episode_index < 0:
        raise ValueError("episode_index must not be negative")
    return base_seed + episode_index

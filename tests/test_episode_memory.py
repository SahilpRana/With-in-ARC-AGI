"""Trajectory buffer behavior."""

import pytest

from arc_agent.memory.episode_memory import EpisodeMemory, Transition


def test_memory_appends_and_returns_recent_transitions() -> None:
    memory = EpisodeMemory()
    transition = Transition(state={"n": 0}, action={"move": "right"}, next_state={"n": 1})
    memory.add(transition)

    assert len(memory) == 1
    assert memory.recent(1) == (transition,)
    assert memory.recent(0) == ()
    with pytest.raises(ValueError):
        memory.recent(-1)

"""Typed trajectory storage for evidence gathered during one episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Observation = Mapping[str, Any]
Action = Mapping[str, Any]


@dataclass(frozen=True)
class Transition:
    """One observed state-action-next-state transition."""

    state: Observation
    action: Action
    next_state: Observation
    reward: float = 0.0
    terminated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMemory:
    """Append-only trajectory buffer that supports simple evidence queries."""

    transitions: list[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        """Append a transition to the episode trajectory."""
        self.transitions.append(transition)

    def __len__(self) -> int:
        """Return the number of recorded transitions."""
        return len(self.transitions)

    def recent(self, count: int) -> tuple[Transition, ...]:
        """Return up to ``count`` most recent transitions in time order."""
        if count < 0:
            raise ValueError("count must not be negative")
        return tuple(self.transitions[-count:]) if count else ()

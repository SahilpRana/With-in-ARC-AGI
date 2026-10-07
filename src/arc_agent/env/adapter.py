"""API-agnostic environment contract for interactive ARC episodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

Observation = Mapping[str, Any]
Action = Mapping[str, Any]
Spec = Mapping[str, Any]


@dataclass(frozen=True)
class StepResult:
    """Normalized result of one environment action."""

    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Abstract observe-act environment boundary used by the agent."""

    @property
    @abstractmethod
    def observation_spec(self) -> Spec:
        """Describe the normalized observation fields provided by this environment."""

    @property
    @abstractmethod
    def action_spec(self) -> Spec:
        """Describe the actions accepted by this environment."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Start an episode and return its initial observation."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply one action and return the resulting observation and status."""


class CompetitionEnvironment(Environment):
    """Placeholder for a future competition API integration.

    Implement this adapter only after the official API contract is available.
    Normalize its observations and actions to the interface above; do not infer
    or assume transport, authentication, or payload details here.
    """

    @property
    def observation_spec(self) -> Spec:
        raise NotImplementedError("Implement after the competition API is available")

    @property
    def action_spec(self) -> Spec:
        raise NotImplementedError("Implement after the competition API is available")

    def reset(self, seed: int | None = None) -> Observation:
        raise NotImplementedError("Implement after the competition API is available")

    def step(self, action: Action) -> StepResult:
        raise NotImplementedError("Implement after the competition API is available")

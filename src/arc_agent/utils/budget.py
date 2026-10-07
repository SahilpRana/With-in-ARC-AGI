"""Deterministic step/compute limits and wall-clock episode budget tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class Budget:
    """Track resource use and report when any configured limit is exhausted."""

    max_steps: int = 100
    max_seconds: float = 60.0
    max_compute_units: int = 10_000
    steps_used: int = 0
    compute_units_used: int = 0
    _started_at: float = field(default_factory=monotonic, repr=False)

    def __post_init__(self) -> None:
        """Validate resource limits."""
        if self.max_steps < 0 or self.max_seconds < 0 or self.max_compute_units < 0:
            raise ValueError("budget limits must not be negative")

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed monotonic wall-clock time."""
        return monotonic() - self._started_at

    @property
    def exhausted(self) -> bool:
        """Return whether a configured resource limit has been reached."""
        return (
            self.steps_used >= self.max_steps
            or self.compute_units_used >= self.max_compute_units
            or self.elapsed_seconds >= self.max_seconds
        )

    def consume_step(self, compute_units: int = 1) -> None:
        """Account for one action and its approximate compute cost."""
        if compute_units < 0:
            raise ValueError("compute_units must not be negative")
        self.steps_used += 1
        self.consume_compute(compute_units)

    def consume_compute(self, compute_units: int) -> None:
        """Account for strategy/model work in abstract compute units."""
        if compute_units < 0:
            raise ValueError("compute_units must not be negative")
        self.compute_units_used += compute_units

    def remaining_steps(self) -> int:
        """Return the non-negative action allowance remaining."""
        return max(0, self.max_steps - self.steps_used)

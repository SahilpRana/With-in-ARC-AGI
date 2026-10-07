"""Action-conditioned candidate mechanics and evidence summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Hypothesis:
    """Candidate prediction for one action and its observed transition evidence.

    Confidence is the mean of a Beta(1 + supports, 1 + contradictions)
    posterior. No-op observations count as tests but are not interpreted as
    support or contradiction for a movement prediction.
    """

    action: str
    expected_displacement: tuple[int, int]
    supports: int = 0
    contradictions: int = 0
    no_ops: int = 0
    evidence: list[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Return the Beta posterior mean for non-no-op observations."""
        return (self.supports + 1) / (self.supports + self.contradictions + 2)

    @property
    def tested_count(self) -> int:
        """Return all observations of this action, including no-ops."""
        return self.supports + self.contradictions + self.no_ops

    @property
    def rule(self) -> str:
        """Render the candidate as a concise action-to-displacement rule."""
        dx, dy = self.expected_displacement
        return f"{self.action!r} -> displacement ({dx}, {dy})"

    @property
    def posterior(self) -> tuple[int, int]:
        """Return Beta posterior alpha and beta parameters."""
        return self.supports + 1, self.contradictions + 1

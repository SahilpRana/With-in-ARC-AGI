"""Aggregate task-agnostic episode metrics for experiments and ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from arc_agent.agent.agent import EpisodeResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """Compact evaluation summary across independent episodes."""

    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    steps_to_solve: tuple[int, ...]


def summarize(results: Sequence[EpisodeResult]) -> EvaluationMetrics:
    """Compute success rate and successful-episode sample efficiency."""
    count = len(results)
    successes = sum(result.success for result in results)
    solved_steps = tuple(result.steps for result in results if result.success)
    return EvaluationMetrics(
        episodes=count,
        successes=successes,
        success_rate=successes / count if count else 0.0,
        mean_steps=sum(result.steps for result in results) / count if count else 0.0,
        steps_to_solve=solved_steps,
    )

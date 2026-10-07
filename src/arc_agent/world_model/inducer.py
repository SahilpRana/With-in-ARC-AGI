"""Propose action-conditioned displacement rules from an action vocabulary."""

from __future__ import annotations

from collections.abc import Sequence

from arc_agent.memory.episode_memory import EpisodeMemory
from arc_agent.world_model.hypothesis import Hypothesis

CARDINAL_DISPLACEMENTS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class RuleInducer:
    """Create candidate displacement hypotheses for each symbolic action."""

    def __init__(
        self,
        actions: Sequence[str] = ("up", "down", "left", "right"),
        displacement_candidates: Sequence[tuple[int, int]] = CARDINAL_DISPLACEMENTS,
    ) -> None:
        """Configure the discrete action and displacement candidate spaces."""
        if not actions or not displacement_candidates:
            raise ValueError("actions and displacement_candidates must not be empty")
        self.actions = tuple(actions)
        self.displacement_candidates = tuple(displacement_candidates)

    def propose(self, memory: EpisodeMemory | None = None) -> list[Hypothesis]:
        """Return one candidate rule for each action/displacement combination."""
        del memory  # Evidence is applied by RuleVerifier to keep proposals stable.
        return [
            Hypothesis(action=action, expected_displacement=displacement)
            for action in self.actions
            for displacement in self.displacement_candidates
        ]


# TODO: Add program synthesis over transition examples.
# TODO: Add an optional LLM proposer behind a provider interface and record prompts.

"""Decision logic for switching between baseline exploration and planning."""

from __future__ import annotations

from enum import Enum
from collections.abc import Sequence
from typing import Protocol

from arc_agent.env.adapter import Action
from arc_agent.world_model.hypothesis import Hypothesis


class Explorer(Protocol):
    """Common action-selection surface for configurable explorers."""

    def choose(self, hypotheses: Sequence[Hypothesis] = (), **context: object) -> Action: ...


def evidence_gate_open(
    hypotheses: Sequence[Hypothesis],
    actions: Sequence[str],
    *,
    min_action_tests: int,
    confidence_threshold: float,
) -> bool:
    """Require every action to be tested k times with a confident candidate."""
    if min_action_tests < 0 or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("invalid action-test gate settings")
    for action in actions:
        candidates = [item for item in hypotheses if item.action == action]
        if not candidates:
            return False
        if min(item.tested_count for item in candidates) < min_action_tests:
            return False
        if max(item.confidence for item in candidates) < confidence_threshold:
            return False
    return bool(actions)


class Mode(str, Enum):
    """Current high-level behavior selected by the policy."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class AgentPolicy:
    """Prefer a verified plan; otherwise spend the action on exploration."""

    def select_action(
        self,
        plan: tuple[Action, ...] | None,
        hypotheses_confident: bool,
        explorer: Explorer,
        hypotheses: Sequence[Hypothesis] = (),
        *,
        current_position: tuple[int, int] | None = None,
        visited_positions: set[tuple[int, int]] | None = None,
        blocked_actions_here: set[str] | None = None,
        forbidden_actions: set[str] | None = None,
        grid: tuple[tuple[int, ...], ...] | None = None,
        grid_origin: tuple[int, int] = (0, 0),
    ) -> tuple[Mode, Action]:
        """Choose the next action and mode, using a plan when one is available."""
        if plan and hypotheses_confident:
            return Mode.EXPLOIT, plan[0]
        return Mode.EXPLORE, explorer.choose(
            hypotheses,
            current_position=current_position,
            visited_positions=visited_positions,
            blocked_actions_here=blocked_actions_here,
            forbidden_actions=forbidden_actions,
            grid=grid,
            grid_origin=grid_origin,
        )

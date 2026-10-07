"""Exploration strategies that choose actions when the model is uncertain."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Sequence

from arc_agent.env.adapter import Action
from arc_agent.world_model.hypothesis import Hypothesis


class NoPermittedActionError(RuntimeError):
    """Raised when every available action is a remembered no-op at this state."""


def observation_signature(observation: object) -> tuple[object, ...]:
    """Normalize player identity out of a grid key but retain map and position."""
    if not hasattr(observation, "get"):
        return (repr(observation),)
    get = observation.get
    grid = get("grid")
    normalized_grid = None
    if grid is not None:
        normalized_grid = tuple(
            tuple(0 if int(tile) == 2 else int(tile) for tile in row)
            for row in grid
        )
    player = get("player")
    origin = get("window_origin", (0, 0))
    return (
        normalized_grid,
        tuple(player) if player is not None else None,
        tuple(origin) if origin is not None else None,
    )


class RandomExplorer:
    """Seeded baseline explorer over a finite action set."""

    def __init__(self, actions: Sequence[Action], seed: int = 0) -> None:
        """Initialize the random policy with reproducible action ordering."""
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions = tuple(actions)
        self._rng = random.Random(seed)

    def choose(
        self,
        hypotheses: Sequence[Hypothesis] = (),
        **context: object,
    ) -> Action:
        """Choose one action uniformly at random."""
        del hypotheses
        forbidden = set(context.get("forbidden_actions", ()))
        candidates = [
            action for action in self._actions
            if str(action.get("move", "")) not in forbidden
        ]
        if not candidates:
            raise NoPermittedActionError("all actions are remembered no-ops at this state")
        return self._rng.choice(candidates)


class InformationGainExplorer:
    """Choose actions with maximal disagreement among live predictions."""

    def __init__(self, actions: Sequence[Action], seed: int = 0) -> None:
        """Initialize deterministic tie-breaking over the available actions."""
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions = tuple(actions)
        self._rng = random.Random(seed)

    def choose(
        self,
        hypotheses: Sequence[Hypothesis],
        *,
        current_position: tuple[int, int] | None = None,
        visited_positions: set[tuple[int, int]] | None = None,
        blocked_actions_here: set[str] | None = None,
        forbidden_actions: set[str] | None = None,
        **context: object,
    ) -> Action:
        """Prefer predicted novel cells; use disagreement only as a tie-break."""
        del context
        by_action: dict[str, set[tuple[int, int]]] = defaultdict(set)
        evidence_by_action: dict[str, list[Hypothesis]] = defaultdict(list)
        tested: dict[str, int] = defaultdict(int)
        names_by_action: dict[str, Action] = {}
        for action in self._actions:
            name = str(action.get("move", action.get("action", "")))
            names_by_action[name] = action
        for hypothesis in hypotheses:
            evidence_by_action[hypothesis.action].append(hypothesis)
            if hypothesis.confidence >= 0.25:
                by_action[hypothesis.action].add(hypothesis.expected_displacement)
            tested[hypothesis.action] = max(tested[hypothesis.action], hypothesis.tested_count)
        visited = visited_positions or set()
        blocked = blocked_actions_here or set()
        forbidden = forbidden_actions or set()
        scores: dict[str, tuple[int, int, int]] = {}
        for name in names_by_action:
            if name in forbidden:
                continue
            candidates = evidence_by_action[name]
            prediction = max(candidates, key=lambda item: (item.confidence, item.supports), default=None)
            expected = prediction.expected_displacement if prediction else (0, 0)
            destination = (
                (current_position[0] + expected[0], current_position[1] + expected[1])
                if current_position is not None
                else None
            )
            novelty = int(destination is not None and destination not in visited)
            # The order is intentional: coverage first, hypothesis disagreement
            # only among equally novel actions, then avoid blocked repeats.
            scores[name] = (
                novelty,
                len(by_action[name]),
                -int(name in blocked) * (tested[name] + 1) - tested[name],
            )
        if not scores:
            raise NoPermittedActionError("all actions are remembered no-ops at this state")
        best_score = max(scores.values())
        best = [name for name, score in scores.items() if score == best_score]
        return names_by_action[self._rng.choice(best)]


class FrontierExplorer:
    """Use BFS with the learned movement model to reach the nearest novel cell."""

    def __init__(self, actions: Sequence[Action], seed: int = 0) -> None:
        """Create a frontier policy with deterministic fallback tie-breaking."""
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions = tuple(actions)
        self._fallback = InformationGainExplorer(actions, seed=seed)

    def choose(
        self,
        hypotheses: Sequence[Hypothesis] = (),
        *,
        current_position: tuple[int, int] | None = None,
        visited_positions: set[tuple[int, int]] | None = None,
        blocked_actions_here: set[str] | None = None,
        forbidden_actions: set[str] | None = None,
        grid: tuple[tuple[int, ...], ...] | None = None,
        grid_origin: tuple[int, int] = (0, 0),
        **context: object,
    ) -> Action:
        """Return the first action toward the nearest unvisited reachable cell."""
        del context
        if current_position is None or grid is None:
            return self._fallback.choose(
                hypotheses,
                current_position=current_position,
                visited_positions=visited_positions,
                blocked_actions_here=blocked_actions_here,
                forbidden_actions=forbidden_actions,
            )
        visited = visited_positions or set()
        blocked = blocked_actions_here or set()
        forbidden = forbidden_actions or set()
        origin_x, origin_y = grid_origin
        height = len(grid)
        width = len(grid[0]) if height else 0

        def is_open(position: tuple[int, int]) -> bool:
            x, y = position[0] - origin_x, position[1] - origin_y
            return 0 <= x < width and 0 <= y < height and grid[y][x] != 1

        candidates_by_action: dict[str, list[Hypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            candidates_by_action[hypothesis.action].append(hypothesis)
        deltas: dict[str, tuple[int, int]] = {}
        for action in self._actions:
            name = str(action.get("move", ""))
            candidates = candidates_by_action.get(name, [])
            if candidates:
                best = max(candidates, key=lambda item: (item.confidence, item.supports))
                deltas[name] = best.expected_displacement
        if not deltas:
            deltas = {
                "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)
            }

        queue = deque([(current_position, None)])
        seen = {current_position}
        while queue:
            position, first_action = queue.popleft()
            if position not in visited and position != current_position and first_action is not None:
                return first_action
            for name, (dx, dy) in deltas.items():
                if position == current_position and (name in blocked or name in forbidden):
                    continue
                destination = (position[0] + dx, position[1] + dy)
                if destination in seen or not is_open(destination):
                    continue
                seen.add(destination)
                action = {"move": name}
                queue.append((destination, first_action or action))
        return self._fallback.choose(
            hypotheses,
            current_position=current_position,
            visited_positions=visited,
            blocked_actions_here=blocked,
            forbidden_actions=forbidden,
        )


def is_oscillating(
    positions: Sequence[tuple[int, int]],
    *,
    window: int = 8,
    max_distinct: int = 3,
) -> bool:
    """Detect repeated cycling when a recent position window has low coverage."""
    if window < 2 or max_distinct < 1:
        raise ValueError("window and max_distinct must be positive")
    recent = positions[-window:]
    return len(recent) == window and len(set(recent)) <= max_distinct


# TODO: Replace disagreement count with expected entropy reduction when
# calibrated transition likelihoods are available.

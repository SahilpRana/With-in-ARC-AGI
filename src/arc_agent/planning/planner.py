"""Short-horizon planning over the currently encoded grid observation."""

from __future__ import annotations

from collections import deque
from typing import Sequence

from arc_agent.env.adapter import Action
from arc_agent.perception.state_encoder import EncodedState
from arc_agent.world_model.hypothesis import Hypothesis


class GridPlanner:
    """Find a shortest cardinal route through currently visible non-wall cells."""

    _DELTAS = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    def plan(
        self,
        state: EncodedState,
        goal: tuple[int, int] | None = None,
        hypotheses: tuple[Hypothesis, ...] = (),
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[Action, ...] | None:
        """Return a shortest path to an inferred goal, if visible structure allows."""
        destination = goal if goal is not None else state.target
        if state.grid is None or state.player is None or destination is None:
            return None
        start = state.player
        if start == destination:
            return ()
        height = len(state.grid)
        width = len(state.grid[0]) if height else 0
        origin_x, origin_y = state.grid_origin
        explicitly_blocked = blocked_cells or set()

        def walkable(position: tuple[int, int]) -> bool:
            x, y = position[0] - origin_x, position[1] - origin_y
            return (
                0 <= x < width
                and 0 <= y < height
                and state.grid[y][x] != 1
                and position not in explicitly_blocked
            )

        action_deltas: dict[str, tuple[int, int]] = {}
        grouped: dict[str, list[Hypothesis]] = {}
        for hypothesis in hypotheses:
            grouped.setdefault(hypothesis.action, []).append(hypothesis)
        for action, candidates in grouped.items():
            best = max(candidates, key=lambda candidate: (candidate.confidence, candidate.supports))
            action_deltas[action] = best.expected_displacement
        if not action_deltas:
            action_deltas = dict(self._DELTAS)
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
            start: None
        }
        while queue:
            current = queue.popleft()
            for move, (dx, dy) in action_deltas.items():
                neighbor = (current[0] + dx, current[1] + dy)
                if not walkable(neighbor) or neighbor in previous:
                    continue
                previous[neighbor] = (current, move)
                if neighbor == destination:
                    return self._reconstruct(previous, destination)
                queue.append(neighbor)
        return None

    @staticmethod
    def _reconstruct(
        previous: dict[tuple[int, int], tuple[tuple[int, int], str] | None],
        goal: tuple[int, int],
    ) -> tuple[Action, ...]:
        moves: list[Action] = []
        current = goal
        while previous[current] is not None:
            prior, move = previous[current]  # type: ignore[misc]
            moves.append({"move": move})
            current = prior
        moves.reverse()
        return tuple(moves)


# TODO: Evaluate test-time search or MCTS against this transparent baseline.

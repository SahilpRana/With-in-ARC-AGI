"""Seeded grid-world with a hidden target rule for local agent experiments."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from .adapter import Action, Environment, Observation, Spec, StepResult


class VisibleGridEnv(Environment):
    """Grid navigation task whose target location varies by episode seed.

    The agent sees a grid with a player, target, and walls. It must reach the
    target using cardinal move actions. Observations are immutable snapshots.
    """

    _MOVES = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    def __init__(self, width: int = 5, height: int = 5, max_steps: int = 40) -> None:
        """Create a grid task; dimensions and step limit must be positive."""
        if width < 2 or height < 2:
            raise ValueError("width and height must both be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self._rng = random.Random()
        self._player = (0, 0)
        self._target = (width - 1, height - 1)
        self._walls: set[tuple[int, int]] = set()
        self._steps = 0
        self._done = False

    @property
    def observation_spec(self) -> Spec:
        """Return fields and coordinate conventions for observations."""
        return {
            "grid": "height x width; 0 empty, 1 wall, 2 player, 3 target",
            "player": "(x, y), origin at top-left",
            "target": "(x, y), origin at top-left",
            "steps": "number of actions applied",
        }

    @property
    def action_spec(self) -> Spec:
        """Return the accepted cardinal movement action format."""
        return {"move": tuple(self._MOVES)}

    def reset(self, seed: int | None = None) -> Observation:
        """Initialize a deterministic layout and return its first observation."""
        self._rng.seed(seed)
        cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        start = (0, 0)
        target_candidates = [cell for cell in cells if cell != start]
        self._target = self._rng.choice(target_candidates)
        self._player = start
        # Sparse walls keep the mock reachable without encoding a path solution.
        candidates = [cell for cell in cells if cell not in (start, self._target)]
        self._walls = {
            cell for cell in candidates if self._rng.random() < 0.12
        }
        route = [start]
        x, y = start
        while x != self._target[0]:
            x += 1 if self._target[0] > x else -1
            route.append((x, y))
        while y != self._target[1]:
            y += 1 if self._target[1] > y else -1
            route.append((x, y))
        self._walls.difference_update(route)
        self._steps = 0
        self._done = False
        return self._observation()

    def step(self, action: Action) -> StepResult:
        """Apply a movement, reporting invalid or blocked moves without motion."""
        if self._done:
            raise RuntimeError("episode is finished; call reset() before step()")
        move = action.get("move")
        if move not in self._MOVES:
            raise ValueError(f"unknown move {move!r}; expected one of {tuple(self._MOVES)}")

        dx, dy = self._MOVES[str(move)]
        candidate = (self._player[0] + dx, self._player[1] + dy)
        if (
            0 <= candidate[0] < self.width
            and 0 <= candidate[1] < self.height
            and candidate not in self._walls
        ):
            self._player = candidate

        self._steps += 1
        success = self._player == self._target
        truncated = self._steps >= self.max_steps and not success
        self._done = success or truncated
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=success,
            truncated=truncated,
            info={"success": success},
        )

    def _observation(self) -> Observation:
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self._walls:
            grid[y][x] = 1
        target_x, target_y = self._target
        grid[target_y][target_x] = 3
        player_x, player_y = self._player
        grid[player_y][player_x] = 2
        return {
            "grid": tuple(tuple(row) for row in grid),
            "player": self._player,
            "target": self._target,
            "steps": self._steps,
        }


@dataclass(frozen=True)
class HiddenDifficulty:
    """Select hidden mechanics without exposing their sampled values."""

    action_remapping: bool = True
    hidden_goal: bool = True
    mechanic: str | None = None
    partial_observability: bool = False
    view_radius: int = 2

    def __post_init__(self) -> None:
        if self.mechanic not in (None, "key_door", "one_way", "button"):
            raise ValueError("mechanic must be key_door, one_way, button, or None")
        if self.view_radius < 0:
            raise ValueError("view_radius must not be negative")

    @classmethod
    def from_value(cls, value: HiddenDifficulty | str | dict[str, Any]) -> HiddenDifficulty:
        """Build a difficulty from a named preset or explicit YAML mapping."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        presets = {
            "easy": cls(action_remapping=False, mechanic=None),
            "medium": cls(action_remapping=True, mechanic="key_door"),
            "hard": cls(action_remapping=True, mechanic="button", partial_observability=True),
        }
        try:
            return presets[value.lower()]
        except (KeyError, AttributeError) as error:
            raise ValueError(f"unknown difficulty {value!r}") from error


class HiddenRuleGridEnv(Environment):
    """Grid task with per-episode hidden action semantics and goal location.

    The observation omits the action-to-displacement mapping and, by default,
    the goal. Optional mechanics are represented by visible tile types but
    their transition rules are not described. A local observation window can
    be enabled independently of the hidden mechanics.
    """

    _MOVES = VisibleGridEnv._MOVES
    _SPECIAL_TILE = 4

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        max_steps: int = 100,
        difficulty: HiddenDifficulty | str | dict[str, Any] = "medium",
    ) -> None:
        """Create an environment; sampled rules are initialized by ``reset``."""
        if width < 3 or height < 3:
            raise ValueError("width and height must both be at least 3")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.difficulty = HiddenDifficulty.from_value(difficulty)
        self._rng = random.Random()
        self._player = (0, 0)
        self._goal = (width - 1, height - 1)
        self._walls: set[tuple[int, int]] = set()
        self._mapping = dict(self._MOVES)
        self._key: tuple[int, int] | None = None
        self._door: tuple[int, int] | None = None
        self._one_way: tuple[int, int] | None = None
        self._button: tuple[int, int] | None = None
        self._key_collected = False
        self._walls_toggled = False
        self._steps = 0
        self._done = False

    @property
    def observation_spec(self) -> Spec:
        """Describe only public fields, never sampled mechanics or hidden goal."""
        return {
            "grid": "integer tile matrix; 0 empty, 1 wall, 2 player, 4 unidentified special tile",
            "player": "(x, y), absolute coordinate",
            "window_origin": "absolute coordinate of grid[0][0]",
            "steps": "number of actions applied",
            "info.progress": "reward-side signal emitted near the hidden goal",
        }

    @property
    def action_spec(self) -> Spec:
        """Return symbolic commands; their sampled effects are not specified."""
        return {"move": tuple(self._MOVES)}

    def reset(self, seed: int | None = None) -> Observation:
        """Sample a new layout and hidden rule set deterministically from seed."""
        self._rng.seed(seed)
        start = (0, 0)
        self._goal = self._rng.choice(
            [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) != start]
        )
        self._player = start
        self._walls = {
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in (start, self._goal) and self._rng.random() < 0.08
        }
        route = [start]
        x, y = start
        while x != self._goal[0]:
            x += 1 if self._goal[0] > x else -1
            route.append((x, y))
        while y != self._goal[1]:
            y += 1 if self._goal[1] > y else -1
            route.append((x, y))
        self._walls.difference_update(route)

        commands = list(self._MOVES)
        directions = list(self._MOVES.values())
        self._rng.shuffle(directions)
        self._mapping = dict(zip(commands, directions))
        if not self.difficulty.action_remapping:
            self._mapping = dict(self._MOVES)

        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self._walls and (x, y) not in (start, self._goal)
        ]
        self._rng.shuffle(free)
        self._key = self._door = self._one_way = self._button = None
        safe_free = [cell for cell in free if cell not in route]
        if self.difficulty.mechanic == "key_door" and len(safe_free) >= 1:
            door_candidates = [cell for cell in route[1:-1]] or safe_free
            self._door = door_candidates[0]
            self._key = safe_free.pop()
            self._carve_reachable_key_route(start, self._key, self._door)
        elif self.difficulty.mechanic == "one_way" and free:
            self._one_way = free.pop()
        elif self.difficulty.mechanic == "button" and free:
            self._button = free.pop()
        self._key_collected = False
        self._walls_toggled = False
        self._steps = 0
        self._done = False
        return self._observation()

    def _carve_reachable_key_route(
        self,
        start: tuple[int, int],
        key: tuple[int, int],
        door: tuple[int, int],
    ) -> None:
        """Clear one in-bounds path to the key that does not cross the door."""
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue and key not in previous:
            x, y = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if neighbor == door or neighbor in previous:
                    continue
                previous[neighbor] = (x, y)
                queue.append(neighbor)
        if key not in previous:
            raise RuntimeError("could not connect the sampled key to the start")
        current: tuple[int, int] | None = key
        while current is not None:
            self._walls.discard(current)
            current = previous[current]

    def step(self, action: Action) -> StepResult:
        """Apply one symbolic action under the episode's hidden mechanics."""
        if self._done:
            raise RuntimeError("episode is finished; call reset() before step()")
        command = action.get("move")
        if command not in self._mapping:
            raise ValueError(f"unknown move {command!r}; expected one of {tuple(self._MOVES)}")
        dx, dy = self._mapping[str(command)]
        candidate = (self._player[0] + dx, self._player[1] + dy)
        blocked = not (0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height)
        blocked = blocked or (candidate in self._walls and not self._walls_toggled)
        blocked = blocked or (candidate == self._door and not self._key_collected)
        if candidate == self._one_way and self._player[0] >= candidate[0]:
            blocked = True
        key_collected_now = False
        if not blocked:
            self._player = candidate
            if self._player == self._key:
                self._key_collected = True
                key_collected_now = True
            if self._player == self._button:
                self._walls_toggled = not self._walls_toggled

        self._steps += 1
        success = self._player == self._goal
        progress = not success and abs(self._player[0] - self._goal[0]) + abs(
            self._player[1] - self._goal[1]
        ) <= 1
        truncated = self._steps >= self.max_steps and not success
        self._done = success or truncated
        reward = 1.0 if success else (0.1 if progress else 0.0)
        info = {
            "progress": progress,
            "blocked": blocked,
            "key_collected": key_collected_now,
        }
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=success,
            truncated=truncated,
            info=info,
        )

    def _observation(self) -> Observation:
        full = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self._walls:
            full[y][x] = 1
        if self._door is not None:
            full[self._door[1]][self._door[0]] = self._SPECIAL_TILE
        if self._key is not None and not self._key_collected:
            full[self._key[1]][self._key[0]] = self._SPECIAL_TILE
        if self._one_way is not None:
            full[self._one_way[1]][self._one_way[0]] = self._SPECIAL_TILE
        if self._button is not None:
            full[self._button[1]][self._button[0]] = self._SPECIAL_TILE
        px, py = self._player
        full[py][px] = 2

        if self.difficulty.partial_observability:
            radius = self.difficulty.view_radius
            left, top = max(0, px - radius), max(0, py - radius)
            right, bottom = min(self.width, px + radius + 1), min(self.height, py + radius + 1)
            grid = tuple(tuple(row[left:right]) for row in full[top:bottom])
            origin = (left, top)
        else:
            grid = tuple(tuple(row) for row in full)
            origin = (0, 0)

        observation: dict[str, Any] = {
            "grid": grid,
            "player": self._player,
            "window_origin": origin,
            "steps": self._steps,
        }
        if not self.difficulty.hidden_goal:
            observation["goal"] = self._goal
        return observation

    def _oracle_state(self) -> dict[str, Any]:
        """Return privileged truth for controlled evaluator ablations only."""
        return {"action_mapping": dict(self._mapping), "goal": self._goal}


# Preserve the original public name for existing examples and tests.
MockEnvironment = VisibleGridEnv

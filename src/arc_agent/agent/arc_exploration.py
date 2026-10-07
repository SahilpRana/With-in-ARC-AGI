"""Observation-driven exploration for variable ARC-AGI-3 action spaces."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from arc_agent.perception.frame_encoder import (
    Component,
    FrameEncoder,
    PersistentCellMask,
)


@dataclass
class BanditArm:
    """Observed change and level-progress outcomes for one action/object pair."""

    trials: int = 0
    frame_changes: int = 0
    level_increases: int = 0
    reward_total: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_total / self.trials if self.trials else 0.0


@dataclass(frozen=True)
class ArcDecision:
    """An environment action plus internal learning labels."""

    action: Mapping[str, Any]
    arm_key: tuple[Any, ...]


class ArcExplorer:
    """Select random, object-click, or movement-frontier actions from observations."""

    def __init__(
        self,
        strategy: str,
        seed: int = 0,
    ) -> None:
        if strategy not in {"random", "click-exploration", "frontier"}:
            raise ValueError(f"unknown ARC exploration strategy {strategy!r}")
        self.strategy = strategy
        self._rng = random.Random(seed)
        self.encoder = FrameEncoder()
        self.persistent_mask = PersistentCellMask()
        self.bandit: dict[tuple[Any, ...], BanditArm] = defaultdict(BanditArm)
        self._noop_clicks: set[tuple[bytes, int, int]] = set()
        self._covered_clicks: set[tuple[int, int]] = set()
        self._visited_positions: set[tuple[int, int]] = set()
        self._movement: dict[str, tuple[int, int]] = {}
        self._tracked_position: tuple[int, int] | None = None
        self._goal_colors: set[int] = set()
        self._frontier_noops: dict[bytes, set[str]] = defaultdict(set)
        self._frontier_last_actions: dict[bytes, str] = {}
        self._action_counts: dict[str, int] = defaultdict(int)
        self._action_successes: dict[str, int] = defaultdict(int)
        self._total_bandit_trials = 0

    def choose(self, observation: Mapping[str, Any]) -> ArcDecision:
        frame = _primary_frame(observation)
        available = tuple(
            name for name in observation["available_actions"]
            if name not in {"RESET"}
        )
        if not available:
            raise ValueError("observation has no available actions")

        if self.strategy == "random":
            name = self._rng.choice(available)
            point = (
                (self._rng.randrange(64), self._rng.randrange(64))
                if name == "ACTION6"
                else None
            )
            action = self._make_action(name, point)
            return ArcDecision(action, (name, "random"))
        if self.strategy == "frontier":
            return self._choose_frontier(observation, frame, available)
        return self._choose_click(observation, frame, available)

    def observe(
        self,
        decision: ArcDecision,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        before_frame = _primary_frame(before)
        after_frame = _primary_frame(after)
        before_levels = int(before["levels_completed"])
        level_increase = max(0, int(after["levels_completed"]) - before_levels)
        action_name = str(decision.action["action"])
        diff = self.encoder.compare(
            before_frame,
            after_frame,
            action=action_name,
            ignored_cells=self.persistent_mask.mask,
        )
        arm = self.bandit[decision.arm_key]
        arm.trials += 1
        arm.frame_changes += int(bool(diff.changed_cells))
        arm.level_increases += level_increase
        frame_reward = float(bool(diff.changed_cells))
        arm.reward_total += frame_reward + 10.0 * level_increase
        self._total_bandit_trials += 1
        self._action_counts[action_name] += 1
        self._action_successes[action_name] += int(bool(diff.changed_cells) or level_increase > 0)
        if (
            level_increase > 0
            and action_name == "ACTION6"
            and decision.arm_key[:2] == ("ACTION6", "object")
        ):
            self._goal_colors.add(int(decision.arm_key[2]))

        state_key = _state_key(before_frame, before_levels)
        frontier_state_key = self._frontier_state_key(
            before_frame,
            before_levels,
        )
        if (
            self.strategy == "frontier"
            and action_name != "ACTION6"
            and not diff.changed_cells
            and level_increase == 0
        ):
            self._frontier_noops[frontier_state_key].add(action_name)
        data = decision.action.get("data")
        if action_name == "ACTION6" and isinstance(data, Mapping):
            x, y = int(data["x"]), int(data["y"])
            self._covered_clicks.add((x, y))
            if not diff.changed_cells and level_increase == 0:
                self._noop_clicks.add((state_key, x, y))
        if diff.moved_object is not None:
            self._movement[action_name] = diff.moved_object.displacement
            self._tracked_position = _rounded_point(
                diff.moved_object.after.centroid
            )
            self._visited_positions.add(self._tracked_position)
        elif diff.whole_scene_change:
            self._tracked_position = None
        self.persistent_mask.observe(before_frame, after_frame, action_name)

    def _choose_click(
        self,
        observation: Mapping[str, Any],
        frame: np.ndarray,
        available: Sequence[str],
    ) -> ArcDecision:
        state_key = _state_key(frame, int(observation["levels_completed"]))
        click_actions = "ACTION6" in available
        if click_actions:
            components = self.encoder.components(frame)
            candidates: list[tuple[tuple[Any, ...], Component, tuple[int, int]]] = []
            for component in components:
                if component.color == 0:
                    continue
                x, y = _rounded_point(component.centroid)
                key = _component_arm_key(component, x, y)
                if (state_key, x, y) not in self._noop_clicks:
                    candidates.append((key, component, (x, y)))

            untried = [
                candidate for candidate in candidates
                if self.bandit[candidate[0]].trials == 0
            ]
            if untried:
                priority = min(
                    (
                        item[1].color not in self._goal_colors,
                        item[1].area,
                        item[1].color == 0,
                        item[1].color,
                    )
                    for item in untried
                )
                tied = sorted(
                    (
                        item for item in untried
                        if (
                            item[1].color not in self._goal_colors,
                            item[1].area,
                            item[1].color == 0,
                            item[1].color,
                        ) == priority
                    ),
                    key=lambda item: (item[2][1], item[2][0]),
                )
                key, component, point = self._rng.choice(tied)
                return ArcDecision(
                    self._make_action("ACTION6", point),
                    key,
                )

            tried = [candidate for candidate in candidates if self.bandit[candidate[0]].trials]
            if tried:
                key, _, point = max(
                    tried,
                    key=lambda item: (
                        _ucb_score(self.bandit[item[0]], self._total_bandit_trials),
                        -item[1].area,
                        -item[1].color,
                    ),
                )
                return ArcDecision(self._make_action("ACTION6", point), key)

        non_clicks = [name for name in available if name != "ACTION6"]
        if non_clicks:
            untried_actions = [
                name for name in non_clicks if self._action_counts[name] == 0
            ]
            if untried_actions:
                name = min(untried_actions)
                return ArcDecision(self._make_action(name, None), (name, "discrete"))

        if click_actions:
            for y in range(frame.shape[0]):
                for x in range(frame.shape[1]):
                    if (
                        (x, y) not in self._covered_clicks
                        and (state_key, x, y) not in self._noop_clicks
                    ):
                        return ArcDecision(
                            self._make_action("ACTION6", (x, y)),
                            ("ACTION6", "cell", x, y),
                        )

        if non_clicks:
            name = min(
                non_clicks,
                key=lambda item: (
                    self._action_counts[item],
                    -self._action_successes[item],
                    item,
                ),
            )
            return ArcDecision(self._make_action(name, None), (name, "discrete"))
        raise RuntimeError("all clickable cells are covered or no-op at this state")

    def _choose_frontier(
        self,
        observation: Mapping[str, Any],
        frame: np.ndarray,
        available: Sequence[str],
    ) -> ArcDecision:
        actions = [name for name in available if name != "ACTION6"]
        if not actions:
            return self._choose_click(observation, frame, available)
        state_key = self._frontier_state_key(
            frame,
            int(observation["levels_completed"]),
        )
        blocked = self._frontier_noops[state_key]
        unblocked = [name for name in actions if name not in blocked]
        candidates_to_rank = unblocked or actions
        last_action = self._frontier_last_actions.get(state_key)
        if not self._movement:
            name = min(
                candidates_to_rank,
                key=lambda item: (
                    self._action_counts[item],
                    item == last_action,
                    item,
                ),
            )
            self._frontier_last_actions[state_key] = name
            return ArcDecision(self._make_action(name, None), (name, "movement"))

        current = self._current_position(frame)
        candidates: list[tuple[int, int, str]] = []
        for name in candidates_to_rank:
            delta = self._movement.get(name)
            if delta is None:
                candidates.append((0, -self._action_counts[name], name))
                continue
            destination = (current[0] + delta[0], current[1] + delta[1])
            novelty = int(destination not in self._visited_positions)
            candidates.append((novelty, -self._action_counts[name], name))
        _, _, selected = max(candidates)
        self._frontier_last_actions[state_key] = selected
        return ArcDecision(
            self._make_action(selected, None),
            (selected, "movement"),
        )

    def _frontier_state_key(
        self,
        frame: np.ndarray,
        levels_completed: int,
    ) -> bytes:
        mask = self.persistent_mask.mask
        if not mask.any():
            return _state_key(frame, levels_completed)
        normalized = frame.copy()
        normalized[mask] = 0
        return _state_key(normalized, levels_completed)

    def _current_position(self, frame: np.ndarray) -> tuple[int, int]:
        if self._tracked_position is not None:
            return self._tracked_position
        components = self.encoder.components(frame)
        non_background = [component for component in components if component.color != 0]
        target = min(non_background or list(components), key=lambda item: item.area)
        return _rounded_point(target.centroid)

    @staticmethod
    def _make_action(name: str, point: tuple[int, int] | None) -> dict[str, Any]:
        action: dict[str, Any] = {"action": name}
        if name == "ACTION6":
            if point is None:
                raise ValueError("ACTION6 requires a click coordinate")
            action["data"] = {"x": point[0], "y": point[1]}
        return action


def _primary_frame(observation: Mapping[str, Any]) -> np.ndarray:
    frames = observation.get("frame")
    if not isinstance(frames, (tuple, list)) or not frames:
        raise ValueError("observation must contain at least one frame layer")
    return np.asarray(frames[0])


def _state_key(frame: np.ndarray, levels_completed: int) -> bytes:
    digest = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
    return digest + levels_completed.to_bytes(4, "little", signed=False)


def _component_arm_key(
    component: Component,
    x: int,
    y: int,
) -> tuple[Any, ...]:
    return ("ACTION6", "object", component.color, component.area, x, y)


def _rounded_point(point: tuple[float, float]) -> tuple[int, int]:
    return (
        min(63, max(0, int(round(point[0])))),
        min(63, max(0, int(round(point[1])))),
    )


def _ucb_score(arm: BanditArm, total_trials: int) -> float:
    if arm.trials == 0:
        return math.inf
    return arm.mean_reward + math.sqrt(2.0 * math.log(max(1, total_trials)) / arm.trials)

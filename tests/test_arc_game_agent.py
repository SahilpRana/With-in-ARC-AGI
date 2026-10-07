"""Budget, visible-progress, and terminal-state tests for ARC game episodes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from arc_agent.agent.arc_game_agent import ArcGameAgent
from arc_agent.env.adapter import StepResult


class ScriptedEnvironment:
    def __init__(
        self,
        states: list[tuple[str, int]],
        actions: tuple[str, ...] = ("ACTION1",),
    ) -> None:
        self._states = states
        self._actions = actions
        self._index = 0
        self._frame = np.zeros((64, 64), dtype=np.int8)
        self.actions: list[str] = []

    @property
    def observation_spec(self) -> Mapping[str, Any]:
        return {}

    @property
    def action_spec(self) -> Mapping[str, Any]:
        return {"actions": ("ACTION1",)}

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        self._index = 0
        self._frame = np.zeros((64, 64), dtype=np.int8)
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> StepResult:
        name = str(action["action"])
        self.actions.append(name)
        if name == "RESET":
            self._frame = np.zeros((64, 64), dtype=np.int8)
            return StepResult(observation=self._observation("NOT_FINISHED"))
        else:
            self._index = min(self._index + 1, len(self._states) - 1)
            self._frame[self._index, self._index] = self._index
        return StepResult(
            observation=self._observation(),
            reward=0.0,
            terminated=self._states[self._index][0] in {"WIN", "GAME_OVER"},
        )

    def _observation(self, state_override: str | None = None) -> Mapping[str, Any]:
        state, levels = self._states[self._index]
        return {
            "game_id": "scripted-test-game",
            "frame": (self._frame.copy(),),
            "state": state_override or state,
            "levels_completed": levels,
            "win_levels": 2,
            "available_actions": self._actions,
        }


def test_game_over_triggers_reset_and_reset_counts_against_action_budget() -> None:
    environment = ScriptedEnvironment([
        ("NOT_FINISHED", 0),
        ("GAME_OVER", 0),
        ("WIN", 1),
    ])

    result = ArcGameAgent(
        environment,
        strategy="random",
        seed=3,
        max_actions=3,
    ).run_episode()

    assert environment.actions == ["ACTION1", "RESET", "ACTION1"]
    assert result.actions_used == ("ACTION1", "RESET", "ACTION1")
    assert result.action_count == 3
    assert result.game_over_count == 1
    assert result.success is True
    assert result.levels_completed == 1
    assert result.trajectory[1]["action"] == {"action": "RESET"}


def test_game_over_reset_is_not_sent_after_action_budget_is_exhausted() -> None:
    environment = ScriptedEnvironment([
        ("NOT_FINISHED", 0),
        ("GAME_OVER", 0),
    ])

    result = ArcGameAgent(
        environment,
        strategy="random",
        seed=3,
        max_actions=1,
    ).run_episode()

    assert environment.actions == ["ACTION1"]
    assert result.action_count == 1
    assert result.stopped_by_budget is True
    assert result.game_over_count == 1


def test_levels_completed_increase_is_recorded_as_goal_event() -> None:
    environment = ScriptedEnvironment([
        ("NOT_FINISHED", 0),
        ("NOT_FINISHED", 1),
        ("WIN", 2),
    ])

    result = ArcGameAgent(
        environment,
        strategy="random",
        seed=3,
        max_actions=3,
    ).run_episode()

    assert result.trajectory[0]["goal_event"] is True
    assert result.trajectory[1]["goal_event"] is True
    assert result.success is True


def test_result_reports_frame_change_fraction_and_json_safe_actions() -> None:
    environment = ScriptedEnvironment([
        ("NOT_FINISHED", 0),
        ("WIN", 1),
    ])

    result = ArcGameAgent(
        environment,
        strategy="random",
        seed=1,
        max_actions=1,
    ).run_episode()
    record = result.to_dict()

    assert record["total_steps"] == record["action_count"] == 1
    assert record["frame_changed_steps"] == 1
    assert record["changed_step_fraction"] == 1.0
    assert record["actions_used"] == ["ACTION1"]


def test_frontier_rejects_games_that_are_not_movement_only() -> None:
    environment = ScriptedEnvironment(
        [("NOT_FINISHED", 0), ("WIN", 1)],
        actions=("ACTION1", "ACTION6"),
    )

    with pytest.raises(RuntimeError, match="requires only ACTION1-ACTION4"):
        ArcGameAgent(
            environment,
            strategy="frontier",
            seed=3,
            max_actions=2,
        ).run_episode()

    assert environment.actions == []

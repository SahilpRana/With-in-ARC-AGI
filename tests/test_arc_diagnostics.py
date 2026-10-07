"""Offline unit tests for observation-only development diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from arc_agent.env.adapter import Environment, Observation, Spec, StepResult
from arc_agent.eval.arc_diagnostics import capture_episode
from arc_agent.perception.frame_encoder import PersistentCellMask


class DiagnosticEnvironment(Environment):
    def __init__(self, *, terminal_after_first: bool = False) -> None:
        self.terminal_after_first = terminal_after_first
        self.frame = np.zeros((64, 64), dtype=np.int8)
        self.index = 0
        self.actions: list[str] = []

    @property
    def observation_spec(self) -> Spec:
        return {}

    @property
    def action_spec(self) -> Spec:
        return {"actions": ("ACTION1",)}

    def reset(self, seed: int | None = None) -> Observation:
        self.frame.fill(0)
        self.index = 0
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> StepResult:
        name = str(action["action"])
        self.actions.append(name)
        if name == "RESET":
            self.frame.fill(0)
            self.index = 0
            return StepResult(self._observation())
        self.index += 1
        self.frame[0, 0] = self.index % 2
        self.frame[10, 10] = min(self.index, 1)
        state = (
            "GAME_OVER"
            if self.terminal_after_first and self.index == 1
            else "NOT_FINISHED"
        )
        return StepResult(self._observation(state))

    def _observation(self, state: str = "NOT_FINISHED") -> Observation:
        return {
            "game_id": "diagnostic-test",
            "frame": (self.frame.copy(),),
            "state": state,
            "levels_completed": 0,
            "available_actions": ("ACTION1",),
        }


def test_mask_learning_uses_observed_transitions_across_episode() -> None:
    environment = DiagnosticEnvironment()
    mask = PersistentCellMask(threshold=0.8, minimum_observations=2)

    capture = capture_episode(
        environment,
        strategy="random",
        seed=1,
        max_actions=3,
        persistent_mask=mask,
        learn_mask=True,
    )

    assert mask.observation_count == 3
    assert mask.mask[0, 0]
    assert capture.action_count == 3
    assert capture.raw_changed_steps == 3
    assert capture.observed_actions == ("ACTION1", "ACTION1", "ACTION1")


def test_diagnostic_records_game_over_and_charges_reset_to_budget() -> None:
    environment = DiagnosticEnvironment(terminal_after_first=True)

    capture = capture_episode(
        environment,
        strategy="random",
        seed=1,
        max_actions=3,
    )

    assert environment.actions == ["ACTION1", "RESET", "ACTION1"]
    assert capture.action_count == 3
    assert capture.game_over_count == 2
    assert capture.observed_actions[1] == "RESET"

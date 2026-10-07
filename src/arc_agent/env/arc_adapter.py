"""Offline adapter from the ARC-AGI-3 toolkit to the local environment API."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

from .adapter import Action, Environment, Observation, Spec, StepResult


class ArcAdapter(Environment):
    """Wrap one ARC-AGI-3 game and normalize its observations and actions."""

    def __init__(
        self,
        game_id: str,
        scorecard_id: str,
        environments_dir: str | Path = "environment_files",
        arcade: Any | None = None,
    ) -> None:
        if not game_id:
            raise ValueError("game_id must not be empty")
        if not scorecard_id:
            raise ValueError("scorecard_id must not be empty")

        if arcade is None:
            from arc_agi import Arcade, OperationMode
            from arcengine import GameAction

            arcade = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(environments_dir),
            )
            self._game_action = GameAction
        else:
            self._game_action = getattr(arcade, "GameAction", None)
            if self._game_action is None:
                from arcengine import GameAction

                self._game_action = GameAction

        self.game_id = game_id
        self.scorecard_id = scorecard_id
        self._arcade = arcade
        self._environment: Any | None = None
        self._steps = 0
        self._terminated = False
        self._levels_completed = 0

    @property
    def observation_spec(self) -> Spec:
        return {
            "game_id": "ARC-AGI-3 game identifier",
            "frame": "tuple of copied 2D frame-layer arrays",
            "state": "toolkit game-state value",
            "levels_completed": "completed levels reported by the toolkit",
            "win_levels": "win-level count reported by the toolkit",
            "available_actions": "currently available toolkit action names",
            "steps": "number of non-RESET actions applied in this episode",
        }

    @property
    def action_spec(self) -> Spec:
        available = self._available_action_names()
        return {
            "actions": available,
            "reset_action": "RESET",
            "action_format": {"action": "ACTION1-7 or RESET", "data": "optional mapping"},
            "action6_data": {"x": "integer", "y": "integer"},
        }

    def reset(self, seed: int | None = None) -> Observation:
        environment = self._arcade.make(
            self.game_id,
            seed=0 if seed is None else seed,
            scorecard_id=self.scorecard_id,
            save_recording=False,
            include_frame_data=True,
        )
        if environment is None:
            raise RuntimeError(f"Arcade could not create local game {self.game_id}")
        self._environment = environment
        self._steps = 0
        self._terminated = False
        observation = self._normalize(environment.observation_space)
        self._levels_completed = int(observation["levels_completed"])
        return observation

    def step(self, action: Action) -> StepResult:
        if self._environment is None:
            raise RuntimeError("call reset() before step()")

        name = action.get("action")
        if not isinstance(name, str):
            raise ValueError("action must contain a string 'action' name")
        if self._terminated and name != "RESET":
            raise RuntimeError("episode is finished; call reset() before step()")
        if name != "RESET" and name not in self._available_action_names():
            raise ValueError(f"action {name!r} is not currently available")

        try:
            toolkit_action = self._game_action[name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"unsupported ARC action {name!r}") from error

        action_data = action.get("data")
        if name == "ACTION6":
            if not isinstance(action_data, Mapping):
                raise ValueError("ACTION6 requires data with integer x and y coordinates")
            coordinates = {axis: action_data.get(axis) for axis in ("x", "y")}
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in coordinates.values()
            ):
                raise ValueError("ACTION6 requires data with integer x and y coordinates")
            action_data = {axis: int(value) for axis, value in coordinates.items()}
        elif action_data is not None and not isinstance(action_data, Mapping):
            raise ValueError("action data must be a mapping")

        previous_levels = 0 if name == "RESET" else self._levels_completed
        if name == "RESET":
            self._steps = 0
            self._terminated = False
        next_raw = self._environment.step(toolkit_action, data=action_data)
        if next_raw is None:
            raise RuntimeError(f"ARC toolkit returned no observation for action {name}")
        if name != "RESET":
            self._steps += 1
        observation = self._normalize(next_raw)
        self._levels_completed = int(observation["levels_completed"])
        self._terminated = observation["state"] in {"WIN", "GAME_OVER"}

        reward = float(max(0, self._levels_completed - previous_levels))
        if observation["state"] == "WIN" and reward == 0:
            reward = 1.0
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=self._terminated,
            info={
                "success": observation["state"] == "WIN",
                "game_state": observation["state"],
                "levels_completed": self._levels_completed,
                "win_levels": observation["win_levels"],
            },
        )

    def _available_action_names(self) -> tuple[str, ...]:
        if self._environment is None:
            return ()
        return tuple(
            action.name
            for action in self._environment.action_space
            if action.name != "RESET"
        )

    def _normalize(self, raw: Any) -> Observation:
        if raw is None:
            raise RuntimeError(f"ARC toolkit returned no observation for {self.game_id}")
        frames = getattr(raw, "frame", None)
        if frames is None:
            raise RuntimeError(f"ARC observation for {self.game_id} has no frame")
        copied_frames = tuple(
            frame.copy() if callable(getattr(frame, "copy", None)) else deepcopy(frame)
            for frame in frames
        )
        if not copied_frames:
            raise RuntimeError(f"ARC observation for {self.game_id} has no frame layers")

        state_value = getattr(raw.state, "value", raw.state)
        state_name = str(state_value)
        if state_name not in {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}:
            raise ValueError(f"unsupported ARC game state {state_name!r}")
        return {
            "game_id": str(raw.game_id),
            "frame": copied_frames,
            "state": state_name,
            "levels_completed": int(raw.levels_completed),
            "win_levels": int(raw.win_levels),
            "available_actions": self._available_action_names(),
            "steps": self._steps,
        }

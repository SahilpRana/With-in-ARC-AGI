"""Offline tests for the ARC toolkit adapter using captured runtime frames."""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from arc_agent.env.adapter import StepResult
from arc_agent.env.arc_adapter import ArcAdapter

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
if not all(
    (_FIXTURE_DIR / filename).is_file()
    for filename in (
        "arc312_ls20_random_probe.json",
        "arc312_ls20_random_probe.npz",
    )
):
    pytest.skip("captured ARC toolkit fixtures are not included", allow_module_level=True)

_FIXTURE_METADATA = json.loads(
    (_FIXTURE_DIR / "arc312_ls20_random_probe.json").read_text(encoding="utf-8")
)


class FakeGameAction(IntEnum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7


class FakeEnvironment:
    def __init__(
        self,
        frames: dict[str, np.ndarray],
        action_ids: list[int],
        forced_state: str | None,
    ) -> None:
        self._frames = frames
        self._action_ids = action_ids
        self._forced_state = forced_state
        self._index = 0
        self.calls: list[tuple[FakeGameAction, dict[str, Any] | None]] = []

    @property
    def action_space(self) -> list[FakeGameAction]:
        return [FakeGameAction(action_id) for action_id in self._action_ids]

    @property
    def observation_space(self) -> Any:
        record = _FIXTURE_METADATA["records"][self._index]
        frame = self._frames[record["frame_keys"][0]]
        state = (
            self._forced_state or record["state"]
            if self._index > 0
            else record["state"]
        )
        return SimpleNamespace(
            game_id=record["game_id"],
            frame=[frame],
            state=state,
            levels_completed=record["levels_completed"],
            win_levels=record["win_levels"],
        )

    def step(
        self,
        action: FakeGameAction,
        data: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((action, data))
        if action is FakeGameAction.RESET:
            self._index = 0
        else:
            self._index = min(self._index + 1, len(_FIXTURE_METADATA["records"]) - 1)
        return self.observation_space


class FakeArcToolkit:
    GameAction = FakeGameAction

    def __init__(
        self,
        action_ids: list[int] | None = None,
        forced_state: str | None = None,
    ) -> None:
        with np.load(
            _FIXTURE_DIR / "arc312_ls20_random_probe.npz",
            allow_pickle=False,
        ) as archive:
            self.frames = {key: archive[key].copy() for key in archive.files}
        self.action_ids = action_ids or [1, 2, 3, 4]
        self.forced_state = forced_state
        self.make_calls: list[dict[str, Any]] = []
        self.environment: FakeEnvironment | None = None

    def make(self, game_id: str, **kwargs: Any) -> FakeEnvironment:
        self.make_calls.append({"game_id": game_id, **kwargs})
        self.environment = FakeEnvironment(
            self.frames,
            self.action_ids,
            self.forced_state,
        )
        return self.environment


def _adapter(
    *,
    action_ids: list[int] | None = None,
    forced_state: str | None = None,
) -> tuple[ArcAdapter, FakeArcToolkit]:
    toolkit = FakeArcToolkit(action_ids=action_ids, forced_state=forced_state)
    return ArcAdapter("ls20-9607627b", "fake-scorecard", arcade=toolkit), toolkit


def test_reset_maps_fixture_observation_and_uses_requested_seed() -> None:
    adapter, toolkit = _adapter()

    observation = adapter.reset(seed=19)

    assert toolkit.make_calls == [{
        "game_id": "ls20-9607627b",
        "seed": 19,
        "scorecard_id": "fake-scorecard",
        "save_recording": False,
        "include_frame_data": True,
    }]
    assert observation["game_id"] == "ls20-9607627b"
    assert observation["state"] == "NOT_FINISHED"
    assert observation["levels_completed"] == 0
    assert observation["win_levels"] == 7
    assert observation["available_actions"] == (
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
    )
    frame = observation["frame"][0]
    assert frame.shape == (64, 64)
    assert frame.dtype == np.int8


def test_step_returns_normalized_result_and_reset_action_restarts_episode() -> None:
    adapter, toolkit = _adapter()
    adapter.reset(seed=0)

    result = adapter.step({"action": "ACTION1"})

    assert isinstance(result, StepResult)
    assert result.observation["steps"] == 1
    assert result.terminated is False
    assert toolkit.environment is not None
    reset_result = adapter.step({"action": "RESET"})
    assert toolkit.environment.calls[-1] == (FakeGameAction.RESET, None)
    assert reset_result.observation["steps"] == 0
    assert reset_result.observation["state"] == "NOT_FINISHED"
    assert reset_result.terminated is False


@pytest.mark.parametrize(
    ("terminal_state", "success"),
    [("GAME_OVER", False), ("WIN", True)],
)
def test_terminal_toolkit_states_map_to_terminated(
    terminal_state: str,
    success: bool,
) -> None:
    adapter, _ = _adapter(forced_state=terminal_state)
    adapter.reset(seed=0)

    result = adapter.step({"action": "ACTION1"})

    assert result.terminated is True
    assert result.info["success"] is success
    assert result.observation["state"] == terminal_state
    with pytest.raises(RuntimeError, match="episode is finished"):
        adapter.step({"action": "ACTION1"})


def test_action6_passes_coordinate_data_without_normalizing_it_away() -> None:
    adapter, toolkit = _adapter(action_ids=[6])
    adapter.reset(seed=0)

    result = adapter.step({
        "action": "ACTION6",
        "data": {"x": np.int64(17), "y": np.int64(42)},
    })

    assert toolkit.environment is not None
    assert toolkit.environment.calls[-1] == (
        FakeGameAction.ACTION6,
        {"x": 17, "y": 42},
    )
    assert result.observation["steps"] == 1


def test_action6_requires_integer_coordinates() -> None:
    adapter, _ = _adapter(action_ids=[6])
    adapter.reset(seed=0)

    with pytest.raises(ValueError, match="integer x and y"):
        adapter.step({"action": "ACTION6", "data": {"x": 2, "y": "3"}})

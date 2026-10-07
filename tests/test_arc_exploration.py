"""Tests for ARC variable-action exploration and click feedback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from arc_agent.agent.arc_exploration import ArcExplorer


def _observation(
    frame: np.ndarray,
    actions: tuple[str, ...] = ("ACTION6",),
    levels: int = 0,
) -> dict[str, Any]:
    return {
        "frame": (frame,),
        "available_actions": actions,
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }


def test_click_exploration_starts_with_small_object_centroids() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10, 10] = 2
    frame[20:22, 20:22] = 3
    explorer = ArcExplorer("click-exploration", seed=0)

    decision = explorer.choose(_observation(frame))

    assert decision.action == {"action": "ACTION6", "data": {"x": 10, "y": 10}}


def test_click_tie_break_is_seeded_and_varies_across_seeds() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10, 10] = 2
    frame[20, 20] = 2
    observation = _observation(frame)

    first = ArcExplorer("click-exploration", seed=0).choose(observation)
    repeat = ArcExplorer("click-exploration", seed=0).choose(observation)
    other = ArcExplorer("click-exploration", seed=1).choose(observation)

    assert first.action == repeat.action
    assert first.action != other.action


def test_noop_click_is_not_repeated_at_the_same_state() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10, 10] = 2
    frame[20, 20] = 3
    observation = _observation(frame, ("ACTION6", "ACTION5", "ACTION7"))
    explorer = ArcExplorer("click-exploration", seed=0)

    first = explorer.choose(observation)
    explorer.observe(first, observation, observation)
    second = explorer.choose(observation)

    assert second.action["action"] == "ACTION6"
    assert second.action["data"] != first.action["data"]


def test_click_explorer_covers_variable_non_click_actions_before_grid_cells() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10, 10] = 2
    observation = _observation(frame, ("ACTION6", "ACTION5", "ACTION7"))
    explorer = ArcExplorer("click-exploration", seed=0)
    observed_names: list[str] = []

    for _ in range(5):
        decision = explorer.choose(observation)
        observed_names.append(str(decision.action["action"]))
        explorer.observe(decision, observation, observation)

    assert observed_names[:3] == ["ACTION6", "ACTION5", "ACTION7"]
    assert observed_names[3] == "ACTION6"
    assert observed_names[4] == "ACTION6"


def test_bandit_records_frame_and_level_progress_effects() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    before_frame[5, 5] = 4
    after_frame = before_frame.copy()
    after_frame[5, 5] = 0
    before = _observation(before_frame, levels=0)
    after = _observation(after_frame, levels=1)
    explorer = ArcExplorer("click-exploration", seed=0)

    decision = explorer.choose(before)
    explorer.observe(decision, before, after)
    arm = explorer.bandit[decision.arm_key]

    assert arm.trials == 1
    assert arm.frame_changes == 1
    assert arm.level_increases == 1
    assert arm.mean_reward == 11.0


def test_level_progress_discovers_and_prioritizes_successful_click_color() -> None:
    first_frame = np.zeros((64, 64), dtype=np.int8)
    first_frame[5, 5] = 2
    first_frame[20:22, 20:22] = 3
    before = _observation(first_frame, levels=0)
    after = _observation(first_frame, levels=1)
    explorer = ArcExplorer("click-exploration", seed=0)

    successful_click = explorer.choose(before)
    assert successful_click.action["data"] == {"x": 5, "y": 5}
    explorer.observe(successful_click, before, after)

    next_frame = np.zeros((64, 64), dtype=np.int8)
    next_frame[10:12, 10:12] = 2
    next_frame[30, 30] = 3
    next_observation = _observation(next_frame, levels=1)
    next_click = explorer.choose(next_observation)

    assert explorer._goal_colors == {2}
    assert next_click.action["data"] == {"x": 10, "y": 10}


def test_bandit_uses_binary_reward_for_nonempty_frame_changes() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    one_cell = before_frame.copy()
    one_cell[0, 0] = 1
    sixteen_cells = before_frame.copy()
    sixteen_cells[:4, :4] = 1
    before = _observation(before_frame)
    explorer = ArcExplorer("click-exploration", seed=0)
    small_decision = explorer.choose(before)
    explorer.observe(small_decision, before, _observation(one_cell))
    large_decision = explorer.choose(before)
    explorer.observe(large_decision, before, _observation(sixteen_cells))

    assert explorer.bandit[small_decision.arm_key].mean_reward == 1.0
    assert explorer.bandit[large_decision.arm_key].mean_reward == 1.0


def test_bandit_binary_signal_rewards_whole_scene_change() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    after_frame = np.ones((64, 64), dtype=np.int8)
    before = _observation(before_frame)
    after = _observation(after_frame)
    explorer = ArcExplorer("click-exploration", seed=0)

    decision = explorer.choose(before)
    explorer.observe(decision, before, after)

    arm = explorer.bandit[decision.arm_key]
    assert arm.frame_changes == 1
    assert arm.mean_reward == 1.0


def test_bandit_ignores_cells_learned_as_persistent_frame_changes() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    after_frame = before_frame.copy()
    after_frame[0, 0] = 1
    before = _observation(before_frame, ("ACTION1",))
    after = _observation(after_frame, ("ACTION1",))
    explorer = ArcExplorer("random", seed=0)

    for _ in range(21):
        decision = explorer.choose(before)
        explorer.observe(decision, before, after)

    arm = explorer.bandit[("ACTION1", "random")]
    assert explorer.persistent_mask.mask[0, 0]
    assert arm.trials == 21
    assert arm.frame_changes == 20
    assert arm.mean_reward == 20 / 21


def test_random_policy_supplies_coordinates_for_action6() -> None:
    explorer = ArcExplorer("random", seed=0)

    decision = explorer.choose(_observation(
        np.zeros((64, 64), dtype=np.int8),
        ("ACTION6",),
    ))

    assert decision.action["action"] == "ACTION6"
    assert 0 <= decision.action["data"]["x"] < 64
    assert 0 <= decision.action["data"]["y"] < 64


def test_frontier_uses_learned_object_displacement() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    before_frame[10, 10] = 4
    after_frame = np.zeros((64, 64), dtype=np.int8)
    after_frame[10, 11] = 4
    before = _observation(before_frame, ("ACTION1", "ACTION2"))
    after = _observation(after_frame, ("ACTION1", "ACTION2"))
    explorer = ArcExplorer("frontier", seed=0)

    decision = explorer.choose(before)
    explorer.observe(decision, before, after)
    next_decision = explorer.choose(after)

    assert explorer._movement["ACTION1"] == (1, 0)
    assert next_decision.action["action"] == "ACTION1"


def test_frontier_position_tracks_moved_object_not_small_static_component() -> None:
    before_frame = np.zeros((64, 64), dtype=np.int8)
    before_frame[10:12, 10:12] = 4
    before_frame[30, 30] = 2
    after_frame = before_frame.copy()
    after_frame[10:12, 10:12] = 0
    after_frame[10:12, 12:14] = 4
    before = _observation(before_frame, ("ACTION1", "ACTION2"))
    after = _observation(after_frame, ("ACTION1", "ACTION2"))
    explorer = ArcExplorer("frontier", seed=0)

    decision = explorer.choose(before)
    explorer.observe(decision, before, after)

    assert explorer._tracked_position == (12, 10)
    assert explorer._current_position(after_frame) == (12, 10)


def test_frontier_avoids_a_noop_action_at_the_same_state() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    observation = _observation(frame, ("ACTION1", "ACTION2"))
    explorer = ArcExplorer("frontier", seed=0)

    first = explorer.choose(observation)
    explorer.observe(first, observation, observation)
    second = explorer.choose(observation)

    assert first.action["action"] == "ACTION1"
    assert second.action["action"] == "ACTION2"


def test_frontier_cycles_when_every_action_is_a_noop() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    observation = _observation(frame, ("ACTION1", "ACTION2"))
    explorer = ArcExplorer("frontier", seed=0)
    selected: list[str] = []

    for _ in range(3):
        decision = explorer.choose(observation)
        selected.append(str(decision.action["action"]))
        explorer.observe(decision, observation, observation)

    assert selected == ["ACTION1", "ACTION2", "ACTION1"]


def test_frontier_noop_memory_ignores_persistent_cell_variation() -> None:
    base_frame = np.zeros((64, 64), dtype=np.int8)
    masked_frame = base_frame.copy()
    masked_frame[0, 0] = 1
    explorer = ArcExplorer("frontier", seed=0)
    for _ in range(20):
        explorer.persistent_mask.observe(base_frame, masked_frame)

    before = _observation(masked_frame, ("ACTION1", "ACTION2"))
    after = _observation(base_frame, ("ACTION1", "ACTION2"))
    first = explorer.choose(before)
    explorer.observe(first, before, after)
    second = explorer.choose(after)

    assert explorer.persistent_mask.mask[0, 0]
    assert first.action["action"] == "ACTION1"
    assert second.action["action"] == "ACTION2"

"""Frame segmentation and transition-diff tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arc_agent.perception.frame_encoder import FrameEncoder, PersistentCellMask

_FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_frames() -> tuple[np.ndarray, np.ndarray]:
    metadata_path = _FIXTURES / "arc312_ls20_random_probe.json"
    frames_path = _FIXTURES / "arc312_ls20_random_probe.npz"
    if not metadata_path.is_file() or not frames_path.is_file():
        pytest.skip("captured ARC toolkit fixtures are not included")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    keys = [
        metadata["records"][index]["frame_keys"][0]
        for index in (0, 1)
    ]
    with np.load(frames_path, allow_pickle=False) as data:
        return data[keys[0]].copy(), data[keys[1]].copy()


def test_components_segment_fixture_by_color_and_cover_every_cell() -> None:
    frame, _ = _fixture_frames()

    components = FrameEncoder().components(frame)

    assert components
    assert all(component.area > 0 for component in components)
    assert {component.color for component in components} == set(
        int(value) for value in np.unique(frame)
    )
    assert sum(component.area for component in components) == frame.size
    assert all(
        frame[y, x] == component.color
        for component in components
        for x, y in component.cells
    )


def test_fixture_diff_records_changed_cells_bbox_and_per_action_statistics() -> None:
    previous, current = _fixture_frames()
    encoder = FrameEncoder()

    diff = encoder.compare(previous, current, action="ACTION4")

    ys, xs = np.nonzero(previous != current)
    assert diff.changed_cells == frozenset(
        (int(x), int(y)) for y, x in zip(ys, xs)
    )
    assert diff.bbox == (
        int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    )
    stats = encoder.action_stats["ACTION4"]
    assert stats.count == 1
    assert stats.changed_steps == 1
    assert stats.no_op_steps == 0
    assert stats.total_changed_cells == len(diff.changed_cells)
    assert stats.changed_fraction == 1.0


def test_noop_transition_has_empty_diff_and_updates_noop_count() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    encoder = FrameEncoder()

    diff = encoder.compare(frame, frame.copy(), action="ACTION6")

    assert diff.changed_cells == frozenset()
    assert diff.bbox is None
    assert diff.changed_fraction == 0.0
    assert diff.whole_scene_change is False
    assert diff.moved_object is None
    assert encoder.action_stats["ACTION6"].no_op_steps == 1


def test_synthetic_component_move_is_tracked_by_color_and_displacement() -> None:
    previous = np.zeros((64, 64), dtype=np.int8)
    current = previous.copy()
    previous[10:12, 10:13] = 5
    current[15:17, 20:23] = 5

    diff = FrameEncoder().compare(previous, current, action="ACTION1")

    assert diff.moved_object is not None
    assert diff.moved_object.color == 5
    assert diff.moved_object.displacement == (10, 5)
    assert diff.moved_object.before.area == 6
    assert diff.moved_object.after.area == 6


def test_whole_scene_change_is_marked_without_claiming_object_tracking() -> None:
    previous = np.zeros((64, 64), dtype=np.int8)
    current = np.ones((64, 64), dtype=np.int8)

    diff = FrameEncoder().compare(previous, current)

    assert diff.changed_fraction == 1.0
    assert diff.whole_scene_change is True
    assert diff.moved_object is None


def test_persistent_cell_mask_learns_and_ignores_only_frequent_changes() -> None:
    baseline = np.zeros((64, 64), dtype=np.int8)
    mask = PersistentCellMask(threshold=0.8, minimum_observations=5)
    for index in range(5):
        current = baseline.copy()
        current[0, 0] = 1
        if index < 3:
            current[10, 10] = 2
        mask.observe(baseline, current, action=f"ACTION{index % 2 + 1}")

    assert mask.mask[0, 0]
    assert not mask.mask[10, 10]
    assert mask.masked_cell_count == 1
    diff = FrameEncoder().compare(
        baseline,
        np.where(mask.mask, 1, baseline).astype(np.int8),
        ignored_cells=mask.mask,
    )
    assert diff.raw_changed_cells == frozenset({(0, 0)})
    assert diff.changed_cells == frozenset()


def test_persistent_cell_mask_waits_for_minimum_observations() -> None:
    mask = PersistentCellMask(threshold=0.8, minimum_observations=3)
    before = np.zeros((64, 64), dtype=np.int8)
    after = before.copy()
    after[3, 4] = 1
    mask.observe(before, after, action="ACTION6")
    mask.observe(before, after, action="ACTION1")

    assert not mask.mask.any()
    assert mask.observation_count == 2
    assert mask.action_counts == {"ACTION6": 1, "ACTION1": 1}


def test_mask_shape_and_dtype_are_validated() -> None:
    before = np.zeros((64, 64), dtype=np.int8)
    with pytest.raises(ValueError, match="boolean mask"):
        FrameEncoder().compare(
            before,
            before,
            ignored_cells=np.zeros((64, 64), dtype=np.uint8),
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((8, 8), dtype=np.int8), "64x64"),
        (np.zeros((64, 64), dtype=np.uint8), "int8"),
    ],
)
def test_components_reject_unexpected_frame_layout(
    frame: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FrameEncoder().components(frame)

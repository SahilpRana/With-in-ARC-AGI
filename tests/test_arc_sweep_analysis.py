"""Tests for exact paired sweep analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_arc_sweep import (
    exact_mcnemar_p,
    summarize_change_sizes,
)


def test_exact_mcnemar_for_one_directional_discordance() -> None:
    assert exact_mcnemar_p(0, 1) == 1.0
    assert exact_mcnemar_p(0, 5) == pytest.approx(0.0625)


def test_exact_mcnemar_is_one_when_discordances_are_balanced() -> None:
    assert exact_mcnemar_p(3, 3) == 1.0


@pytest.mark.parametrize(("random_only", "click_only"), [(-1, 0), (0, -1)])
def test_exact_mcnemar_rejects_negative_counts(
    random_only: int,
    click_only: int,
) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        exact_mcnemar_p(random_only, click_only)


def test_change_size_summary_separates_clicks_other_actions_and_resets(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "trace.jsonl"
    episode = {
        "game_id": "dev-game",
        "trajectory": [
            {
                "action": {"action": "ACTION6"},
                "changed_cell_count": 0,
                "frame_changed": False,
            },
            {
                "action": {"action": "ACTION6"},
                "changed_cell_count": 4,
                "frame_changed": True,
            },
            {
                "action": {"action": "ACTION1"},
                "changed_cell_count": 9,
                "frame_changed": True,
            },
            {
                "action": {"action": "RESET"},
                "changed_cell_count": 16,
                "frame_changed": True,
            },
        ],
    }
    traces.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    rows = summarize_change_sizes(traces)

    click = next(row for row in rows if row["action_class"] == "click")
    assert click["transitions"] == 2
    assert click["frame_changes"] == 1
    assert click["frame_change_fraction"] == 0.5
    assert click["mean_changed_cells_nonzero"] == 4.0
    assert next(row for row in rows if row["action_class"] == "other")[
        "median_changed_cells_all"
    ] == 9.0
    assert next(row for row in rows if row["action_class"] == "reset")[
        "max_changed_cells_all"
    ] == 16


def test_change_size_summary_rejects_inconsistent_frame_flag(tmp_path: Path) -> None:
    traces = tmp_path / "trace.jsonl"
    traces.write_text(
        json.dumps({
            "game_id": "dev-game",
            "trajectory": [{
                "action": {"action": "ACTION1"},
                "changed_cell_count": 1,
                "frame_changed": False,
            }],
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagrees"):
        summarize_change_sizes(traces)

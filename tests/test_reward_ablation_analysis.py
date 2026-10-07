"""Tests for paired reward-ablation analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.analyze_reward_ablation import analyze, exact_mcnemar_p


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "game_id",
        "seed",
        "levels_completed",
        "success",
        "score",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_mcnemar_for_matched_reward_outcomes() -> None:
    assert exact_mcnemar_p(0, 5) == pytest.approx(0.0625)
    assert exact_mcnemar_p(2, 2) == 1.0


def test_analysis_writes_seed_pairs_and_pooled_game_summaries(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "binary.csv"
    log = tmp_path / "log.csv"
    _write(binary, [
        {"game_id": "g1", "seed": 1, "levels_completed": 0, "success": "False", "score": 0},
        {"game_id": "g1", "seed": 2, "levels_completed": 1, "success": "False", "score": 1},
    ])
    _write(log, [
        {"game_id": "g1", "seed": 1, "levels_completed": 1, "success": "False", "score": 2},
        {"game_id": "g1", "seed": 2, "levels_completed": 1, "success": "False", "score": 3},
    ])

    paired, summary = analyze(binary, log, tmp_path / "out")

    with paired.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pooled = next(row for row in rows if row["scope"] == "pooled")
    assert pooled["binary_only"] == "0"
    assert pooled["log_only"] == "1"
    assert float(pooled["mcnemar_exact_p"]) == 1.0
    assert float(pooled["mean_score_difference_log_minus_binary"]) == 2.0


def test_analysis_rejects_mismatched_seed_pairs(tmp_path: Path) -> None:
    binary = tmp_path / "binary.csv"
    log = tmp_path / "log.csv"
    _write(binary, [
        {"game_id": "g1", "seed": 1, "levels_completed": 0, "success": "False", "score": 0},
    ])
    _write(log, [
        {"game_id": "g1", "seed": 2, "levels_completed": 0, "success": "False", "score": 0},
    ])

    with pytest.raises(ValueError, match="exactly the same"):
        analyze(binary, log, tmp_path / "out")

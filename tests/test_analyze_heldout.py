"""Tests for paired held-out analysis and development-sweep validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.analyze_heldout import analyze, exact_mcnemar_p, wilson_interval

def test_primary_analysis_runs_on_development_runner_csv_shape(tmp_path: Path) -> None:
    input_path = tmp_path / "development-sweep.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("game_id", "agent", "seed", "score", "success", "levels_completed"),
        )
        writer.writeheader()
        for seed in range(3):
            for agent in ("random", "click-exploration"):
                writer.writerow({
                    "game_id": "dev-game",
                    "agent": agent,
                    "seed": seed,
                    "score": float(seed),
                    "success": False,
                    "levels_completed": int(agent == "click-exploration"),
                })

    paths = analyze(
        input_path,
        tmp_path,
    )

    assert len(paths) == 3
    with paths[1].open(encoding="utf-8", newline="") as stream:
        summary = next(csv.DictReader(stream))
    assert int(summary["paired_episodes"]) == 3
    assert int(summary["random_episodes_with_level"]) == 0
    assert int(summary["click_episodes_with_level"]) == 3
    assert int(summary["random_only"]) == 0
    assert int(summary["click_only"]) == 3
    assert float(summary["mcnemar_exact_p"]) == pytest.approx(0.25)
    assert int(summary["random_wins"]) == 0
    assert int(summary["click_wins"]) == 0

    with paths[2].open(encoding="utf-8", newline="") as stream:
        per_game = list(csv.DictReader(stream))
    assert len(per_game) == 1
    game = per_game[0]
    assert game["game_id"] == "dev-game"
    assert float(game["click_wilson_95_low"]) == pytest.approx(0.4385, abs=0.001)
    assert float(game["click_wilson_95_high"]) == pytest.approx(1.0)


def test_exact_mcnemar_and_wilson_edge_cases() -> None:
    assert exact_mcnemar_p(0, 5) == pytest.approx(0.0625)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
    with pytest.raises(ValueError, match="at least one"):
        wilson_interval(0, 0)


def test_frontier_results_are_written_per_movement_game(tmp_path: Path) -> None:
    paired = tmp_path / "paired.csv"
    with paired.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "game_id",
                "agent",
                "seed",
                "score",
                "success",
                "levels_completed",
            ),
        )
        writer.writeheader()
        for game_id in ("tr87-cd924810", "tu93-0768757b"):
            for seed in (20270000, 20270001):
                for agent in ("random", "click-exploration"):
                    writer.writerow({
                        "game_id": game_id,
                        "agent": agent,
                        "seed": seed,
                        "score": 1.0,
                        "success": False,
                        "levels_completed": 1 if agent == "click-exploration" else 0,
                    })
    frontier = tmp_path / "frontier.csv"
    with frontier.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("game_id", "agent", "seed", "score", "success", "levels_completed"),
        )
        writer.writeheader()
        for game_id in ("tr87-cd924810", "tu93-0768757b"):
            for seed in (20270000, 20270001):
                writer.writerow({
                    "game_id": game_id,
                    "agent": "frontier",
                    "seed": seed,
                    "score": 2.0,
                    "success": False,
                    "levels_completed": 0,
                })

    paths = analyze(paired, tmp_path / "out", frontier)

    assert len(paths) == 5
    with paths[3].open(encoding="utf-8", newline="") as stream:
        frontier_games = list(csv.DictReader(stream))
    assert {row["game_id"] for row in frontier_games} == {
        "tr87-cd924810",
        "tu93-0768757b",
    }
    assert all(int(row["episodes"]) == 2 for row in frontier_games)


def test_analysis_rejects_incomplete_pairs(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.csv"
    path.write_text(
        "game_id,agent,seed,score,success,levels_completed\n"
        "game-a,random,1,0,False,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly random and click-exploration"):
        analyze(path, tmp_path / "out")

"""Development split and scorecard-alignment tests for the ARC runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agent.eval.arc_runner as arc_runner
from arc_agent.eval.arc_runner import (
    _assert_heldout_run_once,
    _score_for_game,
    build_episode_plan,
    build_parser,
    load_development_games,
    load_heldout_games,
    load_movement_games,
    run_heldout_evaluation,
)

_ROOT = Path(__file__).parents[1]


def test_registered_development_games_are_loaded_without_split_changes() -> None:
    split_path = _ROOT / "results" / "game_split.json"
    if not split_path.is_file():
        pytest.skip("development split registry is not included")
    games = load_development_games(split_path)

    assert len(games) == 13
    assert "ls20-9607627b" in games
    assert "tr87-cd924810" not in games
    assert "tu93-0768757b" not in games


def test_explicit_held_out_game_is_rejected_before_runner_start(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({
            "pre_registered_before_tuning": True,
            "development_game_ids": [
                f"development-game-{index}" for index in range(13)
            ],
            "held_out_game_ids": ["heldout-game"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only registered development"):
        load_development_games(
            split_path,
            ["heldout-game"],
        )


def test_runner_defaults_to_development_split() -> None:
    args = build_parser().parse_args(["--output-prefix", "results/dev-run"])

    assert args.split == "development"


def test_heldout_loader_uses_registered_games_without_reading_game_assets(
    tmp_path: Path,
) -> None:
    development = [f"development-{index}" for index in range(13)]
    heldout = [f"heldout-{index}" for index in range(12)]
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({
            "pre_registered_before_tuning": True,
            "development_game_ids": development,
            "held_out_game_ids": heldout,
        }),
        encoding="utf-8",
    )

    assert load_heldout_games(split_path) == tuple(heldout)


def test_heldout_run_refuses_without_frozen_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arc_runner,
        "_git",
        lambda args: SimpleNamespace(returncode=128, stdout="", stderr=""),
    )

    with pytest.raises(ValueError, match="requires the frozen-agent-v1 tag"):
        arc_runner._heldout_prerequisites()


def test_git_helper_uses_utf8_for_windows_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="notes — résumé", stderr="")

    monkeypatch.setattr(arc_runner.subprocess, "run", fake_run)

    result = arc_runner._git(["show", "HEAD:writeup/notes.md"])

    assert result.stdout == "notes — résumé"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_heldout_run_refuses_without_committed_preregistration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(args: tuple[str, ...] | list[str]) -> SimpleNamespace:
        if args[0] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout="tag-hash\n", stderr="")
        if args[0] == "show":
            return SimpleNamespace(
                returncode=0,
                stdout="# Notes\nNo preregistration yet.\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(arc_runner, "_git", fake_git)

    with pytest.raises(ValueError, match="pre-registration must be committed"):
        arc_runner._heldout_prerequisites()


def test_heldout_run_once_guard_rejects_duplicate_episode(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    manifest = {
        "split": "heldout",
        "planned_episodes": [
            {"game_id": "heldout-game", "seed": 20270000, "agent": "random"}
        ],
    }
    (results / "heldout-existing.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    plan = (arc_runner.EpisodePlan("heldout-game", "random", 20270000),)

    with pytest.raises(ValueError, match="may only be run once"):
        _assert_heldout_run_once(results, plan)


def test_heldout_run_once_guard_checks_legacy_seed_manifests(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    manifest = {
        "held_out_final": True,
        "held_out_start": 20270000,
        "seeds_per_game": 1,
    }
    (results / "held_out-existing.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    plan = (arc_runner.EpisodePlan("heldout-game", "random", 20270000),)

    with pytest.raises(ValueError, match="manifest already exists"):
        _assert_heldout_run_once(results, plan)


def test_debug_heldout_run_uses_isolated_debug_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arc_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        arc_runner,
        "_heldout_prerequisites",
        lambda: ("tag-hash", "prereg-hash"),
    )
    monkeypatch.setattr(
        arc_runner,
        "load_heldout_games",
        lambda split_path, requested_games=None: ("heldout-game",),
    )
    monkeypatch.setattr(arc_runner, "load_movement_games", lambda *args: frozenset())
    monkeypatch.setattr(
        arc_runner,
        "_run_evaluation",
        lambda **kwargs: kwargs,
    )

    result = run_heldout_evaluation(
        environments_dir=tmp_path,
        split_path=tmp_path / "split.json",
        survey_path=tmp_path / "survey.csv",
        output_prefix=tmp_path / "must-not-be-used",
        strategies=("random",),
        seeds=(20270000,),
        max_actions=400,
        allow_rerun_for_debug=True,
    )

    assert result["output_prefix"].is_relative_to(
        tmp_path / "results" / "debug" / "heldout"
    )
    assert result["debug_run"] is True


def test_frontier_plan_includes_only_development_movement_games() -> None:
    split_path = _ROOT / "results" / "game_split.json"
    survey_path = _ROOT / "results" / "game_survey.csv"
    if not split_path.is_file() or not survey_path.is_file():
        pytest.skip("development split and game survey are not included")
    games = load_development_games(split_path)
    movement = load_movement_games(survey_path, games)

    plan = build_episode_plan(
        games,
        movement,
        ("random", "click-exploration", "frontier"),
        (20261006, 20261007),
    )

    frontier_games = {
        episode.game_id for episode in plan if episode.agent == "frontier"
    }
    assert frontier_games == {"ls20-9607627b"}
    assert len(plan) == 13 * 2 * 2 + 1 * 2


def test_scorecard_result_must_match_one_game_and_one_run() -> None:
    card = SimpleNamespace(environments=[
        SimpleNamespace(id="ls20-9607627b", runs=[SimpleNamespace(score=12.5)])
    ])

    assert _score_for_game(card, "ls20-9607627b") == 12.5
    with pytest.raises(RuntimeError, match="0 records"):
        _score_for_game(card, "unknown")
    card.environments[0].runs.append(SimpleNamespace(score=20))
    with pytest.raises(RuntimeError, match="one scorecard run"):
        _score_for_game(card, "ls20-9607627b")

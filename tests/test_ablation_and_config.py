"""Reproducibility and configuration behavior for experiment tooling."""

import csv
import json

import pytest
import yaml

from arc_agent.eval.ablation import (
    DIFFICULTIES,
    collect_failure_traces,
    feasibility_report,
    run_ablation,
    run_oracle_ablation,
    run_selected_difficulty,
    summarize_failure_patterns,
    write_results,
)
from arc_agent.eval.runner import main as run_cli
from arc_agent.utils.config import load_config, merge_overrides
from arc_agent.utils.logging import write_run_manifest


def test_ablation_results_repeat_for_fixed_seed_ranges() -> None:
    first = run_ablation([4, 5], [104, 105], max_steps=20, max_seconds=10)
    second = run_ablation([4, 5], [104, 105], max_steps=20, max_seconds=10)

    assert first == second
    assert {row["split"] for row in first} == {"dev", "held_out"}
    assert len(first) == 10


def test_ablation_rejects_overlapping_seed_ranges() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        run_ablation([1, 2], [2, 3])


def test_config_merge_and_manifest_preserve_full_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 7\nbudgets:\n  max_steps: 20\n", encoding="utf-8")
    config = load_config(config_path)
    merged = merge_overrides(config, {"budgets": {"max_steps": 9, "max_seconds": 3}})
    assert merged == {"seed": 7, "budgets": {"max_steps": 9, "max_seconds": 3}}

    manifest = write_run_manifest(tmp_path / "runs.jsonl", "abc123", merged, cwd=tmp_path)
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    assert contents["run_id"] == "abc123"
    assert contents["config"] == merged
    assert "python_version" in contents


def test_ablation_writes_csv_and_markdown(tmp_path) -> None:
    rows = run_ablation([1], [101], max_steps=10, max_seconds=10)
    csv_path, markdown_path = write_results(rows, tmp_path / "summary")

    assert csv_path.exists()
    assert markdown_path.exists()
    assert "steps to solve" in markdown_path.read_text(encoding="utf-8").lower()


def test_per_seed_outcomes_match_aggregate_success_counts(tmp_path) -> None:
    rows = run_selected_difficulty(
        [1, 2],
        "large-tight-budget",
        max_seconds=10,
    )
    write_results(rows, tmp_path / "per-seed-check")

    per_seed_path = tmp_path / "per-seed-check.per-seed.csv"
    with per_seed_path.open(newline="", encoding="utf-8") as stream:
        per_seed_rows = list(csv.DictReader(stream))

    assert len(per_seed_rows) == sum(row["episodes"] for row in rows)
    for aggregate in rows:
        matching_rows = [
            row for row in per_seed_rows
            if row["agent"] == aggregate["variant"]
            and row["condition"] == aggregate["difficulty"]
        ]
        assert len(matching_rows) == aggregate["episodes"]
        assert sum(row["success"] == "True" for row in matching_rows) == aggregate["successes"]


def test_oracle_ablation_has_requested_conditions_and_failure_trace_fields() -> None:
    rows = run_oracle_ablation([0, 1], max_steps=20)
    assert [row["variant"] for row in rows] == [
        "oracle-actions + oracle-goal",
        "oracle-actions only",
        "oracle-goal only",
        "random-only",
    ]
    assert rows[0]["oracle_actions"] and rows[0]["oracle_goal"]
    failures = collect_failure_traces(range(10), count=1, max_steps=5)
    if failures:
        step = failures[0]["trace"][0]
        assert {"hypotheses", "inferred_goal", "planned_path", "chosen_action", "mode"} <= set(step)
        assert isinstance(summarize_failure_patterns(failures), dict)


def test_full_knowledge_bfs_oracle_respects_key_and_budget() -> None:
    feasible = feasibility_report(range(10), DIFFICULTIES[2], max_steps=24)
    insufficient = feasibility_report(range(10), DIFFICULTIES[1], max_steps=5)

    assert feasible["oracle_agent_success_rate"] == 1.0
    assert feasible["unreachable_count"] == 0
    assert feasible["optimal_path_max"] <= 24
    assert insufficient["oracle_agent_success_rate"] < 1.0
    assert insufficient["optimal_over_budget_count"] > 0


def test_cli_loads_config_and_records_per_episode_budget_and_manifest(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    config["environment"]["kind"] = "visible"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    log_path = tmp_path / "episodes.jsonl"

    assert run_cli([
        "--config", str(config_path),
        "--episodes", "1",
        "--seed", "33",
        "--log-file", str(log_path),
    ]) == 0

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    manifests = list(tmp_path.glob("episodes.*.manifest.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert row["budget_limits"]["max_steps"] == config["budgets"]["max_steps"]
    assert row["elapsed_seconds"] >= 0
    assert row["compute_units_used"] >= 0
    assert manifest["config"]["episodes"] == 1
    assert manifest["config"]["seed"] == 33
"""Config-driven CLI runner with reproducible per-run manifests and JSONL logs."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Sequence

from arc_agent.agent.agent import Agent
from arc_agent.env.mock_env import HiddenRuleGridEnv, VisibleGridEnv
from arc_agent.eval.metrics import summarize
from arc_agent.utils.config import load_config, merge_overrides
from arc_agent.utils.logging import append_run, write_run_manifest
from arc_agent.utils.seeding import episode_seed

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI overrides; unspecified values come from the YAML file."""
    parser = argparse.ArgumentParser(description="Evaluate the ARC interactive baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-compute-units", type=int)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--environment", choices=("hidden", "visible"))
    parser.add_argument("--strategy", choices=("random_only", "hypothesis_gated"))
    parser.add_argument("--explorer", choices=("random", "information_gain"))
    return parser


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    overrides = {
        "episodes": args.episodes,
        "seed": args.seed,
        "budgets": {
            "max_steps": args.max_steps,
            "max_seconds": args.max_seconds,
            "max_compute_units": args.max_compute_units,
        },
        "logging": {"path": args.log_file},
        "environment": {"kind": args.environment},
        "agent": {"strategy": args.strategy, "explorer": args.explorer},
    }
    merged = merge_overrides(config, overrides)
    merged["logging"]["path"] = str(merged["logging"]["path"])
    return merged


def _make_environment(config: dict[str, Any]):
    env_config = config["environment"]
    if env_config["kind"] == "hidden":
        return HiddenRuleGridEnv(
            width=env_config["width"],
            height=env_config["height"],
            max_steps=env_config["max_steps"],
            difficulty=env_config["difficulty"],
        )
    if env_config["kind"] == "visible":
        return VisibleGridEnv(
            width=env_config["width"],
            height=env_config["height"],
            max_steps=env_config["max_steps"],
        )
    raise ValueError(f"unknown environment kind {env_config['kind']!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured episodes and write an auditable run manifest."""
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    episodes = int(config["episodes"])
    if episodes < 1:
        raise SystemExit("episodes must be at least 1")
    run_id = uuid.uuid4().hex[:12]
    log_path = Path(config["logging"]["path"])
    manifest_path = write_run_manifest(log_path, run_id, config, cwd=Path.cwd())

    results = []
    budgets = config["budgets"]
    agent_config = config["agent"]
    for index in range(episodes):
        seed = episode_seed(int(config["seed"]), index)
        agent = Agent(
            _make_environment(config),
            seed=seed,
            max_steps=int(budgets["max_steps"]),
            max_seconds=float(budgets["max_seconds"]),
            max_compute_units=int(budgets["max_compute_units"]),
            explorer_kind=str(agent_config["explorer"]),
            min_action_tests=int(agent_config["min_action_tests"]),
            confidence_threshold=float(agent_config["confidence_threshold"]),
            strategy=str(agent_config["strategy"]),
        )
        result = agent.run_episode()
        results.append(result)
        record = result.to_dict()
        record.update({"episode": index, "run_id": run_id})
        append_run(log_path, record)

    metrics = summarize(results)
    print(json.dumps({
        "run_id": run_id,
        "episodes": metrics.episodes,
        "successes": metrics.successes,
        "success_rate": metrics.success_rate,
        "mean_steps": metrics.mean_steps,
        "steps_to_solve": metrics.steps_to_solve,
        "log_file": str(log_path),
        "manifest_file": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

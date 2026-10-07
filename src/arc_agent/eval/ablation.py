"""Reproducible strategy and oracle ablations with failure-trace reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from arc_agent.agent.agent import Agent
from arc_agent.env.mock_env import HiddenDifficulty, HiddenRuleGridEnv
from arc_agent.utils.logging import write_run_manifest


@dataclass(frozen=True)
class Variant:
    """One action-selection condition in an experiment."""

    name: str
    strategy: str
    explorer: str
    oracle_actions: bool = False
    oracle_goal: bool = False


@dataclass(frozen=True)
class BenchmarkDifficulty:
    """Environment and episode budget settings for sensitivity experiments."""

    name: str
    width: int
    height: int
    max_steps: int
    mechanic: str | None = None
    partial_observability: bool = False


DIFFICULTIES = (
    BenchmarkDifficulty("standard", 7, 7, 100),
    BenchmarkDifficulty("large-tight-budget", 10, 10, 18),
    BenchmarkDifficulty("key-door", 10, 10, 24, mechanic="key_door"),
)


VARIANTS = (
    Variant("random-only", "random_only", "random"),
    Variant("hypothesis-gated + random", "hypothesis_gated", "random"),
    Variant("hypothesis-gated + frontier", "hypothesis_gated", "frontier"),
    Variant("hypothesis-gated + information-gain", "hypothesis_gated", "information_gain"),
    Variant("oracle upper bound", "oracle_bfs", "random"),
)
ORACLE_VARIANTS = (
    Variant("oracle-actions + oracle-goal", "hypothesis_gated", "random", True, True),
    Variant("oracle-actions only", "hypothesis_gated", "random", True, False),
    Variant("oracle-goal only", "hypothesis_gated", "random", False, True),
    VARIANTS[0],
)


def _environment(difficulty: BenchmarkDifficulty | None = None) -> HiddenRuleGridEnv:
    difficulty = difficulty or DIFFICULTIES[0]
    return HiddenRuleGridEnv(
        width=difficulty.width,
        height=difficulty.height,
        max_steps=difficulty.max_steps,
        difficulty=HiddenDifficulty(
            action_remapping=True,
            hidden_goal=True,
            mechanic=difficulty.mechanic,
            partial_observability=difficulty.partial_observability,
        )
    )


def evaluate_variant(
    variant: Variant,
    seeds: Sequence[int],
    *,
    max_steps: int = 100,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
    difficulty: BenchmarkDifficulty | None = None,
) -> dict[str, object]:
    """Run one variant over an explicit seed set and summarize outcomes."""
    outcomes: list[dict[str, object]] = []
    for seed in seeds:
        env = _environment(difficulty)
        if variant.strategy == "oracle_bfs":
            env.reset(seed=seed)
            path = oracle_optimal_path(env)
            success = path is not None and len(path) <= max_steps
            steps = min(len(path), max_steps) if path is not None else max_steps
            if success and path is not None:
                for action in path:
                    step_result = env.step(action)
                    steps = int(env._steps)
                    if step_result.terminated or step_result.truncated:
                        success = step_result.terminated
                        break
            outcomes.append({"seed": seed, "steps": steps, "success": success})
            continue
        result = Agent(
            env,
            seed=seed,
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
            explorer_kind=variant.explorer,
            strategy=variant.strategy,
            min_action_tests=2,
            confidence_threshold=0.7,
            oracle_actions=variant.oracle_actions,
            oracle_goal=variant.oracle_goal,
        ).run_episode()
        outcomes.append({"seed": seed, "steps": result.steps, "success": result.success})
    successes = sum(bool(outcome["success"]) for outcome in outcomes)
    solved = [int(outcome["steps"]) for outcome in outcomes if outcome["success"]]
    interval = wilson_interval(successes, len(outcomes))
    return {
        "variant": variant.name,
        "episodes": len(outcomes),
        "successes": successes,
        "success_rate": successes / len(outcomes) if outcomes else 0.0,
        "success_rate_ci95_low": interval[0],
        "success_rate_ci95_high": interval[1],
        "mean_steps": sum(int(outcome["steps"]) for outcome in outcomes) / len(outcomes) if outcomes else 0.0,
        "steps_to_solve": solved,
        "per_seed_outcomes": outcomes,
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * (
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) ** 0.5 / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def oracle_optimal_path(
    env: HiddenRuleGridEnv,
) -> tuple[dict[str, object], ...] | None:
    """Return a shortest full-knowledge action path including key possession."""
    truth = env._oracle_state()
    start = env._player
    goal = truth["goal"]
    key = env._key
    door = env._door
    initial = (start[0], start[1], False)
    queue = deque([initial])
    previous: dict[
        tuple[int, int, bool], tuple[tuple[int, int, bool], str] | None
    ] = {initial: None}
    goal_state = None
    while queue:
        state = queue.popleft()
        x, y, has_key = state
        if (x, y) == goal:
            goal_state = state
            break
        for action, (dx, dy) in truth["action_mapping"].items():
            destination = (x + dx, y + dy)
            if not (0 <= destination[0] < env.width and 0 <= destination[1] < env.height):
                continue
            if destination in env._walls or (destination == door and not has_key):
                continue
            next_state = (destination[0], destination[1], has_key or destination == key)
            if next_state not in previous:
                previous[next_state] = (state, action)
                queue.append(next_state)
    if goal_state is None:
        return None
    path: list[dict[str, object]] = []
    current = goal_state
    while previous[current] is not None:
        prior, action = previous[current]
        path.append({"move": action})
        current = prior
    path.reverse()
    return tuple(path)


def feasibility_report(
    seeds: Sequence[int],
    difficulty: BenchmarkDifficulty,
    *,
    max_steps: int,
) -> dict[str, object]:
    """Measure oracle-agent success and exact optimal path-length distribution."""
    lengths: list[int | None] = []
    oracle_successes = 0
    for seed in seeds:
        env = _environment(difficulty)
        env.reset(seed=seed)
        optimal = oracle_optimal_path(env)
        lengths.append(len(optimal) if optimal is not None else None)
        if optimal is not None and len(optimal) <= max_steps:
            result = None
            for action in optimal:
                result = env.step(action)
                if result.terminated or result.truncated:
                    break
            oracle_successes += int(result is not None and result.terminated)
    reachable = [length for length in lengths if length is not None]
    histogram = Counter(reachable)
    return {
        "difficulty": difficulty.name,
        "episodes": len(seeds),
        "max_steps": max_steps,
        "oracle_agent_successes": oracle_successes,
        "oracle_agent_success_rate": oracle_successes / len(seeds) if seeds else 0.0,
        "optimal_path_histogram": dict(sorted(histogram.items())),
        "optimal_path_min": min(reachable) if reachable else None,
        "optimal_path_median": sorted(reachable)[len(reachable) // 2] if reachable else None,
        "optimal_path_mean": sum(reachable) / len(reachable) if reachable else None,
        "optimal_path_max": max(reachable) if reachable else None,
        "unreachable_count": len(lengths) - len(reachable),
        "optimal_over_budget_count": sum(length > max_steps for length in reachable),
    }


def run_ablation(
    dev_seeds: Sequence[int],
    heldout_seeds: Sequence[int] | None = None,
    *,
    max_steps: int = 100,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Evaluate baseline variants on disjoint development and held-out seeds."""
    if heldout_seeds is not None and set(dev_seeds) & set(heldout_seeds):
        raise ValueError("development and held-out seeds must be disjoint")
    rows = []
    splits = [("dev", dev_seeds)]
    if heldout_seeds is not None:
        splits.append(("held_out", heldout_seeds))
    for split, seeds in splits:
        for variant in VARIANTS:
            row = evaluate_variant(
                variant,
                seeds,
                max_steps=max_steps,
                max_seconds=max_seconds,
                max_compute_units=max_compute_units,
            )
            row.update({"split": split, "seed_start": seeds[0] if seeds else None, "seed_end": seeds[-1] if seeds else None})
            rows.append(row)
    return rows


def run_sensitivity(
    seed_start: int = 0,
    episodes: int = 100,
    *,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Run all strategy variants at each difficulty on development seeds only."""
    if episodes < 100:
        raise ValueError("sensitivity requires at least 100 episodes per condition")
    rows = []
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        seeds = range(seed_start + difficulty_index * 10_000, seed_start + difficulty_index * 10_000 + episodes)
        for variant in VARIANTS:
            row = evaluate_variant(
                variant,
                seeds,
                max_steps=difficulty.max_steps,
                max_seconds=max_seconds,
                max_compute_units=max_compute_units,
                difficulty=difficulty,
            )
            row.update({
                "split": "dev_sensitivity",
                "difficulty": difficulty.name,
                "seed_start": seeds.start,
                "seed_end": seeds.stop - 1,
                "max_steps": difficulty.max_steps,
            })
            rows.append(row)
    return rows


def run_selected_difficulty(
    seeds: Sequence[int],
    difficulty_name: str,
    *,
    split: str = "held_out_final",
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Evaluate all standard variants on one named benchmark condition."""
    matches = [item for item in DIFFICULTIES if item.name == difficulty_name]
    if not matches:
        raise ValueError(f"unknown difficulty {difficulty_name!r}")
    difficulty = matches[0]
    rows = []
    for variant in VARIANTS:
        row = evaluate_variant(
            variant,
            seeds,
            max_steps=difficulty.max_steps,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
            difficulty=difficulty,
        )
        row.update({
            "split": split,
            "difficulty": difficulty.name,
            "seed_start": seeds[0] if seeds else None,
            "seed_end": seeds[-1] if seeds else None,
            "max_steps": difficulty.max_steps,
        })
        rows.append(row)
    return rows


def run_locked_difficulties(
    seeds: Sequence[int],
    *,
    split: str,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Run the five-agent matrix on the two feasible, locked conditions."""
    rows = []
    for difficulty_name in ("large-tight-budget", "key-door"):
        rows.extend(run_selected_difficulty(
            seeds,
            difficulty_name,
            split=split,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
        ))
    return rows


def run_oracle_ablation(
    seeds: Sequence[int],
    *,
    max_steps: int = 100,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Compare the requested oracle conditions on one identical dev seed set."""
    rows = []
    for variant in ORACLE_VARIANTS:
        row = evaluate_variant(
            variant,
            seeds,
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
        )
        row.update({
            "split": "dev_oracle",
            "seed_start": seeds[0] if seeds else None,
            "seed_end": seeds[-1] if seeds else None,
            "oracle_actions": variant.oracle_actions,
            "oracle_goal": variant.oracle_goal,
        })
        rows.append(row)
    return rows


def collect_failure_traces(
    seeds: Sequence[int],
    *,
    count: int = 10,
    max_steps: int = 100,
    max_seconds: float = 60.0,
    max_compute_units: int = 10_000,
) -> list[dict[str, object]]:
    """Collect full per-step traces from up to ``count`` failed gated episodes."""
    failures = []
    for seed in seeds:
        result = Agent(
            _environment(),
            seed=seed,
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
            explorer_kind="random",
            strategy="hypothesis_gated",
            min_action_tests=2,
            confidence_threshold=0.7,
        ).run_episode()
        if not result.success:
            failures.append({"seed": seed, "steps": result.steps, "trace": list(result.trace)})
        if len(failures) >= count:
            break
    return failures


def summarize_failure_patterns(failures: Sequence[dict[str, object]]) -> dict[str, int]:
    """Count common trace patterns per failed episode (counts are not exclusive)."""
    counts: Counter[str] = Counter()
    for failure in failures:
        trace = failure["trace"]
        if not trace:
            continue
        if all(step["mode"] == "explore" for step in trace):
            counts["budget_burned_in_explore"] += 1
        positions = [tuple(step["next_player"]) for step in trace if step["next_player"] is not None]
        stationary = sum(step["blocked"] for step in trace)
        oscillation = any(
            positions[index] == positions[index - 2] and positions[index] != positions[index - 1]
            for index in range(2, len(positions))
        )
        if oscillation:
            counts["position_oscillation"] += 1
        if stationary >= 3:
            counts["repeated_blocked_noops"] += 1
        if any(step["mode"] == "exploit" and not step["planned_path"] for step in trace):
            counts["exploit_without_path"] += 1
        if any(step["mode"] == "exploit" and step["prediction_correct"] is False for step in trace):
            counts["wrong_mapping_committed"] += 1
        if any(step["goal_matches_oracle"] is False for step in trace if step["inferred_goal"] is not None):
            counts["wrong_goal_inferred"] += 1
    return dict(counts)


def write_trace_report(failures: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    """Write hypotheses, goals, plans, actions, and modes for each traced step."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    patterns = summarize_failure_patterns(failures)
    lines = ["# Gated-Agent Failure Traces", "", "## Failure pattern counts", ""]
    lines.extend(f"- {name}: {count}/{len(failures)} episodes" for name, count in patterns.items())
    if not patterns:
        lines.append("- No classified patterns observed.")
    for failure in failures:
        lines.extend(["", f"## Seed {failure['seed']} ({failure['steps']} steps)", ""])
        lines.extend([
            "| Step | Mode | Action | Inferred goal | Planned path | Active hypotheses (confidence) | Outcome |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ])
        for step in failure["trace"]:
            hypotheses = "; ".join(
                f"{item['rule']}={item['confidence']:.2f}"
                for item in step["hypotheses"]
            )
            outcome = f"{step['next_player']}; reward={step['reward']}; blocked={step['blocked']}"
            lines.append(
                f"| {step['step']} | {step['mode']} | `{step['chosen_action']}` | "
                f"`{step['inferred_goal']}` | `{step['planned_path']}` | {hypotheses} | {outcome} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_results(rows: Sequence[dict[str, object]], output_prefix: str | Path) -> tuple[Path, Path]:
    """Write aggregate results and one row per seed to CSV files."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, markdown_path = prefix.with_suffix(".csv"), prefix.with_suffix(".md")
    per_seed_csv_path = prefix.with_name(f"{prefix.name}.per-seed.csv")
    columns = (
        "split", "difficulty", "variant", "episodes", "successes", "success_rate",
        "success_rate_ci95_low", "success_rate_ci95_high", "mean_steps", "max_steps",
        "steps_to_solve", "seed_start", "seed_end", "oracle_actions", "oracle_goal",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["steps_to_solve"] = json.dumps(row["steps_to_solve"])
            writer.writerow(serialized)
    lines = [
        "| Split | Variant | Episodes | Success rate | Mean steps | Steps to solve |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('split', '')} {row.get('difficulty', '')} | {row['variant']} | {row['episodes']} | "
            f"{row['success_rate']:.3f} [{row.get('success_rate_ci95_low', 0):.3f}, "
            f"{row.get('success_rate_ci95_high', 1):.3f}] | {row['mean_steps']:.2f} | "
            f"{json.dumps(row['steps_to_solve'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with per_seed_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("seed", "agent", "condition", "success", "steps"),
        )
        writer.writeheader()
        for row in rows:
            for outcome in row["per_seed_outcomes"]:
                writer.writerow({
                    "seed": outcome["seed"],
                    "agent": row["variant"],
                    "condition": row.get("difficulty", "standard"),
                    "success": outcome["success"],
                    "steps": outcome["steps"],
                })
    return csv_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for seed ranges, budgets, and diagnostic modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-start", type=int, default=0)
    parser.add_argument("--held-out-start", type=int, default=10_000)
    parser.add_argument("--final-start", type=int, default=970_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-compute-units", type=int, default=10_000)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/ablation"))
    parser.add_argument("--oracle-diagnostics", action="store_true")
    parser.add_argument("--trace-failures", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--held-out-final", action="store_true")
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--locked-difficulties", action="store_true")
    parser.add_argument("--difficulty", choices=tuple(item.name for item in DIFFICULTIES), default="standard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run strategy or oracle comparisons and optionally emit failure traces."""
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    dev = range(args.dev_start, args.dev_start + args.episodes)
    heldout = range(args.held_out_start, args.held_out_start + args.episodes)
    kwargs = {
        "max_steps": args.max_steps,
        "max_seconds": args.max_seconds,
        "max_compute_units": args.max_compute_units,
    }
    split, range_start = "dev", args.dev_start
    if args.locked_difficulties:
        split = "held_out" if args.held_out_final else "final_report"
        range_start = args.held_out_start if args.held_out_final else args.final_start
        report_seeds = range(range_start, range_start + args.episodes)
        rows = run_locked_difficulties(
            report_seeds,
            split=split,
            max_seconds=args.max_seconds,
            max_compute_units=args.max_compute_units,
        )
    elif args.held_out_final:
        split, range_start = "held_out", args.held_out_start
        rows = run_selected_difficulty(
            heldout,
            args.difficulty,
            max_seconds=args.max_seconds,
            max_compute_units=args.max_compute_units,
        )
    elif args.dev_only:
        rows = run_selected_difficulty(
            dev,
            args.difficulty,
            split="dev",
            max_seconds=args.max_seconds,
            max_compute_units=args.max_compute_units,
        )
    elif args.sensitivity:
        split = "dev_sensitivity"
        rows = run_sensitivity(args.dev_start, max(100, args.episodes), **{
            "max_seconds": args.max_seconds,
            "max_compute_units": args.max_compute_units,
        })
    elif args.oracle_diagnostics:
        split = "dev_oracle"
        rows = run_oracle_ablation(dev, **kwargs)
    else:
        split = "ablation"
        rows = run_ablation(dev, **kwargs)
    csv_path, markdown_path = write_results(rows, args.output_prefix)
    per_seed_csv_path = args.output_prefix.with_name(f"{args.output_prefix.name}.per-seed.csv")
    manifest_run_id = f"{split}-{range_start}-{range_start + args.episodes - 1}"
    manifest_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest_path = write_run_manifest(
        per_seed_csv_path,
        manifest_run_id,
        manifest_config,
        cwd=Path.cwd(),
    )
    traces = None
    if args.trace_failures:
        traces = collect_failure_traces(dev, count=10, **kwargs)
        trace_path = args.output_prefix.with_name(f"{args.output_prefix.name}.failures.md")
        write_trace_report(traces, trace_path)
    print(json.dumps({
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "per_seed_csv": str(per_seed_csv_path),
        "manifest": str(manifest_path),
        "failure_traces": len(traces) if traces is not None else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

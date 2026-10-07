"""Development-only observation diagnostics for ARC exploration policies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from arc_agent.agent.arc_exploration import ArcExplorer
from arc_agent.env.adapter import Environment
from arc_agent.env.arc_adapter import ArcAdapter
from arc_agent.eval.arc_runner import DEFAULT_SPLIT, load_development_games
from arc_agent.perception.frame_encoder import FrameEncoder, PersistentCellMask

MASK_TRAIN_SEEDS = (20261020, 20261021)
MASK_TEST_SEED = 20261022
FRONTIER_TRACE_SEEDS = (20261040, 20261041, 20261042)
LF52_ANALYSIS_SEEDS = tuple(range(20261120, 20261140))
MASK_TRAIN_STEPS = 50
DIAGNOSTIC_BUDGET = 50


@dataclass(frozen=True)
class EpisodeCapture:
    game_id: str
    strategy: str
    seed: int
    levels_completed: int
    wins: bool
    game_over_count: int
    action_count: int
    raw_changed_steps: int
    masked_changed_steps: int
    mask_only_changed_steps: int
    observed_actions: tuple[str, ...]
    trace: tuple[Mapping[str, Any], ...]


def _primary_frame(observation: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(observation["frame"][0])


def _action_for_reset() -> dict[str, str]:
    return {"action": "RESET"}


def capture_episode(
    environment: Environment,
    *,
    strategy: str,
    seed: int,
    max_actions: int,
    persistent_mask: PersistentCellMask | None = None,
    learn_mask: bool = False,
) -> EpisodeCapture:
    observation = environment.reset(seed=seed)
    explorer = ArcExplorer(strategy, seed=seed)
    encoder = FrameEncoder()
    trace: list[Mapping[str, Any]] = []
    observed_actions: list[str] = []
    action_count = 0
    raw_changed_steps = 0
    masked_changed_steps = 0
    mask_only_changed_steps = 0
    game_over_count = 0
    max_levels = int(observation["levels_completed"])
    won = observation["state"] == "WIN"
    current_state = str(observation["state"])
    moved_positions: set[tuple[int, int]] = set()

    while action_count < max_actions and not won:
        before = observation
        if current_state == "GAME_OVER":
            action = _action_for_reset()
            decision_key = None
        else:
            decision = explorer.choose(observation)
            action = dict(decision.action)
            decision_key = decision

        result = environment.step(action)
        action_count += 1
        action_name = str(action["action"])
        observed_actions.append(action_name)
        before_frame = _primary_frame(before)
        after_frame = _primary_frame(result.observation)
        raw_diff = encoder.compare(before_frame, after_frame, action=action_name)
        raw_changed = bool(raw_diff.raw_changed_cells)
        ignored = (
            persistent_mask.mask
            if persistent_mask is not None
            else np.zeros((64, 64), dtype=np.bool_)
        )
        masked_diff = encoder.compare(
            before_frame,
            after_frame,
            ignored_cells=ignored,
        )
        if action_name != "RESET":
            raw_changed_steps += int(raw_changed)
            masked_changed_steps += int(bool(masked_diff.changed_cells))
            mask_only_changed_steps += int(
                raw_changed and not masked_diff.changed_cells
            )
            if persistent_mask is not None and learn_mask:
                persistent_mask.observe(before_frame, after_frame, action_name)
            if decision_key is not None:
                explorer.observe(decision_key, before, result.observation)

        moved = raw_diff.moved_object
        moved_visible = moved is not None and moved.color != 0
        moved_position: tuple[int, int] | None = None
        if moved_visible:
            moved_position = (
                int(round(moved.after.centroid[0])),
                int(round(moved.after.centroid[1])),
            )
            moved_positions.add(moved_position)
        click_target = None
        if decision_key is not None and action_name == "ACTION6":
            arm_key = decision_key.arm_key
            if len(arm_key) >= 6 and arm_key[1] == "object":
                click_target = {
                    "kind": "component_centroid",
                    "color": int(arm_key[2]),
                    "area": int(arm_key[3]),
                    "x": int(arm_key[4]),
                    "y": int(arm_key[5]),
                }
            elif len(arm_key) >= 4 and arm_key[1] == "cell":
                click_target = {
                    "kind": "coverage_cell",
                    "x": int(arm_key[2]),
                    "y": int(arm_key[3]),
                }
        levels_before = int(before["levels_completed"])
        levels_after = int(result.observation["levels_completed"])
        trace.append({
            "step": action_count,
            "action": action,
            "click_target": click_target,
            "frame_changed": raw_changed,
            "masked_frame_changed": bool(masked_diff.changed_cells),
            "changed_cell_count": len(raw_diff.raw_changed_cells),
            "tracked_object_moved": moved_visible,
            "tracked_object_color": moved.color if moved_visible else None,
            "tracked_object_position": moved_position,
            "tracked_object_displacement": moved.displacement if moved_visible else None,
            "no_op": not raw_changed,
            "blocked_or_noop": not raw_changed,
            "levels_before": levels_before,
            "levels_after": levels_after,
            "goal_event": levels_after > levels_before,
            "state_after": str(result.observation["state"]),
            "distinct_tracked_positions_so_far": len(moved_positions),
        })

        observation = result.observation
        current_state = str(observation["state"])
        max_levels = max(max_levels, int(observation["levels_completed"]))
        if current_state == "GAME_OVER":
            game_over_count += 1
        won = current_state == "WIN"

    return EpisodeCapture(
        game_id=str(observation["game_id"]),
        strategy=strategy,
        seed=seed,
        levels_completed=max_levels,
        wins=won,
        game_over_count=game_over_count,
        action_count=action_count,
        raw_changed_steps=raw_changed_steps,
        masked_changed_steps=masked_changed_steps,
        mask_only_changed_steps=mask_only_changed_steps,
        observed_actions=tuple(observed_actions),
        trace=tuple(trace),
    )


def run_diagnostics(
    *,
    environments_dir: str | Path,
    split_path: str | Path,
    output_prefix: str | Path,
    mask_train_seeds: Sequence[int] = MASK_TRAIN_SEEDS,
    mask_test_seed: int = MASK_TEST_SEED,
    frontier_seeds: Sequence[int] = FRONTIER_TRACE_SEEDS,
    lf52_seeds: Sequence[int] = LF52_ANALYSIS_SEEDS,
) -> dict[str, Any]:
    development = load_development_games(split_path)
    if "ls20-9607627b" not in development or "lf52-271a04aa" not in development:
        raise ValueError("required diagnostic games are not in development")
    if not set(mask_train_seeds).isdisjoint({mask_test_seed}):
        raise ValueError("mask training and test seeds must be disjoint")

    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    scorecard_id = arcade.create_scorecard(tags=["arc-agent-dev-diagnostics"])
    output: dict[str, Any] = {
        "analysis_only": True,
        "split": "development",
        "game_ids": list(development),
        "held_out_accessed": False,
        "mask": {},
        "frontier": {},
        "lf52": {},
    }
    try:
        for game_id in development:
            mask = PersistentCellMask(threshold=0.8, minimum_observations=20)
            training_actions: set[str] = set()
            training_action_count = 0
            for seed in mask_train_seeds:
                train_capture = capture_episode(
                    ArcAdapter(game_id, scorecard_id, arcade=arcade),
                    strategy="random",
                    seed=seed,
                    max_actions=MASK_TRAIN_STEPS,
                    persistent_mask=mask,
                    learn_mask=True,
                )
                training_actions.update(
                    action for action in train_capture.observed_actions
                    if action != "RESET"
                )
                training_action_count += sum(
                    action != "RESET" for action in train_capture.observed_actions
                )
            held_capture = capture_episode(
                ArcAdapter(game_id, scorecard_id, arcade=arcade),
                strategy="random",
                seed=mask_test_seed,
                max_actions=MASK_TRAIN_STEPS,
                persistent_mask=mask,
            )
            raw_changed = sum(
                bool(event["frame_changed"]) for event in held_capture.trace
                if event["action"]["action"] != "RESET"
            )
            output["mask"][game_id] = {
                "training_seeds": list(mask_train_seeds),
                "test_seed": mask_test_seed,
                "training_transition_count": mask.observation_count,
                "training_action_types": sorted(training_actions),
                "training_action_count": training_action_count,
                "threshold": mask.threshold,
                "minimum_observations_per_cell": mask.minimum_observations,
                "masked_cell_count": mask.masked_cell_count,
                "masked_cells_xy": [
                    [int(x), int(y)]
                    for y, x in zip(*np.nonzero(mask.mask))
                ],
                "test_raw_changed_steps": raw_changed,
                "test_masked_changed_steps": held_capture.masked_changed_steps,
                "test_steps_changed_only_in_mask": held_capture.mask_only_changed_steps,
                "fraction_raw_changed_steps_only_in_mask": (
                    held_capture.mask_only_changed_steps / raw_changed
                    if raw_changed
                    else 0.0
                ),
                "test_raw_changed_fraction": (
                    raw_changed / sum(
                        action != "RESET" for action in held_capture.observed_actions
                    )
                    if any(action != "RESET" for action in held_capture.observed_actions)
                    else 0.0
                ),
                "test_masked_changed_fraction": (
                    held_capture.masked_changed_steps / sum(
                        action != "RESET" for action in held_capture.observed_actions
                    )
                    if any(action != "RESET" for action in held_capture.observed_actions)
                    else 0.0
                ),
            }

        ls20_traces: dict[str, Any] = {}
        for seed in frontier_seeds:
            capture = capture_episode(
                ArcAdapter("ls20-9607627b", scorecard_id, arcade=arcade),
                strategy="frontier",
                seed=seed,
                max_actions=DIAGNOSTIC_BUDGET,
            )
            ls20_traces[str(seed)] = {
                "action_count": capture.action_count,
                "game_over_count": capture.game_over_count,
                "no_op_count": sum(bool(event["no_op"]) for event in capture.trace),
                "blocked_or_noop_count": sum(
                    bool(event["blocked_or_noop"]) for event in capture.trace
                ),
                "tracked_object_moves": sum(
                    bool(event["tracked_object_moved"]) for event in capture.trace
                ),
                "distinct_tracked_positions": max(
                    (
                        int(event["distinct_tracked_positions_so_far"])
                        for event in capture.trace
                    ),
                    default=0,
                ),
                "trace": list(capture.trace),
            }
        output["frontier"] = {
            "game_id": "ls20-9607627b",
            "seeds": list(frontier_seeds),
            "episodes": ls20_traces,
        }

        lf52_runs: dict[str, Any] = {}
        for strategy in ("random", "click-exploration"):
            per_seed: dict[str, Any] = {}
            for seed in lf52_seeds:
                capture = capture_episode(
                    ArcAdapter("lf52-271a04aa", scorecard_id, arcade=arcade),
                    strategy=strategy,
                    seed=seed,
                    max_actions=50,
                )
                click_sequence = [
                    {
                        "step": int(event["step"]),
                        "action": event["action"],
                        "frame_changed": event["frame_changed"],
                        "goal_event": event["goal_event"],
                        "levels_after": event["levels_after"],
                    }
                    for event in capture.trace
                    if event["action"]["action"] == "ACTION6"
                ]
                per_seed[str(seed)] = {
                    "levels_completed": capture.levels_completed,
                    "win": capture.wins,
                    "game_over_count": capture.game_over_count,
                    "action_count": capture.action_count,
                    "click_sequence": click_sequence,
                    "progress_events": [
                        event for event in capture.trace if event["goal_event"]
                    ],
                }
            lf52_runs[strategy] = per_seed
        output["lf52"] = {
            "game_id": "lf52-271a04aa",
            "seeds": list(lf52_seeds),
            "budget": 50,
            "runs": lf52_runs,
        }
    finally:
        arcade.close_scorecard(scorecard_id)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_path = prefix.with_suffix(".csv")
    _write_mask_summary(summary_path, output["mask"])
    return {
        "json": str(json_path),
        "mask_summary_csv": str(summary_path),
        "game_ids": list(development),
        "held_out_accessed": False,
        "mask_cells": {
            game_id: result["masked_cell_count"]
            for game_id, result in output["mask"].items()
        },
    }


def _write_mask_summary(path: Path, mask_data: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "game_id",
        "training_transition_count",
        "training_action_types",
        "masked_cell_count",
        "test_raw_changed_steps",
        "test_masked_changed_steps",
        "test_steps_changed_only_in_mask",
        "fraction_raw_changed_steps_only_in_mask",
        "test_raw_changed_fraction",
        "test_masked_changed_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for game_id, row in mask_data.items():
            writer.writerow({
                "game_id": game_id,
                "training_transition_count": row["training_transition_count"],
                "training_action_types": json.dumps(row["training_action_types"]),
                **{field: row[field] for field in fields[3:]},
            })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_diagnostics(
        environments_dir=args.environments_dir,
        split_path=args.split_file,
        output_prefix=args.output_prefix,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create seed-paired outcome and exact McNemar summaries from sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

AGENTS = ("random", "click-exploration")
BUDGETS = (50, 150, 400)
PAIR_FIELDS = (
    "budget",
    "game_id",
    "seed",
    "random_levels_completed",
    "click_levels_completed",
    "random_any_level",
    "click_any_level",
)
SUMMARY_FIELDS = (
    "budget",
    "scope",
    "game_id",
    "paired_episodes",
    "random_episodes_with_level",
    "click_episodes_with_level",
    "random_only",
    "click_only",
    "discordant_pairs",
    "mcnemar_exact_p",
)
CHANGE_FIELDS = (
    "game_id",
    "action_class",
    "transitions",
    "frame_changes",
    "frame_change_fraction",
    "mean_changed_cells_all",
    "median_changed_cells_all",
    "p90_changed_cells_all",
    "max_changed_cells_all",
    "mean_changed_cells_nonzero",
    "median_changed_cells_nonzero",
    "p90_changed_cells_nonzero",
    "max_changed_cells_nonzero",
)


def exact_mcnemar_p(random_only: int, click_only: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant pairs."""
    if random_only < 0 or click_only < 0:
        raise ValueError("discordant-pair counts cannot be negative")
    discordant = random_only + click_only
    if discordant == 0:
        return 1.0
    tail_count = min(random_only, click_only)
    one_sided = sum(math.comb(discordant, k) for k in range(tail_count + 1))
    return min(1.0, 2.0 * one_sided / (2**discordant))


def _read_budget(input_dir: Path, budget: int) -> dict[tuple[str, int], dict[str, int]]:
    path = input_dir / f"sweep-{budget}.csv"
    outcomes: dict[tuple[str, int], dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            agent = row["agent"]
            if agent not in AGENTS:
                raise ValueError(f"unexpected agent {agent!r} in {path}")
            key = (row["game_id"], int(row["seed"]))
            if agent in outcomes.setdefault(key, {}):
                raise ValueError(f"duplicate {agent} row for {key} in {path}")
            outcomes[key][agent] = int(row["levels_completed"])
    if not outcomes or any(set(pair) != set(AGENTS) for pair in outcomes.values()):
        raise ValueError(f"{path} does not contain complete agent pairs")
    return outcomes


def _summary_row(
    budget: int,
    scope: str,
    game_id: str,
    pairs: Iterable[tuple[int, int]],
) -> dict[str, int | str | float]:
    outcomes = tuple(pairs)
    random_successes = sum(random_levels > 0 for random_levels, _ in outcomes)
    click_successes = sum(click_levels > 0 for _, click_levels in outcomes)
    random_only = sum(
        random_levels > 0 and click_levels == 0
        for random_levels, click_levels in outcomes
    )
    click_only = sum(
        click_levels > 0 and random_levels == 0
        for random_levels, click_levels in outcomes
    )
    return {
        "budget": budget,
        "scope": scope,
        "game_id": game_id,
        "paired_episodes": len(outcomes),
        "random_episodes_with_level": random_successes,
        "click_episodes_with_level": click_successes,
        "random_only": random_only,
        "click_only": click_only,
        "discordant_pairs": random_only + click_only,
        "mcnemar_exact_p": exact_mcnemar_p(random_only, click_only),
    }


def analyze(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write per-seed paired outcomes and pooled/per-game summaries."""
    paired_rows: list[dict[str, int | str]] = []
    summary_rows: list[dict[str, int | str | float]] = []
    for budget in BUDGETS:
        outcomes = _read_budget(input_dir, budget)
        by_game: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for (game_id, seed), agents in sorted(outcomes.items()):
            random_levels = agents["random"]
            click_levels = agents["click-exploration"]
            by_game[game_id].append((random_levels, click_levels))
            paired_rows.append({
                "budget": budget,
                "game_id": game_id,
                "seed": seed,
                "random_levels_completed": random_levels,
                "click_levels_completed": click_levels,
                "random_any_level": int(random_levels > 0),
                "click_any_level": int(click_levels > 0),
            })
        all_pairs = [pair for game_pairs in by_game.values() for pair in game_pairs]
        summary_rows.append(_summary_row(budget, "pooled", "ALL", all_pairs))
        for game_id, game_pairs in sorted(by_game.items()):
            summary_rows.append(
                _summary_row(budget, "game", game_id, game_pairs)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "arc_sweep_paired_episodes.csv"
    summary_path = output_dir / "arc_sweep_mcnemar_summary.csv"
    _write_csv(paired_path, PAIR_FIELDS, paired_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    return paired_path, summary_path


def summarize_change_sizes(
    traces_path: Path,
) -> list[dict[str, int | str | float]]:
    """Summarize observed cell changes by game and click/other/reset action."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    with traces_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            episode = json.loads(line)
            for transition in episode["trajectory"]:
                action_name = transition["action"]["action"]
                action_class = (
                    "reset"
                    if action_name == "RESET"
                    else "click"
                    if action_name == "ACTION6"
                    else "other"
                )
                changed_cells = int(transition["changed_cell_count"])
                frame_changed = bool(transition["frame_changed"])
                if (changed_cells > 0) != frame_changed:
                    raise ValueError(
                        f"frame-change flag disagrees with cell count at "
                        f"{traces_path}:{line_number}"
                    )
                grouped[(episode["game_id"], action_class)].append(changed_cells)

    games = sorted({game_id for game_id, _ in grouped})
    result: list[dict[str, int | str | float]] = []
    for game_id in games:
        for action_class in ("click", "other", "reset"):
            values = grouped.get((game_id, action_class), [])
            nonzero = [value for value in values if value > 0]
            result.append({
                "game_id": game_id,
                "action_class": action_class,
                "transitions": len(values),
                "frame_changes": len(nonzero),
                "frame_change_fraction": (
                    len(nonzero) / len(values) if values else 0.0
                ),
                "mean_changed_cells_all": _mean(values),
                "median_changed_cells_all": _median(values),
                "p90_changed_cells_all": _nearest_rank(values, 0.9),
                "max_changed_cells_all": max(values, default=0),
                "mean_changed_cells_nonzero": _mean(nonzero),
                "median_changed_cells_nonzero": _median(nonzero),
                "p90_changed_cells_nonzero": _nearest_rank(nonzero, 0.9),
                "max_changed_cells_nonzero": max(nonzero, default=0),
            })
    return result


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--change-traces", type=Path)
    args = parser.parse_args()
    paired, summary = analyze(args.input_dir, args.output_dir)
    print(f"paired outcomes: {paired}")
    print(f"McNemar summary: {summary}")
    if args.change_traces:
        change_path = args.output_dir / "arc_sweep_400_action_change_summary.csv"
        _write_csv(
            change_path,
            CHANGE_FIELDS,
            summarize_change_sizes(args.change_traces),
        )
        print(f"action change summary: {change_path}")


if __name__ == "__main__":
    main()

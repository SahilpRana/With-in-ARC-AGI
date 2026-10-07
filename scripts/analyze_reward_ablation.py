"""Pair binary and log change-reward outcomes from matched dev evaluations."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

PAIR_FIELDS = (
    "game_id",
    "seed",
    "binary_levels",
    "log_levels",
    "binary_any_level",
    "log_any_level",
    "binary_success",
    "log_success",
    "binary_score",
    "log_score",
    "score_difference_log_minus_binary",
)
SUMMARY_FIELDS = (
    "scope",
    "game_id",
    "paired_episodes",
    "binary_episodes_with_level",
    "log_episodes_with_level",
    "binary_only",
    "log_only",
    "discordant_pairs",
    "mcnemar_exact_p",
    "binary_mean_levels",
    "log_mean_levels",
    "binary_wins",
    "log_wins",
    "binary_mean_score",
    "log_mean_score",
    "mean_score_difference_log_minus_binary",
)


def exact_mcnemar_p(binary_only: int, log_only: int) -> float:
    """Return the two-sided exact binomial test for discordant pairs."""
    if binary_only < 0 or log_only < 0:
        raise ValueError("discordant-pair counts cannot be negative")
    discordant = binary_only + log_only
    if discordant == 0:
        return 1.0
    tail = min(binary_only, log_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2 * probability / (2**discordant))


def _read(path: Path) -> dict[tuple[str, int], dict[str, float | int]]:
    result: dict[tuple[str, int], dict[str, float | int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["game_id"], int(row["seed"]))
            if key in result:
                raise ValueError(f"duplicate episode key {key} in {path}")
            result[key] = {
                "levels": int(row["levels_completed"]),
                "success": int(row["success"].lower() == "true"),
                "score": float(row["score"]),
            }
    return result


def _summary(
    scope: str,
    game_id: str,
    pairs: list[tuple[dict[str, float | int], dict[str, float | int]]],
) -> dict[str, str | int | float]:
    binary_levels = [int(binary["levels"]) for binary, _ in pairs]
    log_levels = [int(log["levels"]) for _, log in pairs]
    binary_positive = sum(levels > 0 for levels in binary_levels)
    log_positive = sum(levels > 0 for levels in log_levels)
    binary_only = sum(
        int(binary["levels"]) > 0 and int(log["levels"]) == 0
        for binary, log in pairs
    )
    log_only = sum(
        int(log["levels"]) > 0 and int(binary["levels"]) == 0
        for binary, log in pairs
    )
    binary_scores = [float(binary["score"]) for binary, _ in pairs]
    log_scores = [float(log["score"]) for _, log in pairs]
    score_diff = (
        sum(log_scores) / len(log_scores) - sum(binary_scores) / len(binary_scores)
        if pairs
        else 0.0
    )
    return {
        "scope": scope,
        "game_id": game_id,
        "paired_episodes": len(pairs),
        "binary_episodes_with_level": binary_positive,
        "log_episodes_with_level": log_positive,
        "binary_only": binary_only,
        "log_only": log_only,
        "discordant_pairs": binary_only + log_only,
        "mcnemar_exact_p": exact_mcnemar_p(binary_only, log_only),
        "binary_mean_levels": sum(binary_levels) / len(pairs) if pairs else 0.0,
        "log_mean_levels": sum(log_levels) / len(pairs) if pairs else 0.0,
        "binary_wins": sum(int(binary["success"]) for binary, _ in pairs),
        "log_wins": sum(int(log["success"]) for _, log in pairs),
        "binary_mean_score": (
            sum(binary_scores) / len(pairs) if pairs else 0.0
        ),
        "log_mean_score": sum(log_scores) / len(pairs) if pairs else 0.0,
        "mean_score_difference_log_minus_binary": score_diff,
    }


def analyze(
    binary_path: Path,
    log_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Validate matched seeds and write paired episode/per-game CSVs."""
    binary = _read(binary_path)
    log = _read(log_path)
    if set(binary) != set(log):
        raise ValueError("binary and log runs must have exactly the same game/seed pairs")
    grouped: dict[str, list[tuple[dict[str, float | int], dict[str, float | int]]]] = defaultdict(list)
    paired_rows: list[dict[str, str | int | float]] = []
    for key in sorted(binary):
        binary_row = binary[key]
        log_row = log[key]
        game_id, seed = key
        grouped[game_id].append((binary_row, log_row))
        paired_rows.append({
            "game_id": game_id,
            "seed": seed,
            "binary_levels": binary_row["levels"],
            "log_levels": log_row["levels"],
            "binary_any_level": int(int(binary_row["levels"]) > 0),
            "log_any_level": int(int(log_row["levels"]) > 0),
            "binary_success": binary_row["success"],
            "log_success": log_row["success"],
            "binary_score": binary_row["score"],
            "log_score": log_row["score"],
            "score_difference_log_minus_binary": (
                float(log_row["score"]) - float(binary_row["score"])
            ),
        })

    all_pairs = [pair for pairs in grouped.values() for pair in pairs]
    summary_rows = [_summary("pooled", "ALL", all_pairs)]
    summary_rows.extend(
        _summary("game", game_id, pairs)
        for game_id, pairs in sorted(grouped.items())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "reward_ablation_paired_episodes.csv"
    summary_path = output_dir / "reward_ablation_summary.csv"
    _write(paired_path, PAIR_FIELDS, paired_rows)
    _write(summary_path, SUMMARY_FIELDS, summary_rows)
    return paired_path, summary_path


def _write(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str | int | float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    paired, summary = analyze(args.binary, args.log, args.output_dir)
    print(f"paired outcomes: {paired}")
    print(f"per-game summary: {summary}")


if __name__ == "__main__":
    main()

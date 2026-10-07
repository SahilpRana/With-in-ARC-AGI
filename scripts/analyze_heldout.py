"""Analyze paired held-out outcomes and report game-level Wilson intervals."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PRIMARY_AGENTS = ("random", "click-exploration")
MOVEMENT_AGENT = "frontier"
WILSON_Z = 1.959963984540054
PAIRED_FIELDS = (
    "game_id",
    "seed",
    "random_levels_completed",
    "click_levels_completed",
    "random_any_level",
    "click_any_level",
    "random_score",
    "click_score",
    "random_success",
    "click_success",
)
PRIMARY_FIELDS = (
    "paired_episodes",
    "random_episodes_with_level",
    "random_fraction_with_level",
    "random_wilson_95_low",
    "random_wilson_95_high",
    "click_episodes_with_level",
    "click_fraction_with_level",
    "click_wilson_95_low",
    "click_wilson_95_high",
    "random_only",
    "click_only",
    "discordant_pairs",
    "mcnemar_exact_p",
    "random_mean_score",
    "click_mean_score",
    "random_mean_levels",
    "click_mean_levels",
    "random_wins",
    "click_wins",
)
PER_GAME_FIELDS = ("game_id", *PRIMARY_FIELDS)
FRONTIER_FIELDS = (
    "game_id",
    "episodes",
    "episodes_with_level",
    "fraction_with_level",
    "wilson_95_low",
    "wilson_95_high",
    "mean_score",
    "mean_levels_completed",
    "wins",
)


def exact_mcnemar_p(random_only: int, click_only: int) -> float:
    """Return the two-sided exact McNemar p-value."""
    if random_only < 0 or click_only < 0:
        raise ValueError("discordant-pair counts cannot be negative")
    discordant = random_only + click_only
    if discordant == 0:
        return 1.0
    tail = min(random_only, click_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
    if trials == 0:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = successes / trials
    z_squared = WILSON_Z**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        WILSON_Z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _read_episodes(path: Path) -> dict[tuple[str, int], dict[str, dict[str, float | int]]]:
    episodes: dict[
        tuple[str, int],
        dict[str, dict[str, float | int]],
    ] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"game_id", "agent", "seed", "score", "success", "levels_completed"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (row["game_id"], int(row["seed"]))
                agent = row["agent"]
                levels = int(row["levels_completed"])
                score = float(row["score"])
                success_text = row["success"].strip().lower()
                if success_text not in {"true", "false", "1", "0"}:
                    raise ValueError("success must be true/false or 1/0")
                success = int(success_text in {"true", "1"})
                if levels < 0 or not math.isfinite(score):
                    raise ValueError("levels and score must be finite and non-negative")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid row at {path}:{line_number}: {error}") from error
            if agent in episodes[key]:
                raise ValueError(
                    f"duplicate {agent!r} episode for game/seed {key} in {path}"
                )
            episodes[key][agent] = {
                "levels": levels,
                "score": score,
                "success": success,
            }
    if not episodes:
        raise ValueError(f"{path} contains no episode rows")
    return episodes


def _paired_rows(
    episodes: dict[tuple[str, int], dict[str, dict[str, float | int]]],
) -> list[dict[str, int | float | str]]:
    rows = []
    for (game_id, seed), agents in sorted(episodes.items()):
        if set(agents) != set(PRIMARY_AGENTS):
            raise ValueError(
                f"{game_id}/{seed} must contain exactly random and click-exploration"
            )
        random = agents["random"]
        click = agents["click-exploration"]
        random_levels = int(random["levels"])
        click_levels = int(click["levels"])
        rows.append({
            "game_id": game_id,
            "seed": seed,
            "random_levels_completed": random_levels,
            "click_levels_completed": click_levels,
            "random_any_level": int(random_levels > 0),
            "click_any_level": int(click_levels > 0),
            "random_score": float(random["score"]),
            "click_score": float(click["score"]),
            "random_success": int(random["success"]),
            "click_success": int(click["success"]),
        })
    return rows


def _primary_summary(
    rows: Iterable[dict[str, int | float | str]],
) -> dict[str, int | float]:
    pairs = list(rows)
    count = len(pairs)
    if not count:
        raise ValueError("primary analysis requires at least one paired episode")
    random_positive = sum(int(row["random_any_level"]) for row in pairs)
    click_positive = sum(int(row["click_any_level"]) for row in pairs)
    random_only = sum(
        int(row["random_any_level"]) == 1 and int(row["click_any_level"]) == 0
        for row in pairs
    )
    click_only = sum(
        int(row["click_any_level"]) == 1 and int(row["random_any_level"]) == 0
        for row in pairs
    )
    random_low, random_high = wilson_interval(random_positive, count)
    click_low, click_high = wilson_interval(click_positive, count)
    return {
        "paired_episodes": count,
        "random_episodes_with_level": random_positive,
        "random_fraction_with_level": random_positive / count,
        "random_wilson_95_low": random_low,
        "random_wilson_95_high": random_high,
        "click_episodes_with_level": click_positive,
        "click_fraction_with_level": click_positive / count,
        "click_wilson_95_low": click_low,
        "click_wilson_95_high": click_high,
        "random_only": random_only,
        "click_only": click_only,
        "discordant_pairs": random_only + click_only,
        "mcnemar_exact_p": exact_mcnemar_p(random_only, click_only),
        "random_mean_score": sum(float(row["random_score"]) for row in pairs) / count,
        "click_mean_score": sum(float(row["click_score"]) for row in pairs) / count,
        "random_mean_levels": (
            sum(int(row["random_levels_completed"]) for row in pairs) / count
        ),
        "click_mean_levels": (
            sum(int(row["click_levels_completed"]) for row in pairs) / count
        ),
        "random_wins": sum(int(row["random_success"]) for row in pairs),
        "click_wins": sum(int(row["click_success"]) for row in pairs),
    }


def _frontier_rows(
    path: Path,
) -> tuple[list[dict[str, int | float | str]], dict[str, int | float]]:
    episodes = _read_episodes(path)
    rows: list[dict[str, int | float | str]] = []
    for (game_id, _seed), agents in sorted(episodes.items()):
        if set(agents) != {MOVEMENT_AGENT}:
            raise ValueError(
                f"frontier input must contain only {MOVEMENT_AGENT!r} episodes"
            )
        episode = agents[MOVEMENT_AGENT]
        levels = int(episode["levels"])
        rows.append({
            "game_id": game_id,
            "levels_completed": levels,
            "any_level": int(levels > 0),
            "score": float(episode["score"]),
            "success": int(episode["success"]),
        })
    games = sorted({str(row["game_id"]) for row in rows})
    if len(games) != 2:
        raise ValueError("frontier input must contain exactly two movement games")
    result: list[dict[str, int | float | str]] = []
    for game_id in games:
        game_rows = [row for row in rows if row["game_id"] == game_id]
        count = len(game_rows)
        positive = sum(int(row["any_level"]) for row in game_rows)
        low, high = wilson_interval(positive, count)
        result.append({
            "game_id": game_id,
            "episodes": count,
            "episodes_with_level": positive,
            "fraction_with_level": positive / count,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "mean_score": sum(float(row["score"]) for row in game_rows) / count,
            "mean_levels_completed": (
                sum(int(row["levels_completed"]) for row in game_rows) / count
            ),
            "wins": sum(int(row["success"]) for row in game_rows),
        })
    summary = {
        "episodes": len(rows),
        "episodes_with_level": sum(int(row["any_level"]) for row in rows),
        "mean_score": sum(float(row["score"]) for row in rows) / len(rows),
        "mean_levels_completed": (
            sum(int(row["levels_completed"]) for row in rows) / len(rows)
        ),
        "wins": sum(int(row["success"]) for row in rows),
        "movement_games": len(games),
    }
    return result, summary


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    click_random_csv: Path,
    output_dir: Path,
    frontier_csv: Path | None = None,
) -> tuple[Path, ...]:
    """Write paired primary outcomes and separate per-game/frontier summaries."""
    episodes = _read_episodes(click_random_csv)
    paired = _paired_rows(episodes)
    overall = _primary_summary(paired)
    per_game = []
    for game_id in sorted({str(row["game_id"]) for row in paired}):
        summary = _primary_summary(row for row in paired if row["game_id"] == game_id)
        per_game.append({"game_id": game_id, **summary})

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "heldout_paired_episodes.csv"
    summary_path = output_dir / "heldout_primary_summary.csv"
    per_game_path = output_dir / "heldout_per_game.csv"
    _write_csv(paired_path, PAIRED_FIELDS, paired)
    _write_csv(summary_path, PRIMARY_FIELDS, [overall])
    _write_csv(per_game_path, PER_GAME_FIELDS, per_game)
    paths: list[Path] = [paired_path, summary_path, per_game_path]
    if frontier_csv is not None:
        frontier_per_game, frontier_summary = _frontier_rows(frontier_csv)
        frontier_path = output_dir / "heldout_frontier_per_game.csv"
        frontier_summary_path = output_dir / "heldout_frontier_summary.csv"
        _write_csv(frontier_path, FRONTIER_FIELDS, frontier_per_game)
        _write_csv(frontier_summary_path, tuple(frontier_summary), [frontier_summary])
        paths.extend((frontier_path, frontier_summary_path))
    return tuple(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--click-random-csv", type=Path, required=True)
    parser.add_argument("--frontier-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    for path in analyze(args.click_random_csv, args.output_dir, args.frontier_csv):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

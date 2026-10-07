"""Run matched-budget agents on development games or guarded held-out games."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from arc_agent.agent.arc_game_agent import ArcEpisodeResult, ArcGameAgent
from arc_agent.env.arc_adapter import ArcAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT = PROJECT_ROOT / "results" / "game_split.json"
DEFAULT_SURVEY = PROJECT_ROOT / "results" / "game_survey.csv"
DEFAULT_ENVIRONMENTS = Path(r"C:\Users\sahil\arc_data\environment_files")
AGENTS = ("random", "click-exploration", "frontier")
CSV_FIELDS = (
    "game_id",
    "agent",
    "seed",
    "score",
    "success",
    "levels_completed",
    "actions_used",
    "action_count",
    "frame_change_fraction",
    "game_over_count",
    "stopped_by_budget",
)


@dataclass(frozen=True)
class EpisodePlan:
    game_id: str
    agent: str
    seed: int


@dataclass(frozen=True)
class ScoredEpisode:
    plan: EpisodePlan
    result: ArcEpisodeResult
    score: float


def load_development_games(
    split_path: str | Path,
    requested_games: Sequence[str] | None = None,
) -> tuple[str, ...]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if not split.get("pre_registered_before_tuning"):
        raise ValueError("game split is not marked as pre-registered")
    development = tuple(split["development_game_ids"])
    if len(development) != 13 or len(set(development)) != 13:
        raise ValueError("registered development split must contain 13 unique games")
    if requested_games is None:
        return development
    requested = tuple(dict.fromkeys(requested_games))
    disallowed = set(requested) - set(development)
    if disallowed:
        raise ValueError(
            "only registered development games may be evaluated; "
            f"rejected IDs: {', '.join(sorted(disallowed))}"
        )
    if not requested:
        raise ValueError("at least one development game must be selected")
    return requested


def load_heldout_games(
    split_path: str | Path,
    requested_games: Sequence[str] | None = None,
) -> tuple[str, ...]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if not split.get("pre_registered_before_tuning"):
        raise ValueError("game split is not marked as pre-registered")
    heldout = tuple(split["held_out_game_ids"])
    development = set(split["development_game_ids"])
    if len(heldout) != 12 or len(heldout) != len(set(heldout)):
        raise ValueError("registered held-out split must contain 12 unique games")
    if set(heldout) & development:
        raise ValueError("development and held-out games must be disjoint")
    if requested_games is None:
        return heldout
    requested = tuple(dict.fromkeys(requested_games))
    disallowed = set(requested) - set(heldout)
    if disallowed:
        raise ValueError(
            "only registered held-out games may be evaluated; "
            f"rejected IDs: {', '.join(sorted(disallowed))}"
        )
    if not requested:
        raise ValueError("at least one held-out game must be selected")
    return requested


def _git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _require_git(args: Sequence[str], message: str) -> str:
    result = _git(args)
    if result.returncode != 0:
        raise ValueError(message)
    return result.stdout.strip()


def _heldout_prerequisites() -> tuple[str, str]:
    tag_hash = _require_git(
        ["rev-parse", "--verify", "refs/tags/frozen-agent-v1^{commit}"],
        "held-out evaluation requires the frozen-agent-v1 tag",
    )
    ancestor = _git(["merge-base", "--is-ancestor", "frozen-agent-v1", "HEAD"])
    if ancestor.returncode != 0:
        raise ValueError("HEAD must be at or after frozen-agent-v1")

    for args in (
        ["diff", "--quiet", "frozen-agent-v1", "HEAD", "--", "src/arc_agent/agent/"],
        ["diff", "--quiet", "HEAD", "--", "src/arc_agent/agent/"],
        ["diff", "--cached", "--quiet", "--", "src/arc_agent/agent/"],
    ):
        if _git(args).returncode != 0:
            raise ValueError("agent code has changed since frozen-agent-v1")
    if _require_git(
        ["status", "--porcelain", "--", "src/arc_agent/agent/"],
        "could not verify agent-code worktree state",
    ):
        raise ValueError("agent code has changed since frozen-agent-v1")

    notes = _require_git(
        ["show", "HEAD:writeup/notes.md"],
        "held-out pre-registration must be committed in writeup/notes.md",
    )
    heading = "Held-out pre-registration"
    if not any(
        line.lstrip("#").strip() == heading and line.startswith("#")
        for line in notes.splitlines()
    ):
        raise ValueError(
            "held-out pre-registration must be committed in writeup/notes.md"
        )
    prereg_hash = _require_git(
        ["log", "-1", "--format=%H", "-S", heading, "HEAD", "--", "writeup/notes.md"],
        "could not identify the committed held-out pre-registration",
    )
    if not prereg_hash:
        raise ValueError(
            "held-out pre-registration must be committed in writeup/notes.md"
        )
    return tag_hash, prereg_hash


def _episode_keys(plan: Sequence[EpisodePlan]) -> set[tuple[str, int, str]]:
    return {(item.game_id, item.seed, item.agent) for item in plan}


def _assert_heldout_run_once(
    results_dir: str | Path,
    plan: Sequence[EpisodePlan],
) -> None:
    planned = _episode_keys(plan)
    results_path = Path(results_dir)
    for manifest_path in results_path.rglob("*.manifest.json"):
        if "debug" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_name = str(
            manifest.get("split", manifest.get("game_split", ""))
        ).lower()
        is_heldout = (
            split_name.startswith("held")
            or manifest.get("held_out_final") is True
            or "heldout" in manifest_path.name.lower()
            or "held_out" in manifest_path.name.lower()
        )
        if not is_heldout:
            continue
        episodes = manifest.get("planned_episodes")
        if episodes is not None:
            previous = {
                (item["game_id"], int(item["seed"]), item["agent"])
                for item in episodes
            }
            if planned & previous:
                raise ValueError(
                    "held-out episodes may only be run once; "
                    "use --allow-rerun-for-debug for an isolated debug run"
                )
        else:
            seeds = {int(seed) for seed in manifest.get("seeds", [])}
            config = manifest.get("config", {})
            seeds.update(int(seed) for seed in config.get("seeds", []))
            seed_start = config.get(
                "seed_start",
                manifest.get("seed_start", manifest.get("held_out_start")),
            )
            seeds_per_game = config.get(
                "seeds_per_game",
                manifest.get("seeds_per_game"),
            )
            if seed_start is not None and seeds_per_game is not None:
                seeds.update(
                    range(int(seed_start), int(seed_start) + int(seeds_per_game))
                )
            if seeds & {seed for _, seed, _ in planned}:
                raise ValueError(
                    "a held-out manifest already exists for one or more seeds; "
                    "use --allow-rerun-for-debug for an isolated debug run"
                )

    string_keys = {
        (game_id, str(seed), agent)
        for game_id, seed, agent in planned
    }
    for result_path in (*results_path.rglob("*.csv"), *results_path.rglob("*.jsonl")):
        if "debug" in result_path.parts:
            continue
        if result_path.suffix == ".csv":
            with result_path.open(encoding="utf-8", newline="") as stream:
                rows: Iterable[Mapping[str, Any]] = csv.DictReader(stream)
                for row in rows:
                    key = (row.get("game_id"), row.get("seed"), row.get("agent"))
                    if key in string_keys:
                        raise ValueError(
                            "held-out episode results already exist for this run; "
                            "use --allow-rerun-for-debug for an isolated debug run"
                        )
        else:
            with result_path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    key = (
                        record.get("game_id"),
                        str(record.get("seed")),
                        record.get("agent", record.get("strategy")),
                    )
                    if key in string_keys:
                        raise ValueError(
                            "held-out episode results already exist for this run; "
                            "use --allow-rerun-for-debug for an isolated debug run"
                        )


def load_movement_games(
    survey_path: str | Path,
    development_games: Iterable[str],
) -> frozenset[str]:
    allowed = set(development_games)
    with Path(survey_path).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        movement = {
            row["game_id"]
            for row in rows
            if row["data_role"] == "analysis_only_not_agent_input"
            and row["movement_candidate_by_action_surface"] == "True"
            and row["game_id"] in allowed
        }
    return frozenset(movement)


def build_episode_plan(
    game_ids: Sequence[str],
    movement_games: Iterable[str],
    strategies: Sequence[str],
    seeds: Sequence[int],
) -> tuple[EpisodePlan, ...]:
    if not strategies or not seeds:
        raise ValueError("at least one agent and one seed are required")
    invalid = set(strategies) - set(AGENTS)
    if invalid:
        raise ValueError(f"unknown agents: {', '.join(sorted(invalid))}")
    movement = set(movement_games)
    plans: list[EpisodePlan] = []
    for strategy in strategies:
        eligible_games = (
            [game_id for game_id in game_ids if game_id in movement]
            if strategy == "frontier"
            else list(game_ids)
        )
        plans.extend(
            EpisodePlan(game_id, strategy, seed)
            for game_id in eligible_games
            for seed in seeds
        )
    return tuple(plans)


def _score_for_game(scorecard: Any, game_id: str) -> float:
    matches = [item for item in scorecard.environments if item.id == game_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"scorecard returned {len(matches)} records for game {game_id}"
        )
    runs = matches[0].runs
    if len(runs) != 1:
        raise RuntimeError(
            f"expected one scorecard run for {game_id}, received {len(runs)}"
        )
    return float(runs[0].score)


def _revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_csv(path: Path, episodes: Sequence[ScoredEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for episode in episodes:
            writer.writerow(_csv_row(episode))


def _csv_row(episode: ScoredEpisode) -> dict[str, Any]:
    result = episode.result
    return {
        "game_id": result.game_id,
        "agent": result.strategy,
        "seed": result.seed,
        "score": episode.score,
        "success": result.success,
        "levels_completed": result.levels_completed,
        "actions_used": json.dumps(result.actions_used),
        "action_count": result.action_count,
        "frame_change_fraction": result.changed_step_fraction,
        "game_over_count": result.game_over_count,
        "stopped_by_budget": result.stopped_by_budget,
    }


def _summary_table(episodes: Sequence[ScoredEpisode]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ScoredEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(
            (episode.plan.game_id, episode.plan.agent),
            [],
        ).append(episode)
    table: list[dict[str, Any]] = []
    for (game_id, agent), runs in sorted(grouped.items()):
        table.append({
            "game_id": game_id,
            "agent": agent,
            "seeds": len(runs),
            "mean_score": sum(run.score for run in runs) / len(runs),
            "mean_levels_completed": sum(
                run.result.levels_completed for run in runs
            ) / len(runs),
            "mean_actions": sum(
                run.result.action_count for run in runs
            ) / len(runs),
            "mean_frame_change_fraction": sum(
                run.result.changed_step_fraction for run in runs
            ) / len(runs),
            "game_over_count": sum(run.result.game_over_count for run in runs),
            "wins": sum(run.result.success for run in runs),
        })
    return table


def _write_jsonl(path: Path, episodes: Sequence[ScoredEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            record = episode.result.to_dict()
            record.update({
                "score": episode.score,
                "agent": episode.plan.agent,
                "seed": episode.plan.seed,
            })
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _run_evaluation(
    *,
    environments_dir: str | Path,
    split_path: str | Path,
    survey_path: str | Path,
    output_prefix: str | Path,
    strategies: Sequence[str],
    seeds: Sequence[int],
    max_actions: int,
    requested_games: Sequence[str] | None = None,
    split_name: str = "development",
    tag_hash: str | None = None,
    preregistration_commit_hash: str | None = None,
    debug_run: bool = False,
) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"ARC game evaluation requires Python 3.12, got {platform.python_version()}"
        )
    if max_actions < 1:
        raise ValueError("max_actions must be at least 1")

    if split_name == "heldout":
        game_ids = load_heldout_games(split_path, requested_games)
    else:
        game_ids = load_development_games(split_path, requested_games)
    movement_games = load_movement_games(survey_path, game_ids)
    plan = build_episode_plan(game_ids, movement_games, strategies, seeds)

    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    manifest_path = prefix.with_name(f"{prefix.name}.{run_id}.manifest.json")
    manifest = {
        "run_id": run_id,
        "git_revision": _revision(),
        "python_version": platform.python_version(),
        "toolkit_version": version("arc-agi"),
        "operation_mode": "OFFLINE",
        "game_split": split_name,
        "game_ids": list(game_ids),
        "movement_only_game_ids": sorted(movement_games),
        "config": {
            "agents": list(strategies),
            "seeds": list(seeds),
            "max_actions_including_reset": max_actions,
            "competition_data_dir": str(environments_dir),
            "split_file": str(split_path),
            "survey_analysis_file": str(survey_path),
        },
        "artifacts": {"csv": str(prefix.with_suffix(".csv"))},
        "status": "running",
    }
    if split_name == "development":
        manifest["artifacts"]["jsonl"] = str(prefix.with_suffix(".jsonl"))
    if split_name == "heldout":
        manifest["split"] = split_name
        manifest["planned_episodes"] = [
            {
                "game_id": item.game_id,
                "agent": item.agent,
                "seed": item.seed,
            }
            for item in plan
        ]
        manifest["tag_hash"] = tag_hash
        manifest["preregistration_commit_hash"] = preregistration_commit_hash
        manifest["debug_run"] = debug_run
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    scored: list[ScoredEpisode] = []
    try:
        for episode_plan in plan:
            scorecard_id = arcade.create_scorecard(
                tags=[
                    f"arc-agent-{split_name}-eval",
                    f"agent:{episode_plan.agent}",
                    f"seed:{episode_plan.seed}",
                ]
            )
            try:
                environment = ArcAdapter(
                    episode_plan.game_id,
                    scorecard_id,
                    environments_dir=environments_dir,
                    arcade=arcade,
                )
                agent = ArcGameAgent(
                    environment,
                    strategy=episode_plan.agent,
                    seed=episode_plan.seed,
                    max_actions=max_actions,
                )
                result = agent.run_episode()
                scorecard = arcade.close_scorecard(scorecard_id)
                if scorecard is None:
                    raise RuntimeError(
                        f"scorecard {scorecard_id} returned no result"
                    )
                score = _score_for_game(scorecard, episode_plan.game_id)
                scored.append(ScoredEpisode(episode_plan, result, score))
            except Exception:
                arcade.close_scorecard(scorecard_id)
                raise

        csv_path = prefix.with_suffix(".csv")
        _write_csv(csv_path, scored)
        manifest["status"] = "completed"
        manifest["episode_count"] = len(scored)
        manifest["artifacts"]["csv"] = str(csv_path)
        result: dict[str, Any] = {
            "manifest": str(manifest_path),
            "csv": str(csv_path),
            "episodes": len(scored),
            "table": _summary_table(scored),
        }
        if split_name == "development":
            jsonl_path = prefix.with_suffix(".jsonl")
            _write_jsonl(jsonl_path, scored)
            manifest["artifacts"]["jsonl"] = str(jsonl_path)
            result["jsonl"] = str(jsonl_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


def run_development_evaluation(
    *,
    environments_dir: str | Path,
    split_path: str | Path,
    survey_path: str | Path,
    output_prefix: str | Path,
    strategies: Sequence[str],
    seeds: Sequence[int],
    max_actions: int,
    requested_games: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _run_evaluation(
        environments_dir=environments_dir,
        split_path=split_path,
        survey_path=survey_path,
        output_prefix=output_prefix,
        strategies=strategies,
        seeds=seeds,
        max_actions=max_actions,
        requested_games=requested_games,
    )


def run_heldout_evaluation(
    *,
    environments_dir: str | Path,
    split_path: str | Path,
    survey_path: str | Path,
    output_prefix: str | Path | None,
    strategies: Sequence[str],
    seeds: Sequence[int],
    max_actions: int,
    requested_games: Sequence[str] | None = None,
    allow_rerun_for_debug: bool = False,
) -> dict[str, Any]:
    tag_hash, preregistration_commit_hash = _heldout_prerequisites()
    all_game_ids = load_heldout_games(split_path)
    movement_games = load_movement_games(survey_path, all_game_ids)
    if requested_games is None:
        game_ids = all_game_ids
    else:
        game_ids = load_heldout_games(split_path, requested_games)
        if set(strategies) == {"frontier"}:
            if len(movement_games) != 2 or set(game_ids) != movement_games:
                raise ValueError(
                    "frontier held-out runs must use both registered movement games"
                )
        elif set(game_ids) != set(all_game_ids):
            raise ValueError(
                "random and click-exploration held-out runs must include all 12 games"
            )
    plan = build_episode_plan(game_ids, movement_games, strategies, seeds)
    results_dir = PROJECT_ROOT / "results"

    if allow_rerun_for_debug:
        run_id = uuid.uuid4().hex
        output_prefix = (
            results_dir
            / "debug"
            / "heldout"
            / run_id
            / "heldout-debug"
        )
    else:
        _assert_heldout_run_once(results_dir, plan)
        if output_prefix is None:
            agent_label = "-".join(strategies)
            output_prefix = (
                results_dir
                / f"heldout-{agent_label}-{seeds[0]}-{seeds[-1]}"
            )
        prefix = Path(output_prefix)
        if not prefix.resolve().is_relative_to(results_dir.resolve()):
            raise ValueError("official held-out results must be written under results/")
        if "debug" in prefix.resolve().parts:
            raise ValueError("official held-out output cannot use the debug path")
        for artifact in (prefix.with_suffix(".csv"), prefix.with_suffix(".jsonl")):
            if artifact.exists():
                raise ValueError(
                    f"refusing to overwrite existing held-out artifact: {artifact}"
                )

    return _run_evaluation(
        environments_dir=environments_dir,
        split_path=split_path,
        survey_path=survey_path,
        output_prefix=output_prefix,
        strategies=strategies,
        seeds=seeds,
        max_actions=max_actions,
        requested_games=game_ids,
        split_name="heldout",
        tag_hash=tag_hash,
        preregistration_commit_hash=preregistration_commit_hash,
        debug_run=allow_rerun_for_debug,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments-dir", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--survey-file", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument("--games", nargs="*")
    parser.add_argument("--seed-start", type=int, default=20261006)
    parser.add_argument("--seeds-per-game", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--allow-rerun-for-debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds_per_game < 1:
        raise SystemExit("--seeds-per-game must be at least 1")
    if args.max_actions < 1:
        raise SystemExit("--max-actions must be at least 1")
    seeds = tuple(args.seed_start + index for index in range(args.seeds_per_game))
    if args.split == "development":
        if args.allow_rerun_for_debug:
            raise SystemExit("--allow-rerun-for-debug is only valid for --split heldout")
        if args.output_prefix is None:
            raise SystemExit("--output-prefix is required for development evaluations")
        games = load_development_games(args.split_file, args.games)
        summary = run_development_evaluation(
            environments_dir=args.environments_dir,
            split_path=args.split_file,
            survey_path=args.survey_file,
            output_prefix=args.output_prefix,
            strategies=args.agents,
            seeds=seeds,
            max_actions=args.max_actions,
            requested_games=games,
        )
    else:
        summary = run_heldout_evaluation(
            environments_dir=args.environments_dir,
            split_path=args.split_file,
            survey_path=args.survey_file,
            output_prefix=args.output_prefix,
            strategies=args.agents,
            seeds=seeds,
            max_actions=args.max_actions,
            requested_games=args.games,
            allow_rerun_for_debug=args.allow_rerun_for_debug,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

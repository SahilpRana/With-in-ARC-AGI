"""Offline analysis survey of local ARC-AGI-3 games using random actions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


def _observation_frames(observation: Any) -> list[Any]:
    """Return the observation's runtime frame layers or fail explicitly."""
    frames = list(observation.frame)
    if not frames:
        raise RuntimeError(f"Observation for {observation.game_id} has no frame layers")
    return frames


def _frame_changed(previous: list[Any], current: list[Any]) -> bool:
    if len(previous) != len(current):
        return True
    return any(
        first.shape != second.shape or not (first == second).all()
        for first, second in zip(previous, current)
    )


def survey_game(
    arcade: Arcade,
    game_id: str,
    seed: int,
    max_steps: int,
    scorecard_id: str,
) -> dict[str, object]:
    environment = arcade.make(
        game_id,
        seed=seed,
        scorecard_id=scorecard_id,
        save_recording=False,
        include_frame_data=True,
    )
    if environment is None:
        raise RuntimeError(f"Arcade could not create local game {game_id}")

    observation = environment.observation_space
    if observation is None:
        raise RuntimeError(f"Game {game_id} produced no initial observation")
    previous_frames = _observation_frames(observation)
    all_frames = [frame for frame in previous_frames]
    color_values: set[int] = set()
    for frame in previous_frames:
        color_values.update(int(value) for value in set(frame.ravel().tolist()))

    available_actions: set[str] = set()
    changed_steps = 0
    action_steps = 0
    levels_completed = int(observation.levels_completed)
    win_levels = int(observation.win_levels)
    terminal = observation.state in (GameState.WIN, GameState.GAME_OVER)
    terminal_states = {observation.state.value} if terminal else set()

    rng = random.Random(seed)
    while action_steps < max_steps and not terminal:
        legal_actions = [
            action for action in environment.action_space
            if action is not GameAction.RESET
        ]
        if not legal_actions:
            break
        available_actions.update(action.name for action in legal_actions)
        action = rng.choice(legal_actions)
        action_data = (
            {"x": rng.randrange(64), "y": rng.randrange(64)}
            if action is GameAction.ACTION6
            else None
        )
        next_observation = environment.step(action, data=action_data)
        if next_observation is None:
            raise RuntimeError(f"step returned None for {game_id} at step {action_steps + 1}")

        current_frames = _observation_frames(next_observation)
        changed_steps += int(_frame_changed(previous_frames, current_frames))
        action_steps += 1
        all_frames.extend(current_frames)
        for frame in current_frames:
            color_values.update(int(value) for value in set(frame.ravel().tolist()))
        levels_completed = max(levels_completed, int(next_observation.levels_completed))
        win_levels = max(win_levels, int(next_observation.win_levels))
        terminal = next_observation.state in (GameState.WIN, GameState.GAME_OVER)
        if terminal:
            terminal_states.add(next_observation.state.value)
        previous_frames = current_frames

    # Preserve the actions available on the reset observation, including a
    # possible terminal initial state.
    initial_actions = [
        action.name for action in environment.action_space
        if action is not GameAction.RESET
    ]
    available_actions.update(initial_actions)
    ordered_actions = sorted(available_actions)
    frame_shapes = sorted({tuple(frame.shape) for frame in all_frames})
    movement_surface = (
        bool(ordered_actions)
        and all(name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} for name in ordered_actions)
    )
    return {
        "data_role": "analysis_only_not_agent_input",
        "game_id": game_id,
        "seed": seed,
        "steps_requested": max_steps,
        "steps_run": action_steps,
        "available_actions": json.dumps(ordered_actions),
        "action5_available": "ACTION5" in available_actions,
        "action6_available": "ACTION6" in available_actions,
        "action7_available": "ACTION7" in available_actions,
        "frame_shapes": json.dumps([list(shape) for shape in frame_shapes]),
        "number_of_colors_seen": len(color_values),
        "frame_changed_steps": changed_steps,
        "frame_changed_fraction": changed_steps / action_steps if action_steps else 0.0,
        "levels_completed_observed": levels_completed,
        "win_levels_observed": win_levels,
        "reached_win_or_game_over": terminal,
        "terminal_states_observed": json.dumps(sorted(terminal_states)),
        "movement_candidate_by_action_surface": movement_surface,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Game survey requires Python 3.12, got {sys.version.split()[0]}")
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(args.environments_dir),
        recordings_dir=str(Path(tempfile.gettempdir()) / "arc-agent-survey-recordings"),
    )
    games = sorted(arcade.get_environments(), key=lambda game: game.game_id)
    if not games:
        raise RuntimeError(f"No games found under {args.environments_dir}")

    scorecard_id = arcade.create_scorecard(tags=["offline-analysis-game-survey"])
    rows = [
        survey_game(
            arcade,
            game.game_id,
            args.seed + index,
            args.steps,
            scorecard_id,
        )
        for index, game in enumerate(games)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    movement_games = [
        row["game_id"] for row in rows
        if row["movement_candidate_by_action_surface"]
    ]
    action6_games = [row["game_id"] for row in rows if row["action6_available"]]
    print(json.dumps({
        "analysis_only": True,
        "games_surveyed": len(rows),
        "steps_requested_per_game": args.steps,
        "movement_only_by_action_surface": len(movement_games),
        "movement_candidate_game_ids": movement_games,
        "games_with_action6": len(action6_games),
        "action6_game_ids": action6_games,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Budgeted observe-act loop for real ARC-AGI-3 games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from arc_agent.agent.arc_exploration import ArcDecision, ArcExplorer
from arc_agent.env.adapter import Environment, Observation, StepResult
from arc_agent.perception.frame_encoder import FrameEncoder


@dataclass(frozen=True)
class ArcEpisodeResult:
    """Visible episode outcome, action budget use, and JSON-ready trajectory."""

    game_id: str
    seed: int
    strategy: str
    success: bool
    stopped_by_budget: bool
    action_count: int
    levels_completed: int
    game_over_count: int
    frame_changed_steps: int
    total_steps: int
    actions_used: tuple[str, ...]
    trajectory: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def changed_step_fraction(self) -> float:
        return self.frame_changed_steps / self.total_steps if self.total_steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "strategy": self.strategy,
            "success": self.success,
            "stopped_by_budget": self.stopped_by_budget,
            "action_count": self.action_count,
            "levels_completed": self.levels_completed,
            "game_over_count": self.game_over_count,
            "frame_changed_steps": self.frame_changed_steps,
            "total_steps": self.total_steps,
            "changed_step_fraction": self.changed_step_fraction,
            "actions_used": list(self.actions_used),
            "trajectory": list(self.trajectory),
        }


class ArcGameAgent:
    """Play an ARC game using only normalized, agent-visible observations."""

    def __init__(
        self,
        environment: Environment,
        *,
        strategy: str,
        seed: int,
        max_actions: int,
    ) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be at least 1")
        self.environment = environment
        self.strategy = strategy
        self.seed = seed
        self.max_actions = max_actions
        self.explorer = ArcExplorer(strategy, seed=seed)

    def run_episode(self) -> ArcEpisodeResult:
        observation = self.environment.reset(seed=self.seed)
        trajectory: list[Mapping[str, Any]] = []
        actions_used: list[str] = []
        action_count = 0
        game_over_count = 0
        changed_steps = 0
        max_levels_completed = int(observation["levels_completed"])
        success = observation["state"] == "WIN"
        last_state = str(observation["state"])

        while action_count < self.max_actions and not success:
            if last_state == "GAME_OVER":
                before = observation
                result = self.environment.step({"action": "RESET"})
                action_count += 1
                actions_used.append("RESET")
                reset_changed = _frame_changed(before, result.observation)
                changed_steps += int(reset_changed)
                trajectory.append(_trajectory_record(
                    action_count,
                    {"action": "RESET"},
                    before,
                    result,
                    reset_changed,
                ))
                observation = result.observation
                last_state = str(observation["state"])
                max_levels_completed = max(
                    max_levels_completed,
                    int(observation["levels_completed"]),
                )
                success = last_state == "WIN"
                continue

            if self.strategy == "frontier" and not _movement_only(observation):
                raise RuntimeError(
                    "frontier policy requires only ACTION1-ACTION4 in the current action space"
                )

            before = observation
            decision = self.explorer.choose(observation)
            result = self.environment.step(decision.action)
            action_count += 1
            action_name = str(decision.action["action"])
            actions_used.append(action_name)
            frame_changed = _frame_changed(before, result.observation)
            changed_steps += int(frame_changed)
            self.explorer.observe(decision, before, result.observation)
            trajectory.append(_trajectory_record(
                action_count,
                decision.action,
                before,
                result,
                frame_changed,
            ))
            observation = result.observation
            last_state = str(observation["state"])
            max_levels_completed = max(
                max_levels_completed,
                int(observation["levels_completed"]),
            )
            if last_state == "GAME_OVER":
                game_over_count += 1
            success = last_state == "WIN"

        return ArcEpisodeResult(
            game_id=str(observation["game_id"]),
            seed=self.seed,
            strategy=self.strategy,
            success=success,
            stopped_by_budget=action_count >= self.max_actions and not success,
            action_count=action_count,
            levels_completed=max_levels_completed,
            game_over_count=game_over_count,
            frame_changed_steps=changed_steps,
            total_steps=action_count,
            actions_used=tuple(actions_used),
            trajectory=tuple(trajectory),
        )


def _frame_changed(before: Observation, after: Observation) -> bool:
    before_frame = before["frame"][0]
    after_frame = after["frame"][0]
    return bool((before_frame != after_frame).any())


def _movement_only(observation: Observation) -> bool:
    actions = set(observation["available_actions"])
    return bool(actions) and actions <= {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}


def _trajectory_record(
    index: int,
    action: Mapping[str, Any],
    before: Observation,
    result: StepResult,
    frame_changed: bool,
) -> dict[str, Any]:
    levels_before = int(before["levels_completed"])
    levels_after = int(result.observation["levels_completed"])
    return {
        "action_index": index,
        "action": dict(action),
        "state_before": str(before["state"]),
        "state_after": str(result.observation["state"]),
        "levels_before": levels_before,
        "levels_after": levels_after,
        "goal_event": levels_after > levels_before,
        "frame_changed": frame_changed,
        "changed_cell_count": int(
            (before["frame"][0] != result.observation["frame"][0]).sum()
        ),
    }

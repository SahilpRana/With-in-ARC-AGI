"""Episode controller wiring observation, learning, planning, and action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agent.agent.policy import AgentPolicy, Mode, evidence_gate_open
from arc_agent.env.adapter import Action, Environment, Observation
from arc_agent.exploration.explorer import (
    FrontierExplorer,
    InformationGainExplorer,
    NoPermittedActionError,
    RandomExplorer,
    is_oscillating,
    observation_signature,
)
from arc_agent.memory.episode_memory import EpisodeMemory, Transition
from arc_agent.perception.state_encoder import encode_observation
from arc_agent.planning.planner import GridPlanner
from arc_agent.utils.budget import Budget
from arc_agent.world_model.hypothesis import Hypothesis
from arc_agent.world_model.goal import GoalCandidate, GoalInferer
from arc_agent.world_model.inducer import RuleInducer
from arc_agent.world_model.verifier import RuleVerifier


@dataclass(frozen=True)
class EpisodeResult:
    """Summary and auditable trajectory from one agent-environment episode."""

    seed: int
    success: bool
    steps: int
    total_reward: float
    stopped_by_budget: bool
    memory: EpisodeMemory = field(compare=False)
    hypotheses: tuple[Hypothesis, ...]
    goals: tuple[GoalCandidate, ...]
    modes: tuple[Mode, ...]
    budget_limits: dict[str, float | int]
    elapsed_seconds: float
    compute_units_used: int
    trace: tuple[dict[str, Any], ...] = ()
    stalled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to JSON-serializable run-log data."""
        return {
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "stopped_by_budget": self.stopped_by_budget,
            "stalled": self.stalled,
            "modes": [mode.value for mode in self.modes],
            "hypotheses": [
                {
                    "rule": hypothesis.rule,
                    "confidence": hypothesis.confidence,
                    "supports": hypothesis.supports,
                    "contradictions": hypothesis.contradictions,
                    "no_ops": hypothesis.no_ops,
                    "tested_count": hypothesis.tested_count,
                    "evidence": hypothesis.evidence,
                }
                for hypothesis in self.hypotheses
            ],
            "goals": [
                {"position": goal.position, "evidence": goal.evidence, "score": goal.score}
                for goal in self.goals
            ],
            "budget_limits": self.budget_limits,
            "elapsed_seconds": self.elapsed_seconds,
            "compute_units_used": self.compute_units_used,
            "trace": list(self.trace),
            "trajectory": [
                {
                    "state": dict(transition.state),
                    "action": dict(transition.action),
                    "next_state": dict(transition.next_state),
                    "reward": transition.reward,
                    "terminated": transition.terminated,
                }
                for transition in self.memory.transitions
            ],
        }


class Agent:
    """Adapt within an episode using fresh evidence, a model hypothesis, and plans."""

    def __init__(
        self,
        environment: Environment,
        *,
        seed: int = 0,
        max_steps: int = 100,
        max_seconds: float = 60.0,
        max_compute_units: int = 10_000,
        explorer_kind: str = "random",
        min_action_tests: int = 2,
        confidence_threshold: float = 0.7,
        strategy: str = "hypothesis_gated",
        oracle_actions: bool = False,
        oracle_goal: bool = False,
        goal_confidence_threshold: float = 0.8,
    ) -> None:
        """Create an episode agent with deterministic baseline strategies."""
        self.environment = environment
        self.seed = seed
        self.budget = Budget(
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_compute_units=max_compute_units,
        )
        moves = tuple(environment.action_spec.get("move", ()))
        actions: tuple[Action, ...] = tuple({"move": move} for move in moves)
        if not actions:
            actions = ({"action": "noop"},)
        if explorer_kind == "random":
            self.explorer = RandomExplorer(actions, seed=seed)
        elif explorer_kind == "information_gain":
            self.explorer = InformationGainExplorer(actions, seed=seed)
        elif explorer_kind == "frontier":
            self.explorer = FrontierExplorer(actions, seed=seed)
        else:
            raise ValueError(f"unknown explorer_kind {explorer_kind!r}")
        action_names = tuple(str(action.get("move", "")) for action in actions)
        self.planner = GridPlanner()
        self.policy = AgentPolicy()
        self.inducer = RuleInducer(actions=action_names)
        self.verifier = RuleVerifier()
        self.goal_inferer = GoalInferer()
        self.min_action_tests = min_action_tests
        self.confidence_threshold = confidence_threshold
        if strategy not in ("random_only", "hypothesis_gated"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.oracle_actions = oracle_actions
        self.oracle_goal = oracle_goal
        if not 0.0 < goal_confidence_threshold <= 1.0:
            raise ValueError("goal_confidence_threshold must be in (0, 1]")
        self.goal_confidence_threshold = goal_confidence_threshold
        self.frontier_explorer = FrontierExplorer(actions, seed=seed)
        self._oracle: dict[str, Any] = {}

    def run_episode(self) -> EpisodeResult:
        """Run observe -> infer -> plan -> act until done or budget exhaustion."""
        observation = self.environment.reset(seed=self.seed)
        self._door_position: tuple[int, int] | None = None
        self._key_position: tuple[int, int] | None = None
        self._has_key = False
        oracle_state = getattr(self.environment, "_oracle_state", None)
        self._oracle = oracle_state() if callable(oracle_state) else {}
        memory = EpisodeMemory()
        hypotheses: list[Hypothesis] = []
        goals: tuple[GoalCandidate, ...] = ()
        modes: list[Mode] = []
        trace: list[dict[str, Any]] = []
        total_reward = 0.0
        success = False
        environment_done = False
        stalled = False

        while not self.budget.exhausted and not environment_done:
            state = encode_observation(observation)
            hypotheses = self.inducer.propose(memory)
            hypotheses = [self.verifier.verify(item, memory) for item in hypotheses]
            goals = self.goal_inferer.infer(memory)
            self.budget.consume_compute(2)
            special_cells = self._special_cells(observation)
            if self._door_position is None and memory.transitions:
                previous = memory.transitions[-1]
                if previous.info.get("blocked"):
                    previous_action = str(previous.action.get("move", ""))
                    candidates = [item for item in hypotheses if item.action == previous_action]
                    if candidates:
                        best = max(candidates, key=lambda item: (item.confidence, item.supports))
                        before_player = previous.state.get("player")
                        if before_player is not None:
                            predicted = (
                                int(before_player[0]) + best.expected_displacement[0],
                                int(before_player[1]) + best.expected_displacement[1],
                            )
                            if predicted in special_cells:
                                self._door_position = predicted
            if self._door_position is not None and not self._has_key:
                other_specials = [cell for cell in special_cells if cell != self._door_position]
                if other_specials:
                    self._key_position = other_specials[0]
            inferred_goal = (
                goals[0].position
                if goals and state.target is None and goals[0].score >= self.goal_confidence_threshold
                else None
            )
            if self.oracle_goal:
                inferred_goal = self._oracle.get("goal")
            intermediate_goal = self._key_position if not self._has_key else None
            planning_goal = intermediate_goal or inferred_goal
            blocked_cells = (
                {self._door_position}
                if self._door_position is not None and not self._has_key
                else set()
            )
            active_hypotheses = tuple(hypotheses)
            if self.oracle_actions:
                oracle_mapping = self._oracle.get("action_mapping", {})
                active_hypotheses = tuple(
                    Hypothesis(
                        action=action,
                        expected_displacement=tuple(oracle_mapping[action]),
                        supports=10,
                    )
                    for action in self.inducer.actions
                    if action in oracle_mapping
                )
            plan = self.planner.plan(
                state,
                goal=planning_goal,
                hypotheses=active_hypotheses,
                blocked_cells=blocked_cells,
            )
            self.budget.consume_compute(1)
            if self.budget.exhausted:
                break
            hypotheses_confident = evidence_gate_open(
                hypotheses,
                self.inducer.actions,
                min_action_tests=self.min_action_tests,
                confidence_threshold=self.confidence_threshold,
            )
            if self.strategy == "random_only":
                plan = None
                hypotheses_confident = False
            position_history = [
                tuple(transition.next_state["player"])
                for transition in memory.transitions
                if transition.next_state.get("player") is not None
            ]
            if state.player is not None:
                position_history.append(state.player)
            oscillating = is_oscillating(position_history)
            if oscillating:
                plan = None
                hypotheses_confident = False
            current_position = state.player
            visited_positions = {
                tuple(position)
                for transition in memory.transitions
                for position in (
                    transition.state.get("player"),
                    transition.next_state.get("player"),
                )
                if position is not None
            }
            if current_position is not None:
                visited_positions.add(current_position)
            blocked_actions_here = {
                str(transition.action.get("move"))
                for transition in memory.transitions
                if transition.info.get("blocked")
                and tuple(transition.state.get("player", ())) == current_position
                and observation_signature(transition.state) == observation_signature(observation)
            }
            explorer = self.frontier_explorer if oscillating or isinstance(self.explorer, FrontierExplorer) else self.explorer
            forbidden_actions = {
                str(transition.action.get("move"))
                for transition in memory.transitions
                if transition.info.get("blocked")
                and observation_signature(transition.state) == observation_signature(observation)
            }
            try:
                mode, action = self.policy.select_action(
                    plan, hypotheses_confident or self.oracle_actions,
                    explorer,
                    active_hypotheses,
                    current_position=current_position,
                    visited_positions=visited_positions,
                    blocked_actions_here=blocked_actions_here,
                    forbidden_actions=forbidden_actions,
                    grid=state.grid,
                    grid_origin=state.grid_origin,
                )
            except NoPermittedActionError:
                stalled = True
                break

            # The planner uses visible structure when available; exploration is
            # the fallback for observations without a known route.
            result = self.environment.step(action)
            self.budget.consume_step(compute_units=1)
            modes.append(mode)
            trace.append({
                "step": len(memory) + 1,
                "hypotheses": [
                    {
                        "rule": item.rule,
                        "confidence": item.confidence,
                        "supports": item.supports,
                        "contradictions": item.contradictions,
                        "no_ops": item.no_ops,
                    }
                    for item in active_hypotheses
                ],
                "inferred_goal": inferred_goal,
                "intermediate_goal": intermediate_goal,
                "door_position": self._door_position,
                "key_position": self._key_position,
                "has_key": self._has_key,
                "planned_path": list(plan) if plan else None,
                "chosen_action": dict(action),
                "mode": mode.value,
                "player": observation.get("player"),
                "next_player": result.observation.get("player"),
                "reward": result.reward,
                "progress": result.info.get("progress", False),
                "blocked": result.info.get("blocked", False),
                "goal_matches_oracle": (
                    tuple(inferred_goal) == tuple(self._oracle.get("goal"))
                    if inferred_goal is not None and self._oracle.get("goal") is not None
                    else None
                ),
                "prediction_correct": self._prediction_correct(
                    action, observation, result.observation, active_hypotheses
                ),
            })
            memory.add(
                Transition(
                    state=observation,
                    action=action,
                    next_state=result.observation,
                    reward=result.reward,
                    terminated=result.terminated,
                    info=result.info,
                )
            )
            if result.info.get("key_collected"):
                self._has_key = True
                self._key_position = None
            observation = result.observation
            total_reward += result.reward
            success = result.terminated and result.reward > 0
            environment_done = result.terminated or result.truncated

        if not hypotheses:
            hypotheses = self.inducer.propose(memory)
            hypotheses = [self.verifier.verify(item, memory) for item in hypotheses]
        goals = self.goal_inferer.infer(memory)
        return EpisodeResult(
            seed=self.seed,
            success=success,
            steps=len(memory),
            total_reward=total_reward,
            stopped_by_budget=not environment_done and self.budget.exhausted,
            memory=memory,
            hypotheses=tuple(hypotheses),
            goals=goals,
            modes=tuple(modes),
            budget_limits={
                "max_steps": self.budget.max_steps,
                "max_seconds": self.budget.max_seconds,
                "max_compute_units": self.budget.max_compute_units,
            },
            elapsed_seconds=self.budget.elapsed_seconds,
            compute_units_used=self.budget.compute_units_used,
            trace=tuple(trace),
            stalled=stalled,
        )

    @staticmethod
    def _special_cells(observation: Observation) -> set[tuple[int, int]]:
        """Return global coordinates of generic special tiles in the visible grid."""
        grid = observation.get("grid")
        if grid is None:
            return set()
        origin = observation.get("window_origin", (0, 0))
        origin_x, origin_y = int(origin[0]), int(origin[1])
        return {
            (origin_x + x, origin_y + y)
            for y, row in enumerate(grid)
            for x, tile in enumerate(row)
            if int(tile) == 4
        }

    @staticmethod
    def _prediction_correct(
        action: Action,
        before: Observation,
        after: Observation,
        hypotheses: tuple[Hypothesis, ...],
    ) -> bool | None:
        """Compare actual displacement with the strongest active action candidate."""
        action_name = action.get("move")
        start, end = before.get("player"), after.get("player")
        if action_name is None or start is None or end is None or tuple(start) == tuple(end):
            return None
        candidates = [item for item in hypotheses if item.action == action_name]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: (item.confidence, item.supports))
        displacement = (int(end[0]) - int(start[0]), int(end[1]) - int(start[1]))
        return displacement == best.expected_displacement

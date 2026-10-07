"""Tests for information-gain exploration and goal inference."""

import pytest

from arc_agent.exploration.explorer import (
    FrontierExplorer,
    InformationGainExplorer,
    NoPermittedActionError,
    RandomExplorer,
    is_oscillating,
    observation_signature,
)
from arc_agent.agent.policy import evidence_gate_open
from arc_agent.memory.episode_memory import EpisodeMemory, Transition
from arc_agent.world_model.goal import GoalInferer
from arc_agent.world_model.hypothesis import Hypothesis


def test_information_gain_prefers_untried_action_when_disagreement_ties() -> None:
    actions = tuple({"move": name} for name in ("up", "down"))
    hypotheses = [
        Hypothesis("up", delta, supports=1, contradictions=1)
        for delta in ((0, -1), (0, 1))
    ]
    hypotheses.extend(
        Hypothesis("down", delta)
        for delta in ((0, -1), (0, 1))
    )

    selected = InformationGainExplorer(actions, seed=5).choose(hypotheses)

    assert selected == {"move": "down"}


def test_information_gain_prioritizes_unvisited_cells_over_disagreement() -> None:
    actions = tuple({"move": name} for name in ("up", "down"))
    hypotheses = [
        Hypothesis("up", (0, -1), supports=3),
        Hypothesis("up", (0, 1), supports=3),
        Hypothesis("down", (0, 1), supports=3),
    ]

    selected = InformationGainExplorer(actions, seed=0).choose(
        hypotheses,
        current_position=(1, 1),
        visited_positions={(1, 1), (1, 0)},
    )

    assert selected == {"move": "down"}


def test_information_gain_avoids_repeating_action_blocked_here() -> None:
    actions = tuple({"move": name} for name in ("left", "right"))
    hypotheses = [
        Hypothesis("left", (-1, 0), supports=2),
        Hypothesis("right", (1, 0), supports=2),
    ]

    selected = InformationGainExplorer(actions, seed=0).choose(
        hypotheses,
        current_position=(0, 1),
        visited_positions={(0, 1), (-1, 1), (1, 1)},
        blocked_actions_here={"left"},
    )

    assert selected == {"move": "right"}


def test_blocked_action_memory_is_state_scoped_and_resets_when_map_changes() -> None:
    actions = ({"move": "left"}, {"move": "right"})
    same_state = {"player": (0, 0), "grid": ((2, 0), (1, 0)), "window_origin": (0, 0)}
    equivalent_state = {"player": (0, 0), "grid": ((0, 0), (1, 0)), "window_origin": (0, 0)}
    changed_map = {"player": (0, 0), "grid": ((2, 0), (0, 0)), "window_origin": (0, 0)}

    assert observation_signature(same_state) == observation_signature(equivalent_state)
    assert observation_signature(same_state) != observation_signature(changed_map)
    selected = RandomExplorer(actions, seed=1).choose(forbidden_actions={"left"})
    assert selected == {"move": "right"}
    with pytest.raises(NoPermittedActionError):
        RandomExplorer(({"move": "left"},), seed=1).choose(forbidden_actions={"left"})


def test_frontier_explorer_bfs_targets_nearest_reachable_unvisited_cell() -> None:
    actions = tuple({"move": name} for name in ("up", "down", "left", "right"))
    hypotheses = [
        Hypothesis(name, delta, supports=3)
        for name, delta in (
            ("up", (0, -1)),
            ("down", (0, 1)),
            ("left", (-1, 0)),
            ("right", (1, 0)),
        )
    ]
    selected = FrontierExplorer(actions, seed=0).choose(
        hypotheses,
        current_position=(1, 1),
        visited_positions={(1, 1), (1, 0)},
        grid=((0, 0, 0), (0, 2, 0), (1, 0, 0)),
    )

    assert selected in ({"move": "down"}, {"move": "right"})


def test_oscillation_detector_triggers_on_recent_revisits_only() -> None:
    assert is_oscillating([(0, 0), (1, 0)] * 4)
    assert not is_oscillating([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)])


def test_goal_inferer_prioritizes_reward_and_progress_positions() -> None:
    memory = EpisodeMemory(
        transitions=[
            Transition(
                state={"player": (1, 1)},
                action={"move": "right"},
                next_state={"player": (2, 1)},
                info={"progress": True},
            ),
            Transition(
                state={"player": (2, 1)},
                action={"move": "down"},
                next_state={"player": (2, 2)},
                reward=1.0,
                terminated=True,
            ),
        ]
    )

    goals = GoalInferer().infer(memory)

    assert goals[0].position == (2, 2)
    assert goals[0].score == 1.0


def test_goal_inference_does_not_commit_one_ambiguous_progress_signal() -> None:
    inferer = GoalInferer()
    one_signal = EpisodeMemory(
        transitions=[
            Transition(
                state={"player": (2, 2)},
                action={"move": "right"},
                next_state={"player": (3, 2)},
                reward=0.1,
                info={"progress": True},
            )
        ]
    )
    candidates = inferer.infer(one_signal)
    assert len(candidates) == 5
    assert max(item.score for item in candidates) == 0.2

    for position in ((3, 2), (2, 3), (4, 3)):
        one_signal.add(
            Transition(
                state={"player": (0, 0)},
                action={"move": "right"},
                next_state={"player": position},
                reward=0.1,
                info={"progress": True},
            )
        )
    candidates = inferer.infer(one_signal)
    assert len(candidates) == 1
    assert candidates[0].position == (3, 3)
    assert candidates[0].score == 1.0


def test_planning_evidence_gate_requires_k_tests_for_every_action() -> None:
    actions = ("up", "down")
    candidates = [
        Hypothesis(action, displacement, supports=1)
        for action in actions
        for displacement in ((0, -1), (0, 1))
    ]

    assert not evidence_gate_open(
        candidates,
        actions,
        min_action_tests=2,
        confidence_threshold=0.6,
    )
    for candidate in candidates:
        candidate.no_ops = 2
    assert evidence_gate_open(
        candidates,
        actions,
        min_action_tests=2,
        confidence_threshold=0.6,
    )
    for candidate in candidates[-2:]:
        candidate.supports = 0
        candidate.contradictions = 4
    assert not evidence_gate_open(
        candidates,
        actions,
        min_action_tests=2,
        confidence_threshold=0.6,
    )
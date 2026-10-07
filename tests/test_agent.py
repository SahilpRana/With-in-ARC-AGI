"""End-to-end episode and reproducibility checks."""

from arc_agent.agent.agent import Agent
from arc_agent.agent.policy import Mode
from arc_agent.env.mock_env import HiddenDifficulty, HiddenRuleGridEnv, MockEnvironment
from arc_agent.memory.episode_memory import EpisodeMemory, Transition
from arc_agent.world_model.hypothesis import Hypothesis
from arc_agent.world_model.verifier import RuleVerifier


def test_agent_solves_seeded_mock_episode_and_records_learning_cycle() -> None:
    result = Agent(
        MockEnvironment(), seed=4, max_steps=40, max_seconds=10, max_compute_units=500
    ).run_episode()

    assert result.success
    assert result.steps == len(result.memory)
    assert result.hypotheses
    assert any(hypothesis.confidence > 0.5 for hypothesis in result.hypotheses)
    assert Mode.EXPLORE in result.modes
    assert Mode.EXPLOIT in result.modes
    assert result.to_dict()["trajectory"]


def test_agent_obeys_action_budget() -> None:
    result = Agent(
        MockEnvironment(), seed=0, max_steps=1, max_seconds=10, max_compute_units=100
    ).run_episode()

    assert result.steps <= 1
    assert result.stopped_by_budget or result.success


def test_verifier_separates_noops_from_support_and_contradictions() -> None:
    memory = EpisodeMemory(
        transitions=[
            Transition(state={"player": (1, 1)}, action={"move": "right"}, next_state={"player": (2, 1)}),
            Transition(state={"player": (2, 1)}, action={"move": "right"}, next_state={"player": (2, 1)}),
            Transition(state={"player": (2, 1)}, action={"move": "right"}, next_state={"player": (2, 2)}),
        ]
    )
    hypothesis = Hypothesis(action="right", expected_displacement=(1, 0))

    verified = RuleVerifier().verify(hypothesis, memory)

    assert verified.supports == 1
    assert verified.no_ops == 1
    assert verified.contradictions == 1
    assert verified.tested_count == 3
    assert verified.posterior == (2, 2)
    assert verified.confidence == 0.5


def test_agent_uses_key_as_intermediate_goal_after_detecting_locked_door() -> None:
    environment = HiddenRuleGridEnv(
        width=7,
        height=7,
        max_steps=60,
        difficulty=HiddenDifficulty(
            action_remapping=False,
            hidden_goal=True,
            mechanic="key_door",
        ),
    )
    result = Agent(
        environment,
        seed=2,
        max_steps=60,
        oracle_actions=True,
        oracle_goal=True,
    ).run_episode()

    assert result.success
    assert any(step["door_position"] is not None for step in result.trace)
    assert any(step["intermediate_goal"] is not None for step in result.trace)
    assert any(step["has_key"] for step in result.trace)

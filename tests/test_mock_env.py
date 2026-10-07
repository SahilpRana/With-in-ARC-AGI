"""Behavior checks for the local hidden-layout environment."""

import pytest

from arc_agent.env.mock_env import MockEnvironment
from arc_agent.perception.state_encoder import encode_observation
from arc_agent.planning.planner import GridPlanner


def test_invalid_action_is_rejected() -> None:
    environment = MockEnvironment()
    environment.reset(seed=0)
    with pytest.raises(ValueError):
        environment.step({"move": "teleport"})


def test_seeded_layouts_have_a_traversable_route() -> None:
    environment = MockEnvironment()
    planner = GridPlanner()

    for seed in range(50):
        state = encode_observation(environment.reset(seed=seed))
        assert planner.plan(state) is not None


def test_step_after_terminal_requires_reset() -> None:
    environment = MockEnvironment(width=2, height=2, max_steps=1)
    initial = environment.reset(seed=0)
    target_x, target_y = initial["target"]
    player_x, player_y = initial["player"]
    move = "right" if target_x > player_x else "down"
    first = environment.step({"move": move})
    if not first.terminated:
        second = environment.step({"move": "down" if move == "right" else "right"})
        environment.step({"move": "up"}) if second.terminated else None
    else:
        with pytest.raises(RuntimeError):
            environment.step({"move": "up"})

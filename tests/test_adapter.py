"""Contract tests for environment reset and step behavior."""

from arc_agent.env.adapter import StepResult
from arc_agent.env.mock_env import MockEnvironment


def test_reset_is_reproducible_and_step_returns_normalized_result() -> None:
    environment = MockEnvironment()
    first = environment.reset(seed=17)
    second = environment.reset(seed=17)

    assert first == second
    result = environment.step({"move": "right"})
    assert isinstance(result, StepResult)
    assert result.observation["steps"] == 1
    assert result.terminated is False

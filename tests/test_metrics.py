"""Evaluation metric aggregation."""

from arc_agent.agent.agent import Agent
from arc_agent.env.mock_env import MockEnvironment
from arc_agent.eval.metrics import summarize


def test_metrics_summarize_episode_results() -> None:
    results = [
        Agent(MockEnvironment(), seed=seed, max_steps=40).run_episode()
        for seed in range(2)
    ]
    metrics = summarize(results)

    assert metrics.episodes == 2
    assert metrics.successes == 2
    assert metrics.success_rate == 1.0
    assert len(metrics.steps_to_solve) == 2

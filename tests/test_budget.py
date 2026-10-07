"""Resource accounting behavior."""

from arc_agent.utils.budget import Budget


def test_budget_tracks_step_and_compute_limits() -> None:
    budget = Budget(max_steps=1, max_seconds=10, max_compute_units=5)
    assert not budget.exhausted
    budget.consume_step(compute_units=2)
    assert budget.steps_used == 1
    assert budget.compute_units_used == 2
    assert budget.exhausted
    assert budget.remaining_steps() == 0

"""Behavior and information-boundary tests for hidden-rule mock tasks."""

from arc_agent.env.mock_env import HiddenDifficulty, HiddenRuleGridEnv, MockEnvironment, VisibleGridEnv


def test_hidden_rules_and_goal_are_not_returned_in_observations() -> None:
    environment = HiddenRuleGridEnv(difficulty="medium")
    observation = environment.reset(seed=21)

    assert "goal" not in observation
    assert "target" not in observation
    assert "mapping" not in observation
    assert "action_mapping" not in observation
    assert "key_collected" not in observation
    assert "walls_toggled" not in observation
    assert not any(name in str(environment.observation_spec).lower() for name in ("key", "door", "one-way", "button", "mapping"))
    assert all(tile in (0, 1, 2, 4) for row in observation["grid"] for tile in row)
    assert set(environment.action_spec["move"]) == {"up", "down", "left", "right"}


def test_hidden_sampling_is_reproducible_but_not_exposed() -> None:
    first = HiddenRuleGridEnv(difficulty="hard")
    second = HiddenRuleGridEnv(difficulty="hard")
    first_observation = first.reset(seed=13)
    second_observation = second.reset(seed=13)

    assert first_observation == second_observation
    assert first._mapping == second._mapping
    assert "goal" not in first_observation
    assert first.difficulty.partial_observability


def test_action_remapping_changes_effect_without_publishing_mapping() -> None:
    environment = HiddenRuleGridEnv(width=5, height=5, difficulty="medium")
    observation = environment.reset(seed=11)
    environment._player = (2, 2)
    expected = environment._mapping["up"]

    result = environment.step({"move": "up"})

    assert result.observation["player"] == (2 + expected[0], 2 + expected[1])
    assert "mapping" not in result.observation
    assert "mapping" not in environment.observation_spec


def test_each_optional_mechanic_has_hidden_state_and_observable_effect() -> None:
    key_env = HiddenRuleGridEnv(difficulty=HiddenDifficulty(action_remapping=False, mechanic="key_door"))
    key_env.reset(seed=2)
    assert key_env._key is not None
    assert key_env._door is not None
    assert "key_collected" not in key_env._observation()

    door = key_env._door
    key_env._player = (door[0] - 1, door[1])
    before = key_env._player
    key_env.step({"move": "right"})
    assert key_env._player == before
    key_env._key_collected = True
    key_env.step({"move": "right"})
    assert key_env._player == door

    one_way_env = HiddenRuleGridEnv(
        difficulty=HiddenDifficulty(action_remapping=False, mechanic="one_way")
    )
    one_way_env.reset(seed=3)
    one_way = one_way_env._one_way
    assert one_way is not None
    assert "one_way_rule" not in one_way_env._observation()
    assert 6 not in {tile for row in one_way_env._observation()["grid"] for tile in row}
    one_way_env._player = (one_way[0] + 1, one_way[1])
    one_way_env.step({"move": "left"})
    assert one_way_env._player != one_way

    button_env = HiddenRuleGridEnv(
        difficulty=HiddenDifficulty(action_remapping=False, mechanic="button")
    )
    button_env.reset(seed=5)
    button = button_env._button
    assert button is not None
    assert "walls_toggled" not in button_env._observation()
    assert 7 not in {tile for row in button_env._observation()["grid"] for tile in row}
    button_env._player = (button[0] - 1, button[1])
    wall = (button[0] + 1, button[1])
    button_env._walls.add(wall)
    before_toggle = button_env._walls_toggled
    button_env.step({"move": "right"})
    assert button_env._player == button
    assert button_env._walls_toggled is not before_toggle
    button_env.step({"move": "right"})
    assert button_env._player == wall


def test_partial_observation_returns_local_window_and_visible_env_alias() -> None:
    environment = HiddenRuleGridEnv(
        width=7,
        height=7,
        difficulty=HiddenDifficulty(partial_observability=True, view_radius=1),
    )
    observation = environment.reset(seed=0)

    assert len(observation["grid"]) < 7
    assert observation["window_origin"] != (0, 0) or observation["player"] == (0, 0)
    assert MockEnvironment is VisibleGridEnv


def test_key_pickup_is_reported_as_an_event_not_a_map_rule() -> None:
    environment = HiddenRuleGridEnv(
        difficulty=HiddenDifficulty(action_remapping=False, mechanic="key_door")
    )
    environment.reset(seed=2)
    key = environment._key
    assert key is not None
    neighbors = (
        ((key[0] - 1, key[1]), "right"),
        ((key[0] + 1, key[1]), "left"),
        ((key[0], key[1] - 1), "down"),
        ((key[0], key[1] + 1), "up"),
    )
    player, move = next(
        (position, action)
        for position, action in neighbors
        if 0 <= position[0] < environment.width
        and 0 <= position[1] < environment.height
        and position not in environment._walls
        and position != environment._door
    )
    environment._player = player

    result = environment.step({"move": move})

    assert result.observation["player"] == key
    assert result.info["key_collected"] is True
    assert "key" not in str(environment.observation_spec).lower()

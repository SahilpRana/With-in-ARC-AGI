"""Convert environment observations to compact structured agent states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EncodedState:
    """Minimal state representation; richer object extraction is a research hook."""

    raw: Mapping[str, Any]
    grid: tuple[tuple[int, ...], ...] | None
    player: tuple[int, int] | None
    target: tuple[int, int] | None
    grid_origin: tuple[int, int]


def encode_observation(observation: Mapping[str, Any]) -> EncodedState:
    """Normalize known grid fields while retaining the original observation."""
    grid_value = observation.get("grid")
    grid = None
    if grid_value is not None:
        grid = tuple(tuple(int(cell) for cell in row) for row in grid_value)

    def coordinate(name: str) -> tuple[int, int] | None:
        value = observation.get(name)
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two coordinates")
        return int(value[0]), int(value[1])

    # TODO: Add object-centric segmentation and transition-diff features.
    return EncodedState(
        raw=observation,
        grid=grid,
        player=coordinate("player"),
        target=coordinate("target") or coordinate("goal"),
        grid_origin=coordinate("window_origin") or (0, 0),
    )

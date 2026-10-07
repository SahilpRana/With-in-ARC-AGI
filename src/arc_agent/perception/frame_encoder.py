"""Object components and transition differences for ARC grid frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.int8]
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Component:
    """One four-connected same-color region in a frame."""

    color: int
    cells: frozenset[tuple[int, int]]
    bbox: BoundingBox
    centroid: tuple[float, float]

    @property
    def area(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class MovedObject:
    """A same-color, same-shape component matched across two frames."""

    color: int
    before: Component
    after: Component
    displacement: tuple[int, int]


@dataclass(frozen=True)
class FrameDiff:
    """Cell changes and a conservative moved-object match for one transition."""

    changed_cells: frozenset[tuple[int, int]]
    raw_changed_cells: frozenset[tuple[int, int]]
    bbox: BoundingBox | None
    changed_fraction: float
    whole_scene_change: bool
    moved_object: MovedObject | None


@dataclass
class ActionChangeStats:
    """Aggregated frame-change evidence for one action label."""

    count: int = 0
    changed_steps: int = 0
    no_op_steps: int = 0
    total_changed_cells: int = 0

    @property
    def changed_fraction(self) -> float:
        return self.changed_steps / self.count if self.count else 0.0

    @property
    def mean_changed_cells(self) -> float:
        return self.total_changed_cells / self.count if self.count else 0.0


@dataclass
class FrameEncoder:
    """Segment colors, compare consecutive frames, and track action statistics."""

    whole_scene_threshold: float = 0.35
    action_stats: dict[Hashable, ActionChangeStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.whole_scene_threshold <= 1.0:
            raise ValueError("whole_scene_threshold must be in (0, 1]")

    def components(self, frame: NDArray[np.generic]) -> tuple[Component, ...]:
        """Extract four-connected components independently for each color."""
        image = _validated_frame(frame)
        height, width = image.shape
        visited = np.zeros(image.shape, dtype=np.bool_)
        result: list[Component] = []

        for y in range(height):
            for x in range(width):
                if visited[y, x]:
                    continue
                color = int(image[y, x])
                visited[y, x] = True
                queue = deque([(x, y)])
                cells: set[tuple[int, int]] = set()
                while queue:
                    current_x, current_y = queue.popleft()
                    cells.add((current_x, current_y))
                    for next_x, next_y in (
                        (current_x - 1, current_y),
                        (current_x + 1, current_y),
                        (current_x, current_y - 1),
                        (current_x, current_y + 1),
                    ):
                        if (
                            0 <= next_x < width
                            and 0 <= next_y < height
                            and not visited[next_y, next_x]
                            and int(image[next_y, next_x]) == color
                        ):
                            visited[next_y, next_x] = True
                            queue.append((next_x, next_y))
                xs = [cell[0] for cell in cells]
                ys = [cell[1] for cell in cells]
                result.append(Component(
                    color=color,
                    cells=frozenset(cells),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                ))
        return tuple(result)

    def compare(
        self,
        previous: NDArray[np.generic],
        current: NDArray[np.generic],
        action: Hashable | None = None,
        ignored_cells: NDArray[np.bool_] | None = None,
    ) -> FrameDiff:
        """Diff frames, optionally ignoring a caller-learned cell mask."""
        before = _validated_frame(previous)
        after = _validated_frame(current)
        if before.shape != after.shape:
            raise ValueError("frame shapes must match")
        if ignored_cells is not None:
            ignored = np.asarray(ignored_cells)
            if ignored.shape != before.shape or ignored.dtype != np.dtype(np.bool_):
                raise ValueError("ignored_cells must be a 64x64 boolean mask")
        else:
            ignored = np.zeros(before.shape, dtype=np.bool_)

        raw_mask = before != after
        raw_ys, raw_xs = np.nonzero(raw_mask)
        raw_changed_cells = frozenset(
            (int(x), int(y)) for y, x in zip(raw_ys, raw_xs)
        )
        changed_mask = raw_mask & ~ignored
        ys, xs = np.nonzero(changed_mask)
        changed_cells = frozenset((int(x), int(y)) for y, x in zip(ys, xs))
        bbox = (
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if len(xs)
            else None
        )
        changed_fraction = len(changed_cells) / before.size
        whole_scene_change = bool(
            changed_cells and changed_fraction >= self.whole_scene_threshold
        )
        moved = (
            None
            if not changed_cells or whole_scene_change
            else _match_moved_component(before, after, ignored)
        )
        diff = FrameDiff(
            changed_cells=changed_cells,
            raw_changed_cells=raw_changed_cells,
            bbox=bbox,
            changed_fraction=changed_fraction,
            whole_scene_change=whole_scene_change,
            moved_object=moved,
        )
        if action is not None:
            stats = self.action_stats.setdefault(action, ActionChangeStats())
            stats.count += 1
            stats.changed_steps += int(bool(changed_cells))
            stats.no_op_steps += int(not changed_cells)
            stats.total_changed_cells += len(changed_cells)
        return diff


@dataclass
class PersistentCellMask:
    """Learn cells that change on most observed transitions, independent of action."""

    threshold: float = 0.8
    minimum_observations: int = 20
    observation_count: int = 0
    changed_counts: NDArray[np.uint32] = field(
        default_factory=lambda: np.zeros((64, 64), dtype=np.uint32)
    )
    action_counts: dict[Hashable, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")

    def observe(
        self,
        previous: NDArray[np.generic],
        current: NDArray[np.generic],
        action: Hashable | None = None,
    ) -> None:
        """Update frequencies using only two observed consecutive frames."""
        before = _validated_frame(previous)
        after = _validated_frame(current)
        if before.shape != after.shape:
            raise ValueError("frame shapes must match")
        self.changed_counts += (before != after).astype(np.uint32)
        self.observation_count += 1
        if action is not None:
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Return cells changed on at least ``threshold`` of sampled transitions."""
        if self.observation_count < self.minimum_observations:
            return np.zeros((64, 64), dtype=np.bool_)
        return self.changed_counts / self.observation_count >= self.threshold

    @property
    def masked_cell_count(self) -> int:
        return int(self.mask.sum())


def _validated_frame(frame: NDArray[np.generic]) -> Frame:
    image = np.asarray(frame)
    if image.shape != (64, 64):
        raise ValueError(f"expected a 64x64 frame, got {image.shape}")
    if image.dtype != np.dtype(np.int8):
        raise ValueError(f"expected int8 frame values, got {image.dtype}")
    return image


def _match_moved_component(
    before: Frame,
    after: Frame,
    ignored_cells: NDArray[np.bool_] | None = None,
) -> MovedObject | None:
    """Match equal-shape same-color regions removed and added in one transition."""
    pairs: list[tuple[int, float, int, MovedObject]] = []

    for color in np.unique(before):
        color_value = int(color)
        removed_mask = (before == color) & (after != color)
        added_mask = (after == color) & (before != color)
        if ignored_cells is not None:
            removed_mask &= ~ignored_cells
            added_mask &= ~ignored_cells
        removed_regions = _components_in_mask(removed_mask)
        added_regions = _components_in_mask(added_mask)
        for removed in removed_regions:
            for added in added_regions:
                if _relative_shape(removed) != _relative_shape(added):
                    continue
                removed_component = _component_from_cells(color_value, removed)
                added_component = _component_from_cells(color_value, added)
                dx = round(added_component.centroid[0] - removed_component.centroid[0])
                dy = round(added_component.centroid[1] - removed_component.centroid[1])
                if dx == 0 and dy == 0:
                    continue
                distance = dx * dx + dy * dy
                pairs.append((
                    int(color_value == 0),
                    distance,
                    -len(removed),
                    MovedObject(
                        color=color_value,
                        before=removed_component,
                        after=added_component,
                        displacement=(dx, dy),
                    ),
                ))
    return min(
        pairs,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )[3] if pairs else None


def _components_in_mask(mask: NDArray[np.bool_]) -> tuple[frozenset[tuple[int, int]], ...]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    regions: list[frozenset[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(int(x), int(y))])
        cells: set[tuple[int, int]] = set()
        while queue:
            current_x, current_y = queue.popleft()
            cells.add((current_x, current_y))
            for next_x, next_y in (
                (current_x - 1, current_y),
                (current_x + 1, current_y),
                (current_x, current_y - 1),
                (current_x, current_y + 1),
            ):
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        regions.append(frozenset(cells))
    return tuple(regions)


def _relative_shape(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return frozenset((x - min_x, y - min_y) for x, y in cells)


def _component_from_cells(
    color: int,
    cells: frozenset[tuple[int, int]],
) -> Component:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return Component(
        color=color,
        cells=cells,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
    )

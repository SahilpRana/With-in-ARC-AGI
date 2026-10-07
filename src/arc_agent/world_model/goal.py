"""Within-episode goal candidates inferred from reward and state transitions."""

from __future__ import annotations

from dataclasses import dataclass

from arc_agent.memory.episode_memory import EpisodeMemory


@dataclass(frozen=True)
class GoalCandidate:
    """Observed position with accumulated goal evidence and normalized score."""

    position: tuple[int, int]
    evidence: int
    score: float


class GoalInferer:
    """Rank candidate goal positions without consulting hidden environment state."""

    def infer(self, memory: EpisodeMemory) -> tuple[GoalCandidate, ...]:
        """Intersect goal locations consistent with all observed progress signals."""
        positive_positions: list[tuple[int, int]] = []
        negative_positions: list[tuple[int, int]] = []
        terminal_positions: list[tuple[int, int]] = []
        for transition in memory.transitions:
            position = transition.next_state.get("player")
            if position is None:
                continue
            candidate = (int(position[0]), int(position[1]))
            if transition.terminated:
                terminal_positions.append(candidate)
            elif transition.info.get("progress") or transition.reward > 0:
                positive_positions.append(candidate)
            elif "progress" in transition.info:
                negative_positions.append(candidate)

        if terminal_positions:
            goal = terminal_positions[-1]
            return (GoalCandidate(position=goal, evidence=len(terminal_positions), score=1.0),)
        if not positive_positions:
            return ()

        candidates = _nearby_positions(positive_positions[0])
        for position in positive_positions[1:]:
            candidates.intersection_update(_nearby_positions(position))
        for position in negative_positions:
            candidates.difference_update(_nearby_positions(position))
        if not candidates:
            return ()
        confidence = 1.0 / len(candidates)
        return tuple(
            GoalCandidate(
                position=position,
                evidence=sum(position in _nearby_positions(item) for item in positive_positions),
                score=confidence,
            )
            for position in sorted(candidates)
        )


def _nearby_positions(position: tuple[int, int]) -> set[tuple[int, int]]:
    """Return the current cell and its cardinal neighbors."""
    x, y = position
    return {(x, y), (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)}

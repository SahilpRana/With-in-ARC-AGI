"""Verify action-conditioned predictions against episode transitions."""

from __future__ import annotations

from arc_agent.memory.episode_memory import EpisodeMemory
from arc_agent.world_model.hypothesis import Hypothesis


class RuleVerifier:
    """Recount support, contradiction, and no-op evidence for one candidate."""

    def verify(self, hypothesis: Hypothesis, memory: EpisodeMemory) -> Hypothesis:
        """Update candidate evidence; blocked no-ops are neither support nor contradiction."""
        hypothesis.supports = 0
        hypothesis.contradictions = 0
        hypothesis.no_ops = 0
        hypothesis.evidence = []
        for index, transition in enumerate(memory.transitions, start=1):
            if transition.action.get("move") != hypothesis.action:
                continue
            before = transition.state.get("player")
            after = transition.next_state.get("player")
            if before is None or after is None:
                continue
            displacement = (int(after[0]) - int(before[0]), int(after[1]) - int(before[1]))
            if displacement == (0, 0):
                hypothesis.no_ops += 1
                hypothesis.evidence.append(f"transition {index}: no-op")
            elif displacement == hypothesis.expected_displacement:
                hypothesis.supports += 1
                hypothesis.evidence.append(f"transition {index}: support {displacement}")
            else:
                hypothesis.contradictions += 1
                hypothesis.evidence.append(f"transition {index}: contradiction {displacement}")
        return hypothesis

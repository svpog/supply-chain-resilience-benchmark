"""
Random-agent для multi-agent recovery M1.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from src.recovery.agents.base_agent import AgentBase


class RandomAgent(AgentBase):
    """Обирає випадковий видимий failed-вузол."""

    def choose_action(self, observation: dict, locked: set | None = None):
        locked = locked or set()
        candidates = sorted(set(observation["visible_failed"]) - locked)
        if not candidates:
            return None
        idx = int(self.rng.integers(0, len(candidates)))
        return candidates[idx]

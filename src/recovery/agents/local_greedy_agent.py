"""
Local greedy-agent для multi-agent recovery M1.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from src.recovery.agents.base_agent import AgentBase
from src.recovery.strategies.flow_greedy_local import FlowGreedyLocalStrategy


class LocalGreedyAgent(AgentBase):
    """Обирає failed-вузол з максимальним локальним Δmax_flow."""

    def __init__(self, agent_id: int, r: int | None, network, rng=None) -> None:
        super().__init__(agent_id=agent_id, r=r, network=network, rng=rng)
        self._strategy = FlowGreedyLocalStrategy(r=r)

    def choose_action(self, observation: dict, locked: set | None = None):
        locked = locked or set()
        candidates = set(observation["visible_failed"]) - locked
        if not candidates:
            return None
        return self._strategy.select_single(
            network=observation["network"],
            candidates=candidates,
            rng=self.rng,
        )

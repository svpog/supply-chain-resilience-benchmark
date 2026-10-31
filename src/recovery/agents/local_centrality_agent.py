"""
Local centrality-agent для multi-agent recovery M1.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

import networkx as nx

from src.recovery.agents.base_agent import AgentBase


class LocalCentralityAgent(AgentBase):
    """Обирає видимий failed-вузол з найбільшою betweenness centrality."""

    def __init__(self, agent_id: int, r: int | None, network, rng=None) -> None:
        super().__init__(agent_id=agent_id, r=r, network=network, rng=rng)
        self._centrality = {
            node: float(value)
            for node, value in nx.betweenness_centrality(self.G_base).items()
        }

    def choose_action(self, observation: dict, locked: set | None = None):
        locked = locked or set()
        candidates = sorted(set(observation["visible_failed"]) - locked)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda node: (-self._centrality.get(node, 0.0), node),
        )[0]

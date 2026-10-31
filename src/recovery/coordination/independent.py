"""
Independent coordinator для M1.

Реалізує: Протокол M1, §4.3
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from src.recovery.coordination.base_coordinator import CoordinatorBase


class IndependentCoordinator(CoordinatorBase):
    """Агенти діють незалежно; дублювання repair-цілей дозволене."""

    def allocate(self, agents: list, network, failed_nodes: set) -> dict[int, object | None]:
        actions: dict[int, object | None] = {}
        for agent in agents:
            obs = agent.observe(network, failed_nodes)
            actions[agent.agent_id] = agent.choose_action(obs, locked=None)
        return actions

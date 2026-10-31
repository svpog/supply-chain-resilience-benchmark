"""
Token coordinator для M1.

Реалізує: Протокол M1, §4.3
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from src.recovery.coordination.base_coordinator import CoordinatorBase


class TokenCoordinator(CoordinatorBase):
    """Mutex: уже обраний вузол блокується для наступних агентів."""

    def allocate(self, agents: list, network, failed_nodes: set) -> dict[int, object | None]:
        actions: dict[int, object | None] = {}
        locked: set = set()
        for agent in sorted(agents, key=lambda a: a.agent_id):
            obs = agent.observe(network, failed_nodes)
            action = agent.choose_action(obs, locked=locked)
            actions[agent.agent_id] = action
            if action is not None:
                locked.add(action)
        return actions

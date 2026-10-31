"""
Базовий coordinator для multi-agent recovery.

Реалізує: Протокол M1, §4.3
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CoordinatorBase(ABC):
    """Інтерфейс розподілу repair-дій між агентами."""

    @abstractmethod
    def allocate(self, agents: list, network, failed_nodes: set) -> dict[int, object | None]:
        """Повертає {agent_id: node_to_repair | None} для timestep."""

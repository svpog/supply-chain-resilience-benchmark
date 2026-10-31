"""
Абстрактний клас RecoveryStrategy — hook для 2-ї статті.

Реалізує: розділ 4 Math Spec; архітектурний hook для Стаття 2
(див. docs/Math_Spec_Стаття_1.docx та docs/Скелети_статей_2_3_4.docx).

У 1-й статті використовується лише RandomRecoveryStrategy.
Цей абстрактний клас дозволить додати у 2-й статті:
- CentralityPriorityStrategy
- FlowGreedyStrategy
- MASStrategy (мультиагентна)

Формат елемента: ("node", node_id) або ("edge", (u, v)).

Приклад використання:
    >>> from src.recovery.random_strategy import RandomRecoveryStrategy
    >>> strategy = RandomRecoveryStrategy()
    >>> element = strategy.select_element_to_repair(network, t=5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.network.base import Network


class RecoveryStrategy(ABC):
    """
    Абстрактна стратегія відновлення мережі.

    Підкласи реалізують select_element_to_repair(), яка повертає
    один пошкоджений елемент для відновлення на кроці t.
    Repair виконується методом repair() з відновленням до базових значень.
    """

    @abstractmethod
    def select_element_to_repair(
        self,
        network: "Network",
        t: int,
        rng: np.random.Generator | None = None,
    ) -> tuple | None:
        """
        Обирає елемент для відновлення на кроці t.

        Args:
            network: поточний стан мережі.
            t: поточний крок симуляції.
            rng: генератор випадкових чисел (для відтворюваності).

        Returns:
            ("node", node_id) або ("edge", (u, v)) для відновлення,
            або None якщо нема чого відновлювати.
        """

    def repair(self, network: "Network", element: tuple) -> None:
        """
        Відновлює елемент до базового стану (Math Spec 4.2).

        Args:
            network: мережа (модифікується in-place).
            element: ("node", node_id) або ("edge", (u, v)).
        """
        kind, target = element
        if kind == "node":
            network.set_node_state(target, "active")
        elif kind == "edge":
            u, v = target
            network.restore_edge(u, v)
        else:
            raise ValueError(f"Невідомий тип елемента: '{kind}'. Має бути 'node' або 'edge'.")

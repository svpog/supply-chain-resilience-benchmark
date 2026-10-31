"""
Стратегія відновлення: random (єдина у 1-й статті).

Реалізує: розділ 4.2 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

На кожному кроці t uniformly random обирається один пошкоджений елемент
(вузол або ребро) і повністю відновлюється до базових значень. µ = 1 (фіксовано).

Пріоритет відновлення: спочатку вузли, потім ребра. Це забезпечує
відновлення потоку якомога раніше (Math Spec 4.2).

Приклад використання:
    >>> from src.recovery.random_strategy import RandomRecoveryStrategy
    >>> strategy = RandomRecoveryStrategy()
    >>> element = strategy.select_element_to_repair(network, t=5)
    >>> if element:
    ...     strategy.repair(network, element)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from src.network.base import Network


class RandomRecoveryStrategy(RecoveryStrategy):
    """
    Випадкова стратегія відновлення: uniformly random серед пошкоджених елементів.

    Math Spec 4.2:
        На кожному кроці t: обираємо один елемент зі списку failed
        рівноймовірно і повністю відновлюємо його.
        µ = 1 елемент/крок (базова швидкість відновлення).
    """

    def select_element_to_repair(
        self,
        network: "Network",
        t: int,
        rng: np.random.Generator | None = None,
    ) -> tuple | None:
        """
        Рівноймовірно обирає один пошкоджений елемент для відновлення.

        Якщо є failed вузли — обирає серед них.
        Якщо вузлів нема — обирає серед деградованих ребер.
        Незворотно пошкоджені елементи виключаються (через get_failed_elements).

        Args:
            network: поточний стан мережі.
            t: поточний крок симуляції (не використовується у random стратегії,
               але є у сигнатурі для сумісності зі складнішими стратегіями).
            rng: генератор випадкових чисел. Якщо None — створюється новий.

        Returns:
            ("node", node_id), ("edge", (u, v)) або None якщо нема чого відновлювати.
        """
        if rng is None:
            rng = np.random.default_rng()

        failed = network.get_failed_elements()

        # Пріоритет вузлам: відновлення вузла розблоковує всі суміжні ребра
        if failed["nodes"]:
            node = int(rng.choice(sorted(failed["nodes"])))
            return ("node", node)

        if failed["edges"]:
            edges = sorted(failed["edges"])
            idx = int(rng.integers(len(edges)))
            return ("edge", edges[idx])

        return None

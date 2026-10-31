"""
Абстрактний базовий клас для сценаріїв стохастичних збоїв.

Реалізує: розділ 2 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Усі конкретні сценарії збоїв успадковуються від Disruption і реалізують
метод apply(). Метод повертає НОВУ копію мережі — оригінал не змінюється.

Приклад використання:
    >>> from src.disruptions.node_failure import NodeFailureDisruption
    >>> d = NodeFailureDisruption(p_v=0.20)
    >>> disrupted_net = d.apply(network, seed=42)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.network.base import Network


class Disruption(ABC):
    """
    Абстрактний клас сценарію збою.

    Кожен конкретний збій реалізує apply(), який приймає мережу і seed,
    та повертає нову мережу з накладеним збоєм.
    """

    def __init__(self, w: float = 0.0) -> None:
        """
        Args:
            w: частка незворотно пошкоджених елементів ∈ [0, 1] (Math Spec S6).
               w=0 означає що всі збої потенційно зворотні (типовий випадок).
        """
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"w={w} має бути у [0, 1].")
        self.w = w

    @abstractmethod
    def apply(self, network: "Network", seed: int | None = None) -> "Network":
        """
        Застосовує збій до мережі і повертає пошкоджену копію.

        Args:
            network: початкова мережа (не змінюється).
            seed: random seed для відтворюваності.

        Returns:
            Нова Network зі збоєм. Оригінал залишається незмінним.
        """

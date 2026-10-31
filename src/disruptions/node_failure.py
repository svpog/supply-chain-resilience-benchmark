"""
Сценарій 1: випадкові відмови вузлів (Bernoulli).

Реалізує: розділ 2.1 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Кожен вузол незалежно відмовляє з імовірністю p_v (Bernoulli).
При відмові вузол переходить у стан 'failed': пропозиція/попит = 0,
capacity всіх суміжних ребер ефективно = 0.

При w > 0: частка w від failed вузлів позначається як irreversible
(Math Spec S6).

Приклад використання:
    >>> from src.disruptions.node_failure import NodeFailureDisruption
    >>> d = NodeFailureDisruption(p_v=0.20)
    >>> disrupted = d.apply(network, seed=42)
"""

from __future__ import annotations

import numpy as np

from src.disruptions.base import Disruption
from src.network.base import Network


class NodeFailureDisruption(Disruption):
    """
    Bernoulli-збій вузлів: кожен вузол відмовляє незалежно з імовірністю p_v.

    Math Spec 2.1:
        X_i ~ Bernoulli(p_v), i = 1..N
        Якщо X_i = 1: вузол i переходить у 'failed'
    """

    def __init__(self, p_v: float, w: float = 0.0) -> None:
        """
        Args:
            p_v: імовірність відмови окремого вузла ∈ [0, 1].
            w: частка незворотних відмов (Math Spec S6). За замовчуванням 0.
        """
        super().__init__(w=w)
        if not (0.0 <= p_v <= 1.0):
            raise ValueError(f"p_v={p_v} має бути у [0, 1].")
        self.p_v = p_v

    def apply(self, network: Network, seed: int | None = None) -> Network:
        """
        Застосовує Bernoulli-збій вузлів і повертає нову пошкоджену мережу.

        Args:
            network: початкова мережа (не змінюється).
            seed: random seed для відтворюваності.

        Returns:
            Нова Network з пошкодженими вузлами.
        """
        rng = np.random.default_rng(seed)
        net = network.copy()

        nodes = list(net.graph.nodes())
        # Кожен вузол незалежно відмовляє з імовірністю p_v
        failures = rng.random(len(nodes)) < self.p_v
        failed_list = [n for n, f in zip(nodes, failures) if f]

        for node in failed_list:
            net.set_node_state(node, "failed")

        # Позначаємо частку w як незворотно пошкоджених (Math Spec S6)
        if self.w > 0.0 and failed_list:
            n_irreversible = max(1, round(self.w * len(failed_list)))
            rng.shuffle(failed_list)
            for node in failed_list[:n_irreversible]:
                net.irreversible_nodes.add(node)

        return net

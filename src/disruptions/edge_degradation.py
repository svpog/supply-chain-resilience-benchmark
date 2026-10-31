"""
Сценарій 2: деградація ребер (Bernoulli + gamma).

Реалізує: розділ 2.2 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Кожне ребро незалежно деградує з імовірністю p_e (Bernoulli).
При деградації capacity знижується у (1 − γ) разів:
    c_ij(t) = (1 − γ) · c_ij(0)

При w > 0: частка w від degraded ребер позначається як irreversible
(Math Spec S6).

Приклад використання:
    >>> from src.disruptions.edge_degradation import EdgeDegradationDisruption
    >>> d = EdgeDegradationDisruption(p_e=0.20, gamma=0.40)
    >>> disrupted = d.apply(network, seed=42)
"""

from __future__ import annotations

import numpy as np

from src.disruptions.base import Disruption
from src.network.base import Network


class EdgeDegradationDisruption(Disruption):
    """
    Bernoulli-деградація ребер: кожне ребро деградує з імовірністю p_e.

    Math Spec 2.2:
        Y_ij ~ Bernoulli(p_e)
        Якщо Y_ij = 1: c_ij(t) = (1 − γ) · c_ij(0)
    """

    VALID_TARGETS = {"all", "trunk_only", "backup_only"}

    def __init__(
        self,
        p_e: float,
        gamma: float,
        w: float = 0.0,
        target: str = "all",
    ) -> None:
        """
        Args:
            p_e: імовірність деградації ребра ∈ [0, 1].
            gamma: коефіцієнт деградації ∈ [0, 1]. gamma=1 → повна відмова.
            w: частка незворотних деградацій (Math Spec S6). За замовчуванням 0.
            target: пул ребер для деградації:
                    'all' — усі ребра (legacy behavior);
                    'trunk_only' — лише магістральні ребра;
                    'backup_only' — лише резервні ребра.
        """
        super().__init__(w=w)
        if not (0.0 <= p_e <= 1.0):
            raise ValueError(f"p_e={p_e} має бути у [0, 1].")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma={gamma} має бути у [0, 1].")
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"target='{target}' має бути одним із {sorted(self.VALID_TARGETS)}."
            )
        self.p_e = p_e
        self.gamma = gamma
        self.target = target

    def _is_backup_edge(self, network: Network, edge: tuple) -> bool:
        """Визначає, чи є ребро резервним за edge_type='backup'."""
        u, v = edge
        data = network.graph.edges[u, v] if network.graph.has_edge(u, v) else {}
        return data.get("edge_type") == "backup"

    def _candidate_edges(self, network: Network) -> list[tuple]:
        """
        Повертає ребра, на які застосовується Bernoulli-деградація.

        target='all' лишає стару поведінку побайтово сумісною: список ребер
        дорівнює list(network.graph.edges()) у тому самому порядку.
        """
        edges = list(network.graph.edges())
        if self.target == "all":
            return edges
        if self.target == "trunk_only":
            return [edge for edge in edges if not self._is_backup_edge(network, edge)]
        return [edge for edge in edges if self._is_backup_edge(network, edge)]

    def apply(self, network: Network, seed: int | None = None) -> Network:
        """
        Застосовує деградацію ребер і повертає нову пошкоджену мережу.

        Args:
            network: початкова мережа (не змінюється).
            seed: random seed для відтворюваності.

        Returns:
            Нова Network зі зниженими capacity деградованих ребер.
        """
        rng = np.random.default_rng(seed)
        net = network.copy()

        edges = self._candidate_edges(net)
        failures = rng.random(len(edges)) < self.p_e
        degraded_list = [e for e, f in zip(edges, failures) if f]

        for u, v in degraded_list:
            base_cap = net._base_capacities.get((u, v), 0.0)
            new_cap = (1.0 - self.gamma) * base_cap
            net.set_edge_capacity(u, v, new_cap)

        # Позначаємо частку w як незворотно деградованих (Math Spec S6)
        if self.w > 0.0 and degraded_list:
            n_irreversible = max(1, round(self.w * len(degraded_list)))
            rng.shuffle(degraded_list)
            for edge in degraded_list[:n_irreversible]:
                net.irreversible_edges.add(edge)

        metadata = dict(getattr(net, "disruption_metadata", {}) or {})
        metadata.update({
            "edge_degradation_target": self.target,
            "degraded_edges": list(degraded_list),
        })
        net.disruption_metadata = metadata
        return net

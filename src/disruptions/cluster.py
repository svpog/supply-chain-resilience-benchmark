"""
Сценарій 3: кластерне локальне пошкодження (BFS-куля).

Реалізує: розділ 2.3 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Підтримує два режими вибору розміру кластера:
  - r_clust (int): фіксований BFS-радіус (оригінальна Math Spec).
  - target_fraction (float): цільова частка вузлів N, яку треба охопити.
    Алгоритм розширює BFS рівень за рівнем до досягнення target_fraction × N
    вузлів; останній рівень може бути взятий частково (випадкова підмножина).
    Це дає топологічно-нейтральну «дозу» збою незалежно від щільності графу.

Один з параметрів r_clust або target_fraction має бути вказаний.
target_fraction має пріоритет, якщо вказані обидва.

При w > 0: частка w від failed вузлів позначається як irreversible
(Math Spec S6).

Приклад використання (r_clust):
    >>> from src.disruptions.cluster import ClusterDisruption
    >>> d = ClusterDisruption(r_clust=2)
    >>> disrupted = d.apply(network, seed=42)

Приклад використання (target_fraction):
    >>> d = ClusterDisruption(target_fraction=0.20)
    >>> disrupted = d.apply(network, seed=42)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.disruptions.base import Disruption
from src.network.base import Network


class ClusterDisruption(Disruption):
    """
    Кластерний збій: BFS-куля навколо випадкового епіцентру.

    Режими:
        r_clust        — фіксований радіус (Math Spec 2.3).
        target_fraction — цільова частка вузлів (топологічно-нейтральний).
    """

    def __init__(
        self,
        r_clust: int | None = None,
        target_fraction: float | None = None,
        w: float = 0.0,
    ) -> None:
        """
        Args:
            r_clust: радіус BFS-кулі (≥ 0). Ігнорується, якщо задано target_fraction.
            target_fraction: цільова частка вузлів у (0, 1]. Має пріоритет над r_clust.
            w: частка незворотних відмов (Math Spec S6). За замовчуванням 0.

        Raises:
            ValueError: якщо ні r_clust, ні target_fraction не задані,
                або якщо параметри виходять за допустимі межі.
        """
        super().__init__(w=w)
        if r_clust is None and target_fraction is None:
            raise ValueError("Потрібно вказати r_clust або target_fraction.")
        if r_clust is not None and r_clust < 0:
            raise ValueError(f"r_clust={r_clust} має бути ≥ 0.")
        if target_fraction is not None and not (0 < target_fraction <= 1.0):
            raise ValueError(f"target_fraction={target_fraction} має бути у (0, 1].")
        self.r_clust = r_clust
        self.target_fraction = target_fraction

    def apply(self, network: Network, seed: int | None = None) -> Network:
        """
        Застосовує кластерний збій і повертає нову пошкоджену мережу.

        Режим r_clust:
            Failed = { v ∈ V : dist_undirected(v*, v) ≤ r_clust }  (Math Spec 2.3)

        Режим target_fraction:
            BFS розширюється від епіцентру рівень за рівнем до тих пір, поки
            |Failed| ≥ floor(target_fraction × N). Останній рівень може бути
            взятий частково (випадкова підмножина без повторень).

        Args:
            network: початкова мережа (не змінюється).
            seed: random seed для відтворюваності.

        Returns:
            Нова Network з пошкодженим кластером. Епіцентр зберігається
            у disruption_metadata["epicenter"].
        """
        rng = np.random.default_rng(seed)
        net = network.copy()
        undirected = net.graph.to_undirected()

        nodes = list(net.graph.nodes())
        epicenter = int(rng.choice(nodes))
        net.disruption_metadata["epicenter"] = epicenter

        if self.target_fraction is not None:
            failed_list = self._bfs_target(undirected, epicenter,
                                           self.target_fraction, net.N, rng)
        else:
            ball = nx.single_source_shortest_path_length(
                undirected, epicenter, cutoff=self.r_clust
            )
            failed_list = list(ball.keys())

        for node in failed_list:
            net.set_node_state(node, "failed")

        if self.w > 0.0 and failed_list:
            n_irreversible = max(1, round(self.w * len(failed_list)))
            shuffled = list(failed_list)
            rng.shuffle(shuffled)
            for node in shuffled[:n_irreversible]:
                net.irreversible_nodes.add(node)

        return net

    @staticmethod
    def _bfs_target(
        undirected: nx.Graph,
        epicenter: int,
        target_fraction: float,
        N: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """
        BFS-розширення до досягнення target_fraction × N вузлів.

        Повні рівні додаються цілком; останній рівень — випадкова підмножина
        без повторень до потрібного залишку.
        """
        target_n = max(1, int(target_fraction * N))

        visited: set[int] = {epicenter}
        frontier: set[int] = {epicenter}

        while len(visited) < target_n:
            next_level: set[int] = set()
            for node in frontier:
                for neighbor in undirected.neighbors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)

            if not next_level:
                break  # Граф вичерпаний — менше вузлів ніж target_n

            remaining = target_n - len(visited)
            if len(next_level) <= remaining:
                # Беремо весь рівень
                visited.update(next_level)
                frontier = next_level
            else:
                # Беремо випадкову підмножину рівня
                chosen = rng.choice(
                    sorted(next_level), size=remaining, replace=False
                )
                visited.update(chosen.tolist())
                break

        return list(visited)

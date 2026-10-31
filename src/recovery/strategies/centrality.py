"""
Centrality-based recovery strategies для статті 3.

Реалізує: розділ 3.4 Math Spec Article 3
(див. docs/Math_Spec_Article_3.docx).

Стратегія обирає пошкоджений вузол або ребро з найбільшою статичною
центральністю у непошкодженій топології. Центральність кешується один раз.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from src.network.base import Network


class CentralityStrategy(RecoveryStrategy):
    """Recovery за пріоритетом degree / betweenness / pagerank."""

    def __init__(self, centrality_type: str = "betweenness") -> None:
        self.centrality_type = centrality_type
        self._cached_centrality: dict | None = None
        self._cached_graph_signature: tuple[int, int] | None = None

    def _graph_signature(self, network: "Network") -> tuple[int, int]:
        return (network.graph.number_of_nodes(), network.graph.number_of_edges())

    def _compute_centrality(self, network: "Network") -> dict:
        """
        Обчислює центральність один раз для статичної топології.

        Degree — total directed degree (in + out), як у протоколі R1.
        Betweenness і PageRank рахуються через NetworkX на поточному graph.
        """
        signature = self._graph_signature(network)
        if self._cached_centrality is not None and self._cached_graph_signature == signature:
            return self._cached_centrality

        G = network.graph
        if self.centrality_type == "degree":
            centrality = {
                node: float(G.in_degree(node) + G.out_degree(node))
                for node in G.nodes()
            }
        elif self.centrality_type == "betweenness":
            centrality = {node: float(value) for node, value in nx.betweenness_centrality(G).items()}
        elif self.centrality_type == "pagerank":
            centrality = {node: float(value) for node, value in nx.pagerank(G).items()}
        else:
            raise ValueError(f"Unknown centrality type: {self.centrality_type}")

        self._cached_centrality = centrality
        self._cached_graph_signature = signature
        return centrality

    def _score_element(self, element: tuple, centrality: dict) -> float:
        kind, target = element
        if kind == "node":
            return float(centrality.get(target, 0.0))
        if kind == "edge":
            u, v = target
            return max(float(centrality.get(u, 0.0)), float(centrality.get(v, 0.0)))
        return 0.0

    def _rank_with_seeded_ties(
        self,
        elements: list[tuple],
        centrality: dict,
        rng: np.random.Generator,
    ) -> list[tuple]:
        groups: dict[float, list[tuple]] = defaultdict(list)
        for element in elements:
            groups[self._score_element(element, centrality)].append(element)

        ranked: list[tuple] = []
        for score in sorted(groups.keys(), reverse=True):
            group = sorted(groups[score], key=repr)
            if len(group) > 1:
                order = rng.permutation(len(group))
                group = [group[int(i)] for i in order]
            ranked.extend(group)
        return ranked

    def select_element_to_repair(
        self,
        network: "Network",
        t: int,
        rng: np.random.Generator | None = None,
    ) -> tuple | None:
        """Обирає failed element з найбільшою центральністю."""
        if rng is None:
            rng = np.random.default_rng()

        failed = network.get_failed_elements()
        elements = [("node", node) for node in sorted(failed["nodes"])]
        elements.extend(("edge", edge) for edge in sorted(failed["edges"]))

        if not elements:
            return None
        if len(elements) == 1:
            return elements[0]

        centrality = self._compute_centrality(network)
        return self._rank_with_seeded_ties(elements, centrality, rng)[0]

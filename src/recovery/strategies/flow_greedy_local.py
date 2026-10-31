"""
Single-agent flow_greedy з обмеженим радіусом видимості.

Реалізує: Math Spec Article 4 v4.1, §3.2 та Протокол M0, §4.2
(див. docs/Math_Spec_Article_4_v4_1.docx та docs/Протокол_M0.docx).

Приклад використання:
    >>> from src.recovery.strategies.flow_greedy_local import FlowGreedyLocalStrategy
    >>> strategy = FlowGreedyLocalStrategy(r=2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src.network.base import Network
from src.network.flow import compute_max_flow
from src.recovery.base import RecoveryStrategy
from src.recovery.strategies.flow_greedy import FlowGreedyStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable


class FlowGreedyLocalStrategy(RecoveryStrategy):
    """
    Жадібна recovery-стратегія з локальною видимістю r.

    r=None означає повну видимість і делегує вибір існуючій
    FlowGreedyStrategy. r=0 означає, що локальний підграф для вузла-кандидата
    містить лише сам вузол; у типовому node_failure сценарії всі локальні
    прирости тоді нульові, і вибір зводиться до детермінованого tie-break.
    """

    def __init__(self, r: int | None = None) -> None:
        if r is not None and r < 0:
            raise ValueError("r має бути невід'ємним цілим або None для повної видимості.")
        self.r = r
        self._full_greedy = FlowGreedyStrategy()

    def _element_center_nodes(self, element: tuple) -> list:
        """Повертає вузли-центри для локальної околиці елемента."""
        kind, target = element
        if kind == "node":
            return [target]
        if kind == "edge":
            u, v = target
            return [u, v]
        raise ValueError(f"Unknown failed element kind: {kind}")

    def _get_restricted_subgraph(self, G: nx.DiGraph, node) -> nx.DiGraph:
        """
        Повертає r-hop підграф навколо node у неорієнтованій метриці.

        Цей метод відкритий для unit-тестів M0 і повторного використання
        майбутніми agent-стратегіями статті 4.
        """
        if self.r is None:
            return G.copy()
        G_undir = G.to_undirected()
        neighborhood = nx.single_source_shortest_path_length(
            G_undir, node, cutoff=self.r
        ).keys()
        return G.subgraph(neighborhood).copy()

    def _restricted_nodes(self, network: "Network", element: tuple) -> set:
        """Обчислює множину вузлів локального підграфа для element."""
        if self.r is None:
            return set(network.graph.nodes())

        nodes: set = set()
        undirected = network.graph.to_undirected()
        for center in self._element_center_nodes(element):
            if center not in network.graph:
                continue
            reachable = nx.single_source_shortest_path_length(
                undirected, center, cutoff=self.r
            )
            nodes.update(reachable.keys())
        return nodes

    def _make_restricted_network(self, network: "Network", element: tuple) -> "Network":
        """
        Створює Network-обгортку для локального підграфа зі збереженням стану.

        Важливо: базові supply/demand беруться з оригінальної мережі, а поточні
        node_supply/node_demand/edge_capacities/failed_nodes копіюють стан t.
        """
        nodes = self._restricted_nodes(network, element)
        G_sub = network.graph.subgraph(nodes).copy()
        base_capacities = {
            (u, v): network._base_capacities[(u, v)]
            for u, v in G_sub.edges()
            if (u, v) in network._base_capacities
        }
        base_supply = {node: network._base_supply.get(node, 0.0) for node in G_sub.nodes()}
        base_demand = {node: network._base_demand.get(node, 0.0) for node in G_sub.nodes()}

        local = Network(
            graph=G_sub,
            suppliers=frozenset(s for s in network.suppliers if s in nodes),
            consumers=frozenset(c for c in network.consumers if c in nodes),
            base_capacities=base_capacities,
            node_supply=base_supply,
            node_demand=base_demand,
        )
        local.edge_capacities = {
            edge: network.edge_capacities.get(edge, 0.0)
            for edge in base_capacities
        }
        local.node_supply = {
            node: network.node_supply.get(node, 0.0)
            for node in G_sub.nodes()
        }
        local.node_demand = {
            node: network.node_demand.get(node, 0.0)
            for node in G_sub.nodes()
        }
        local.failed_nodes = set(network.failed_nodes).intersection(nodes)
        local.irreversible_nodes = set(network.irreversible_nodes).intersection(nodes)
        local.irreversible_edges = set(network.irreversible_edges).intersection(base_capacities)
        local.disruption_metadata = dict(network.disruption_metadata)
        return local

    def _restore_on_copy(self, network: "Network", element: tuple) -> "Network":
        """Відновлює element на копії локальної мережі."""
        restored = network.copy()
        kind, target = element
        if kind == "node":
            if target in restored.graph:
                restored.set_node_state(target, "active")
        elif kind == "edge":
            u, v = target
            if (u, v) in restored._base_capacities:
                restored.restore_edge(u, v)
        else:
            raise ValueError(f"Unknown failed element kind: {kind}")
        return restored

    def _compute_local_flow_gain(self, network: "Network", element: tuple) -> float:
        """ΔF у r-hop локальному підграфі після тимчасового відновлення element."""
        local_before = self._make_restricted_network(network, element)
        F_before = compute_max_flow(local_before)
        local_after = self._restore_on_copy(local_before, element)
        F_after = compute_max_flow(local_after)
        return float(F_after - F_before)

    def _candidate_elements(self, network: "Network") -> list[tuple]:
        """Повертає repairable elements у стабільному порядку."""
        failed = network.get_failed_elements()
        elements = [("node", node) for node in sorted(failed["nodes"])]
        elements.extend(("edge", edge) for edge in sorted(failed["edges"]))
        return elements

    def _element_sort_key(self, element: tuple) -> tuple:
        """Детермінований tie-break: node перед edge, далі id/repr."""
        kind, target = element
        kind_rank = 0 if kind == "node" else 1
        if kind == "node" and isinstance(target, (int, float, str)):
            return (kind_rank, target)
        return (kind_rank, repr(target))

    def select_element_to_repair(
        self,
        network: "Network",
        t: int,
        rng: np.random.Generator | None = None,
    ) -> tuple | None:
        """
        Обирає repairable element з максимальним локальним Δmax_flow.

        Args:
            network: поточний стан мережі.
            t: часовий крок recovery.
            rng: приймається для сумісності з RecoveryStrategy; finite-r
                 tie-break детермінований і rng не використовує.
        """
        if self.r is None:
            return self._full_greedy.select_element_to_repair(network, t=t, rng=rng)

        elements = self._candidate_elements(network)
        if not elements:
            return None
        if len(elements) == 1:
            return elements[0]

        gains = {
            element: self._compute_local_flow_gain(network, element)
            for element in elements
        }
        return sorted(
            elements,
            key=lambda element: (-gains[element], self._element_sort_key(element)),
        )[0]

    def select_single(
        self,
        network: "Network",
        candidates: "Iterable",
        t: int = 0,
        rng: np.random.Generator | None = None,
    ) -> object | None:
        """
        Атомарний вибір одного failed-вузла для multi-agent runner M1.

        На відміну від select_element_to_repair(), цей метод отримує вже
        відфільтровану множину вузлів-кандидатів (наприклад, після mutex
        coordinator-а) і повертає тільки node_id. Edge-repair у M1 не
        використовується, бо базовий disruption — node_failure.
        """
        candidate_nodes = sorted(candidates)
        if not candidate_nodes:
            return None
        if len(candidate_nodes) == 1:
            return candidate_nodes[0]

        if self.r is None:
            full = FlowGreedyStrategy()
            elements = [("node", node) for node in candidate_nodes]
            current_flow = compute_max_flow(network)
            gains = {
                element: compute_max_flow(full._restore_on_copy(network, element)) - current_flow
                for element in elements
            }
            selected = full._rank_with_seeded_ties(gains, rng or np.random.default_rng())[0]
            return selected[1]

        elements = [("node", node) for node in candidate_nodes]
        gains = {
            element: self._compute_local_flow_gain(network, element)
            for element in elements
        }
        selected = sorted(
            elements,
            key=lambda element: (-gains[element], self._element_sort_key(element)),
        )[0]
        return selected[1]

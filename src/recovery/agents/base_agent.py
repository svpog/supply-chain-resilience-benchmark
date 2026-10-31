"""
Базовий клас агента для multi-agent recovery.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src.network.flow import compute_max_flow

if TYPE_CHECKING:
    from src.network.base import Network


class AgentBase(ABC):
    """
    Базовий recovery-agent без явної позиції.

    Для M1 агент бачить r-hop околиці навколо пошкоджених вузлів. Це
    спрощена модель видимості з Протоколу M1: позиційні агенти відкладаються
    на майбутній patch, якщо M1 покаже потребу.
    """

    def __init__(
        self,
        agent_id: int,
        r: int | None,
        network: "Network",
        rng: np.random.Generator | None = None,
    ) -> None:
        if r is not None and r < 0:
            raise ValueError("r має бути невід'ємним або None.")
        self.agent_id = int(agent_id)
        self.r = r
        self.G_base = network.graph.copy()
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_neighborhood(self, G_current: nx.DiGraph, failed_nodes: set) -> set:
        """Union r-hop околиць навколо всіх failed-вузлів."""
        if self.r is None:
            return set(G_current.nodes())
        if not failed_nodes:
            return set()
        undirected = G_current.to_undirected()
        visible: set = set()
        for node in failed_nodes:
            if node not in undirected:
                continue
            reachable = nx.single_source_shortest_path_length(
                undirected, node, cutoff=self.r
            )
            visible.update(reachable.keys())
        return visible

    def observe(self, network: "Network", failed_nodes: set) -> dict:
        """
        Повертає O_k(t): видимі вузли, видимі failed-вузли і підграф.
        """
        visible = self._get_neighborhood(network.graph, set(failed_nodes))
        visible_failed = set(failed_nodes).intersection(visible)
        return {
            "network": network,
            "visible_nodes": visible,
            "visible_failed": visible_failed,
            "G_sub": network.graph.subgraph(visible).copy(),
        }

    def _make_observed_network(self, network: "Network", visible_nodes: set) -> "Network":
        """
        Створює Network-обгортку для видимої агентом частини графа.

        Auction/broadcast у M3 оцінюють bid через локальний приріст max-flow.
        Щоб оцінка була чесно обмежена спостереженням агента, max-flow
        рахується не на повній мережі, а на підграфі visible_nodes.
        """
        from src.network.base import Network

        nodes = set(visible_nodes)
        G_sub = network.graph.subgraph(nodes).copy()
        base_capacities = {
            (u, v): network._base_capacities[(u, v)]
            for u, v in G_sub.edges()
            if (u, v) in network._base_capacities
        }
        base_supply = {node: network._base_supply.get(node, 0.0) for node in G_sub.nodes()}
        base_demand = {node: network._base_demand.get(node, 0.0) for node in G_sub.nodes()}
        observed = Network(
            graph=G_sub,
            suppliers=frozenset(s for s in network.suppliers if s in nodes),
            consumers=frozenset(c for c in network.consumers if c in nodes),
            base_capacities=base_capacities,
            node_supply=base_supply,
            node_demand=base_demand,
        )
        observed.edge_capacities = {
            edge: network.edge_capacities.get(edge, 0.0)
            for edge in base_capacities
        }
        observed.node_supply = {
            node: network.node_supply.get(node, 0.0)
            for node in G_sub.nodes()
        }
        observed.node_demand = {
            node: network.node_demand.get(node, 0.0)
            for node in G_sub.nodes()
        }
        observed.failed_nodes = set(network.failed_nodes).intersection(nodes)
        observed.irreversible_nodes = set(network.irreversible_nodes).intersection(nodes)
        observed.irreversible_edges = set(network.irreversible_edges).intersection(base_capacities)
        observed.disruption_metadata = dict(network.disruption_metadata)
        return observed

    def estimate_gain(self, node, observation: dict) -> float:
        """
        Оцінює локальний Δmax_flow від ремонту failed-вузла node.

        Реалізує bid(v) для auction/broadcast у Math Spec Article 4 §3.5.
        Повертає 0.0, якщо вузол не є видимим failed-кандидатом. Tie-break
        у координаторах виконується окремо через node_id, тому ця функція
        повертає лише числовий gain.
        """
        visible_failed = set(observation.get("visible_failed", set()))
        if node not in visible_failed:
            return 0.0
        network = observation["network"]
        visible_nodes = set(observation.get("visible_nodes", network.graph.nodes()))
        if node not in visible_nodes:
            return 0.0

        observed_before = self._make_observed_network(network, visible_nodes)
        F_before = compute_max_flow(observed_before)
        observed_after = observed_before.copy()
        observed_after.set_node_state(node, "active")
        F_after = compute_max_flow(observed_after)
        return float(F_after - F_before)

    @abstractmethod
    def choose_action(self, observation: dict, locked: set | None = None):
        """Повертає node_id для ремонту або None."""

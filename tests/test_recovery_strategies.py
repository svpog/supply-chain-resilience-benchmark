"""
Unit-тести recovery-стратегій статті 3.

Реалізує: розділи 3.2–3.4 протоколу R1
(див. docs/Протокол_R1.docx).
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest

from src.network.base import Network
from src.recovery.strategies import get_recovery_strategy
from src.recovery.strategies.centrality import CentralityStrategy
from src.recovery.strategies.flow_greedy import FlowGreedyStrategy
from src.recovery.strategies.flow_greedy_local import FlowGreedyLocalStrategy


def make_degree_network() -> Network:
    """Мережа, де вузол 1 має однозначно найбільший directed degree."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])
    base_cap = {edge: 10.0 for edge in G.edges()}
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 10.0
    demand[4] = 10.0
    return Network(G, frozenset({0}), frozenset({4}), base_cap, supply, demand)


def make_flow_choice_network() -> Network:
    """Мережа, де відновлення вузла 1 дає більший max-flow gain ніж вузла 2."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)])
    base_cap = {
        (0, 1): 10.0,
        (1, 4): 10.0,
        (0, 2): 2.0,
        (2, 4): 2.0,
        (0, 3): 1.0,
        (3, 4): 1.0,
    }
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 13.0
    demand[4] = 13.0
    return Network(G, frozenset({0}), frozenset({4}), base_cap, supply, demand)


def make_local_choice_network() -> Network:
    """Мережа, де r=1 локальний flow-gain вузла 1 більший ніж вузла 2."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)])
    base_cap = {
        (0, 1): 10.0,
        (1, 4): 10.0,
        (0, 2): 3.0,
        (2, 4): 3.0,
        (0, 3): 1.0,
        (3, 4): 1.0,
    }
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 14.0
    demand[4] = 14.0
    return Network(G, frozenset({0}), frozenset({4}), base_cap, supply, demand)


def test_strategy_registry_creates_all_required_strategies() -> None:
    for name in [
        "random",
        "centrality_degree",
        "centrality_betweenness",
        "flow_greedy",
        "flow_greedy_local_r0",
        "flow_greedy_local_r1",
        "flow_greedy_local_r2",
        "flow_greedy_local_r3",
        "flow_greedy_local_inf",
    ]:
        strategy = get_recovery_strategy(name)
        assert strategy is not None


def test_unknown_strategy_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Невідома recovery strategy"):
        get_recovery_strategy("does_not_exist")


def test_degree_centrality_selects_max_degree_node() -> None:
    net = make_degree_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    strategy = CentralityStrategy("degree")

    element = strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(1))

    assert element == ("node", 1)


def test_tie_breaking_is_deterministic_for_same_seed() -> None:
    net = make_degree_network()
    net.set_node_state(2, "failed")
    net.set_node_state(3, "failed")
    strategy_a = CentralityStrategy("degree")
    strategy_b = CentralityStrategy("degree")

    choice_a = strategy_a.select_element_to_repair(net, t=1, rng=np.random.default_rng(123))
    choice_b = strategy_b.select_element_to_repair(net, t=1, rng=np.random.default_rng(123))

    assert choice_a == choice_b


def test_single_failed_element_is_returned_directly() -> None:
    net = make_degree_network()
    net.set_node_state(2, "failed")
    strategy = CentralityStrategy("betweenness")

    assert strategy.select_element_to_repair(net, t=1) == ("node", 2)


def test_flow_greedy_selects_largest_flow_gain() -> None:
    net = make_flow_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    strategy = FlowGreedyStrategy()

    element = strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(7))

    assert element == ("node", 1)


def test_flow_greedy_tie_breaking_is_deterministic() -> None:
    net = make_flow_choice_network()
    net.set_node_state(2, "failed")
    net.set_node_state(3, "failed")
    strategy_a = FlowGreedyStrategy()
    strategy_b = FlowGreedyStrategy()

    choice_a = strategy_a.select_element_to_repair(net, t=1, rng=np.random.default_rng(9))
    choice_b = strategy_b.select_element_to_repair(net, t=1, rng=np.random.default_rng(9))

    assert choice_a == choice_b


def test_flow_greedy_select_performance_under_200ms() -> None:
    net = Network.create("hub", N=50, h=3, seed=42)
    for node in sorted(net.graph.nodes())[:10]:
        if node not in net.suppliers and node not in net.consumers:
            net.set_node_state(node, "failed")
    strategy = FlowGreedyStrategy()

    start = time.perf_counter()
    strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(1))
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200.0


def test_fgl_r_none_equals_full_greedy() -> None:
    """r=None має бути еквівалентним централізованому FlowGreedyStrategy."""
    net = make_flow_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    full = FlowGreedyStrategy().select_element_to_repair(net, t=1, rng=rng_a)
    local_inf = FlowGreedyLocalStrategy(r=None).select_element_to_repair(net, t=1, rng=rng_b)

    assert local_inf == full


def test_fgl_r0_sees_only_self() -> None:
    """r=0 локальний підграф навколо вузла містить тільки сам вузол."""
    net = make_flow_choice_network()
    strategy = FlowGreedyLocalStrategy(r=0)

    subgraph = strategy._get_restricted_subgraph(net.graph, 2)

    assert set(subgraph.nodes()) == {2}
    assert subgraph.number_of_edges() == 0


def test_fgl_r1_correct_neighborhood() -> None:
    """r=1 використовує undirected BFS-окіл навколо кандидата."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    strategy = FlowGreedyLocalStrategy(r=1)

    subgraph = strategy._get_restricted_subgraph(G, 2)

    assert set(subgraph.nodes()) == {1, 2, 3}
    assert set(subgraph.edges()) == {(1, 2), (2, 3)}


def test_fgl_chooses_max_local_gain() -> None:
    """finite-r flow_greedy_local обирає вузол з найбільшим локальним ΔF."""
    net = make_local_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    strategy = FlowGreedyLocalStrategy(r=1)

    element = strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(7))

    assert element == ("node", 1)


def test_fgl_deterministic_tiebreak() -> None:
    """При рівних local gain finite-r стратегія обирає менший id незалежно від rng."""
    net = make_flow_choice_network()
    net.set_node_state(2, "failed")
    net.set_node_state(3, "failed")
    strategy = FlowGreedyLocalStrategy(r=0)

    choice_a = strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(1))
    choice_b = strategy.select_element_to_repair(net, t=1, rng=np.random.default_rng(999))

    assert choice_a == ("node", 2)
    assert choice_b == ("node", 2)


def test_fgl_empty_failed_returns_none() -> None:
    """Порожній repair pool → None у поточному RecoveryStrategy API."""
    net = make_flow_choice_network()
    strategy = FlowGreedyLocalStrategy(r=1)

    assert strategy.select_element_to_repair(net, t=1) is None


def test_fgl_single_failed_element_is_returned_directly() -> None:
    """Якщо repairable element один, стратегія повертає його без max-flow sweep."""
    net = make_flow_choice_network()
    net.set_node_state(2, "failed")
    strategy = FlowGreedyLocalStrategy(r=1)

    assert strategy.select_element_to_repair(net, t=1) == ("node", 2)


def test_fgl_r2_neighborhood_superset_of_r1() -> None:
    """Збільшення r не зменшує локальний підграф."""
    net = make_flow_choice_network()
    r1 = FlowGreedyLocalStrategy(r=1)._get_restricted_subgraph(net.graph, 1)
    r2 = FlowGreedyLocalStrategy(r=2)._get_restricted_subgraph(net.graph, 1)

    assert set(r1.nodes()).issubset(set(r2.nodes()))
    assert set(r1.edges()).issubset(set(r2.edges()))

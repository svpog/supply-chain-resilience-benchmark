"""
Unit-тести M1 agent-модулів.

Реалізує: Протокол M1, §5
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.network.base import Network
from src.recovery.agents import create_agent
from src.recovery.agents.local_centrality_agent import LocalCentralityAgent
from src.recovery.agents.local_greedy_agent import LocalGreedyAgent
from src.recovery.agents.offline_oracle_agent import OfflineOracleAgent
from src.recovery.agents.random_agent import RandomAgent
from src.recovery.strategies.flow_greedy_local import FlowGreedyLocalStrategy


def make_path_network() -> Network:
    """Простий path-граф для тестів видимості."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    base_cap = {edge: 10.0 for edge in G.edges()}
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 10.0
    demand[4] = 10.0
    return Network(G, frozenset({0}), frozenset({4}), base_cap, supply, demand)


def make_choice_network() -> Network:
    """Мережа, де вузол 1 має більший flow-gain ніж вузол 2."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 4), (0, 2), (2, 4)])
    base_cap = {(0, 1): 10.0, (1, 4): 10.0, (0, 2): 2.0, (2, 4): 2.0}
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 12.0
    demand[4] = 12.0
    return Network(G, frozenset({0}), frozenset({4}), base_cap, supply, demand)


def test_agent_observe_r2() -> None:
    """r=2 бачить BFS-окіл failed-вузла у два хопи."""
    net = make_path_network()
    net.set_node_state(0, "failed")
    agent = LocalGreedyAgent(agent_id=0, r=2, network=net)

    obs = agent.observe(net, failed_nodes={0})

    assert obs["visible_nodes"] == {0, 1, 2}
    assert obs["visible_failed"] == {0}


def test_agent_observe_r_none() -> None:
    """r=None означає повну видимість."""
    net = make_path_network()
    net.set_node_state(2, "failed")
    agent = LocalGreedyAgent(agent_id=0, r=None, network=net)

    obs = agent.observe(net, failed_nodes={2})

    assert obs["visible_nodes"] == set(net.graph.nodes())
    assert obs["visible_failed"] == {2}


def test_flow_greedy_local_select_single_respects_candidates() -> None:
    """select_single() повертає node_id і не виходить за candidates."""
    net = make_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    strategy = FlowGreedyLocalStrategy(r=1)

    assert strategy.select_single(net, candidates={2}) == 2
    assert strategy.select_single(net, candidates={1, 2}) == 1


def test_random_agent_is_deterministic_for_same_seed() -> None:
    """RandomAgent відтворюваний при однаковому rng seed."""
    net = make_path_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    obs = {"visible_failed": {1, 2}}
    a = RandomAgent(agent_id=0, r=2, network=net, rng=np.random.default_rng(7))
    b = RandomAgent(agent_id=0, r=2, network=net, rng=np.random.default_rng(7))

    assert a.choose_action(obs) == b.choose_action(obs)


def test_local_centrality_chooses_highest_visible_centrality() -> None:
    """LocalCentralityAgent обирає найбільш центральний visible failed-вузол."""
    net = make_path_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    agent = LocalCentralityAgent(agent_id=0, r=None, network=net)
    obs = agent.observe(net, failed_nodes={1, 2})

    assert agent.choose_action(obs) == 2


def test_offline_oracle_registered() -> None:
    """offline_oracle доступний через AGENT_REGISTRY factory."""
    net = make_choice_network()
    agent = create_agent("offline_oracle", agent_id=0, r=None, network=net)

    assert isinstance(agent, OfflineOracleAgent)


def test_offline_oracle_uses_global_flow_gain() -> None:
    """offline_oracle обирає вузол з найбільшим глобальним Δmax_flow."""
    net = make_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    agent = OfflineOracleAgent(agent_id=0, r=None, network=net)
    obs = agent.observe(net, failed_nodes={1, 2})

    assert agent.choose_action(obs) == 1


def test_offline_oracle_respects_token_locked_nodes() -> None:
    """offline_oracle не повертає вузол, уже заблокований token coordinator."""
    net = make_choice_network()
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    agent = OfflineOracleAgent(agent_id=0, r=None, network=net)
    obs = agent.observe(net, failed_nodes={1, 2})

    assert agent.choose_action(obs, locked={1}) == 2

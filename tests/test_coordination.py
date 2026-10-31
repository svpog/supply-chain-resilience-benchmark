"""
Unit-тести M1 coordination-модулів.

Реалізує: Протокол M1, §5
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

import networkx as nx

from src.network.base import Network
from src.recovery.agents.local_centrality_agent import LocalCentralityAgent
from src.recovery.agents.local_greedy_agent import LocalGreedyAgent
from src.recovery.coordination.auction import AuctionCoordinator
from src.recovery.coordination.broadcast import BroadcastCoordinator
from src.recovery.coordination.independent import IndependentCoordinator
from src.recovery.coordination.token import TokenCoordinator


class DummyBidAgent:
    """Мінімальний агент для детермінованих auction/broadcast тестів."""

    def __init__(self, agent_id: int, visible_failed: set, gains: dict) -> None:
        self.agent_id = agent_id
        self.visible_failed = set(visible_failed)
        self.gains = dict(gains)

    def observe(self, network: Network, failed_nodes: set) -> dict:
        visible_failed = self.visible_failed.intersection(failed_nodes)
        return {
            "network": network,
            "visible_nodes": set(network.graph.nodes()),
            "visible_failed": visible_failed,
            "G_sub": network.graph.copy(),
        }

    def estimate_gain(self, node, observation: dict) -> float:
        return float(self.gains.get(node, 0.0))


def make_one_failed_network() -> Network:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])
    cap = {edge: 10.0 for edge in G.edges()}
    supply = {0: 10.0, 1: 0.0, 2: 0.0}
    demand = {0: 0.0, 1: 0.0, 2: 10.0}
    net = Network(G, frozenset({0}), frozenset({2}), cap, supply, demand)
    net.set_node_state(1, "failed")
    return net


def make_two_failed_network() -> Network:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 3), (0, 2), (2, 3)])
    cap = {edge: 10.0 for edge in G.edges()}
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 20.0
    demand[3] = 20.0
    net = Network(G, frozenset({0}), frozenset({3}), cap, supply, demand)
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    return net


def make_weighted_two_failed_network() -> Network:
    """Мережа, де ремонт 1 дає більший max-flow gain ніж ремонт 2."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 3), (0, 2), (2, 3)])
    cap = {(0, 1): 10.0, (1, 3): 10.0, (0, 2): 2.0, (2, 3): 2.0}
    supply = {node: 0.0 for node in G.nodes()}
    demand = {node: 0.0 for node in G.nodes()}
    supply[0] = 12.0
    demand[3] = 12.0
    net = Network(G, frozenset({0}), frozenset({3}), cap, supply, demand)
    net.set_node_state(1, "failed")
    net.set_node_state(2, "failed")
    return net


def test_independent_may_duplicate() -> None:
    """IndependentCoordinator дозволяє агентам обрати той самий вузол."""
    net = make_one_failed_network()
    agents = [
        LocalCentralityAgent(agent_id=0, r=None, network=net),
        LocalCentralityAgent(agent_id=1, r=None, network=net),
    ]

    actions = IndependentCoordinator().allocate(agents, net, failed_nodes={1})

    assert actions == {0: 1, 1: 1}


def test_token_no_duplicates() -> None:
    """TokenCoordinator блокує вже обрану repair-ціль."""
    net = make_two_failed_network()
    agents = [
        LocalCentralityAgent(agent_id=0, r=None, network=net),
        LocalCentralityAgent(agent_id=1, r=None, network=net),
    ]

    actions = TokenCoordinator().allocate(agents, net, failed_nodes={1, 2})
    chosen = [node for node in actions.values() if node is not None]

    assert len(chosen) == len(set(chosen))


def test_token_deterministic_agent_order() -> None:
    """TokenCoordinator використовує стабільний порядок agent_id."""
    net = make_two_failed_network()
    agents = [
        LocalCentralityAgent(agent_id=1, r=None, network=net),
        LocalCentralityAgent(agent_id=0, r=None, network=net),
    ]

    actions = TokenCoordinator().allocate(agents, net, failed_nodes={1, 2})

    assert list(actions.keys()) == [0, 1]


def test_agent_estimate_gain_prefers_higher_flow_impact() -> None:
    """AgentBase.estimate_gain дає більший bid вузлу з більшим Δflow."""
    net = make_weighted_two_failed_network()
    agent = LocalGreedyAgent(agent_id=0, r=None, network=net)
    obs = agent.observe(net, failed_nodes={1, 2})

    assert agent.estimate_gain(1, obs) == 10.0
    assert agent.estimate_gain(2, obs) == 2.0


def test_auction_no_node_collision() -> None:
    """AuctionCoordinator не призначає той самий вузол двом агентам."""
    net = make_two_failed_network()
    agents = [
        DummyBidAgent(0, {1, 2}, {1: 5.0, 2: 4.0}),
        DummyBidAgent(1, {1, 2}, {1: 5.0, 2: 4.0}),
    ]

    actions = AuctionCoordinator().allocate(agents, net, failed_nodes={1, 2})
    chosen = [node for node in actions.values() if node is not None]

    assert len(chosen) == len(set(chosen))


def test_auction_highest_bid_wins() -> None:
    """Вузол з найвищим bid дістається відповідному агенту."""
    net = make_two_failed_network()
    agents = [
        DummyBidAgent(0, {1, 2}, {1: 1.0, 2: 8.0}),
        DummyBidAgent(1, {1, 2}, {1: 10.0, 2: 0.0}),
    ]

    actions = AuctionCoordinator().allocate(agents, net, failed_nodes={1, 2})

    assert actions[1] == 1
    assert actions[0] == 2


def test_auction_deterministic_tie_break() -> None:
    """Однакові bid-и розв'язуються стабільно через node_id і agent_id."""
    net = make_two_failed_network()
    agents = [
        DummyBidAgent(1, {1, 2}, {1: 5.0, 2: 5.0}),
        DummyBidAgent(0, {1, 2}, {1: 5.0, 2: 5.0}),
    ]

    first = AuctionCoordinator().allocate(agents, net, failed_nodes={1, 2})
    second = AuctionCoordinator().allocate(agents, net, failed_nodes={1, 2})

    assert first == second
    assert first == {1: 2, 0: 1}


def test_broadcast_shared_pool() -> None:
    """Broadcast використовує union видимих failed-вузлів усіх агентів."""
    net = make_two_failed_network()
    agents = [
        DummyBidAgent(0, {1}, {1: 1.0, 2: 10.0}),
        DummyBidAgent(1, {2}, {1: 1.0, 2: 10.0}),
    ]

    actions = BroadcastCoordinator().allocate(agents, net, failed_nodes={1, 2})

    assert set(actions.values()) == {1, 2}


def test_broadcast_topK_assignment() -> None:
    """Broadcast призначає top-K вузлів за shared flow-impact."""
    net = make_two_failed_network()
    agents = [
        DummyBidAgent(0, {1, 2}, {1: 2.0, 2: 9.0}),
        DummyBidAgent(1, {1, 2}, {1: 2.0, 2: 9.0}),
    ]

    actions = BroadcastCoordinator().allocate(agents, net, failed_nodes={1, 2})

    assert actions == {0: 2, 1: 1}


def test_broadcast_still_limited_to_shared_observations() -> None:
    """Broadcast не ремонтує вузли, яких немає у shared visible_failed."""
    net = make_two_failed_network()
    net.graph.add_node(4)
    net.failed_nodes.add(4)
    agents = [
        DummyBidAgent(0, {1}, {1: 1.0, 2: 10.0, 4: 100.0}),
        DummyBidAgent(1, {2}, {1: 1.0, 2: 10.0, 4: 100.0}),
    ]

    actions = BroadcastCoordinator().allocate(agents, net, failed_nodes={1, 2, 4})

    assert 4 not in set(actions.values())


def test_all_coordinators_K1_equivalent() -> None:
    """При одному агенті всі coordination modes дають однакову дію."""
    net = make_weighted_two_failed_network()
    agent = LocalGreedyAgent(agent_id=0, r=None, network=net)
    coordinators = [
        IndependentCoordinator(),
        TokenCoordinator(),
        AuctionCoordinator(),
        BroadcastCoordinator(),
    ]

    actions = [
        coord.allocate([agent], net, failed_nodes={1, 2})[0]
        for coord in coordinators
    ]

    assert actions == [1, 1, 1, 1]

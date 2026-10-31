"""
Unit-тести моделі потоку (max flow).

Реалізує: розділ 11.2 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Тести охоплюють:
    - Лінія S→A→B→T з відомими capacities → аналітична відповідь.
    - Алмаз → max_flow = 2.
    - Розʼєднана мережа → max_flow = 0.
    - Усі 4 топології: F(0) > 0.
    - Q(t) = 1.0 для нерушеної мережі.
    - D_sat і ΔC — коректні значення.
"""

import networkx as nx
import pytest

from src.network.base import Network
from src.network.flow import (
    build_extended_graph,
    compute_max_flow,
    compute_demand_satisfaction,
    compute_Q,
    compute_delta_C,
    _SUPER_SOURCE,
    _SUPER_SINK,
)


# ─────────────────────────────────────────────
# Допоміжні фабрики мінімальних мереж
# ─────────────────────────────────────────────

def _make_line_network() -> Network:
    """
    Лінія: super-source → A → B → super-sink.
    c(super-source→A) = 5 (supply), c(A→B) = 3, c(B→super-sink) = 7 (demand).
    Очікуваний max_flow = min(5, 3, 7) = 3.
    """
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])   # 0=A, 1=B
    G.add_edge(0, 1, capacity=3.0)
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([1]),
        base_capacities={(0, 1): 3.0},
        node_supply={0: 5.0, 1: 0.0},
        node_demand={0: 0.0, 1: 7.0},
    )


def _make_diamond_network() -> Network:
    """
    Алмаз: super-source → S_node → {A, B} → T_node → super-sink,
    усі ребра capacity = 1, supply(S_node) = 2, demand(T_node) = 2.
    Очікуваний max_flow = 2 (два паралельних шляхи).
    """
    # Вузли: 0=S_node, 1=A, 2=B, 3=T_node
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, capacity=1.0)  # S → A
    G.add_edge(0, 2, capacity=1.0)  # S → B
    G.add_edge(1, 3, capacity=1.0)  # A → T
    G.add_edge(2, 3, capacity=1.0)  # B → T
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([3]),
        base_capacities={(0, 1): 1.0, (0, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0},
        node_supply={0: 2.0, 1: 0.0, 2: 0.0, 3: 0.0},
        node_demand={0: 0.0, 1: 0.0, 2: 0.0, 3: 2.0},
    )


def _make_disconnected_network() -> Network:
    """
    Розʼєднана мережа: постачальник і споживач у різних компонентах.
    Очікуваний max_flow = 0.
    """
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])  # без жодного ребра між ними
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([1]),
        base_capacities={},
        node_supply={0: 100.0, 1: 0.0},
        node_demand={0: 0.0, 1: 100.0},
    )


def _make_bottleneck_network() -> Network:
    """
    Класичний приклад Ford-Fulkerson з відомим розв'язком.

    Граф (5 вузлів, 7 ребер):
        S → A (cap=10), S → B (cap=10)
        A → C (cap=4),  A → B (cap=2), A → T (cap=8)
        B → C (cap=9),  C → T (cap=10)

    В нашій моделі: V_s={S_node}, V_c={T_node}
    supply(S) = 20 (не обмежує), demand(T) = 20 (не обмежує)
    Відомий max_flow = 18 (шляхи: 0→1→4=8, 0→2→3→4=9, 0→1→3→4=1).
    """
    G = nx.DiGraph()
    # 0=S, 1=A, 2=B, 3=C, 4=T
    edges = [
        (0, 1, 10.0), (0, 2, 10.0),
        (1, 3, 4.0),  (1, 2, 2.0), (1, 4, 8.0),
        (2, 3, 9.0),  (3, 4, 10.0),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    base_cap = {(u, v): cap for u, v, cap in edges}
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([4]),
        base_capacities=base_cap,
        node_supply={0: 20.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        node_demand={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 20.0},
    )


# ─────────────────────────────────────────────
# Тести build_extended_graph
# ─────────────────────────────────────────────

class TestBuildExtendedGraph:

    def test_super_source_and_sink_added(self):
        """Розширений граф містить вузли super-source і super-sink."""
        net = _make_line_network()
        G_ext = build_extended_graph(net)
        assert _SUPER_SOURCE in G_ext.nodes()
        assert _SUPER_SINK in G_ext.nodes()

    def test_super_source_connects_to_suppliers(self):
        """super-source → кожен постачальник з capacity = s_i."""
        net = _make_line_network()
        G_ext = build_extended_graph(net)
        for s in net.suppliers:
            assert G_ext.has_edge(_SUPER_SOURCE, s), (
                f"Немає ребра super-source → {s}"
            )
            cap = G_ext[_SUPER_SOURCE][s]["capacity"]
            assert cap == net.node_supply[s], (
                f"capacity super-source→{s}: {cap} ≠ supply={net.node_supply[s]}"
            )

    def test_consumers_connect_to_super_sink(self):
        """Кожен споживач → super-sink з capacity = d_j."""
        net = _make_line_network()
        G_ext = build_extended_graph(net)
        for c in net.consumers:
            assert G_ext.has_edge(c, _SUPER_SINK), (
                f"Немає ребра {c} → super-sink"
            )
            cap = G_ext[c][_SUPER_SINK]["capacity"]
            assert cap == net.node_demand[c], (
                f"capacity {c}→super-sink: {cap} ≠ demand={net.node_demand[c]}"
            )

    def test_failed_node_has_zero_capacity_edges(self):
        """Ребра з'єднані з failed-вузлом мають capacity = 0."""
        net = _make_line_network()
        # Встановлюємо постачальника (0) як failed
        net.set_node_state(0, "failed")
        G_ext = build_extended_graph(net)
        # super-source → failed supplier має capacity 0 (node_supply = 0)
        cap = G_ext[_SUPER_SOURCE][0]["capacity"]
        assert cap == 0.0, f"Очікувалась capacity=0 для failed вузла, отримано {cap}"
        # Вихідне ребро 0→1 теж має бути 0 (get_current_capacity)
        cap_edge = G_ext[0][1]["capacity"]
        assert cap_edge == 0.0


# ─────────────────────────────────────────────
# Тести compute_max_flow — еталонні приклади
# ─────────────────────────────────────────────

class TestComputeMaxFlow:

    def test_line_network(self):
        """Лінія: max_flow = min(5, 3, 7) = 3 (Math Spec 11.2)."""
        net = _make_line_network()
        flow = compute_max_flow(net)
        assert flow == pytest.approx(3.0), (
            f"Очікувалось 3.0, отримано {flow}"
        )

    def test_diamond_network(self):
        """Алмаз: max_flow = 2 (два паралельних шляхи, Math Spec 11.2)."""
        net = _make_diamond_network()
        flow = compute_max_flow(net)
        assert flow == pytest.approx(2.0), (
            f"Очікувалось 2.0, отримано {flow}"
        )

    def test_disconnected_network(self):
        """Розʼєднана мережа: max_flow = 0 (Math Spec 3.3 і 11.2)."""
        net = _make_disconnected_network()
        flow = compute_max_flow(net)
        assert flow == pytest.approx(0.0), (
            f"Очікувалось 0.0, отримано {flow}"
        )

    def test_bottleneck_classic(self):
        """Класичний приклад Ford-Fulkerson: max_flow = 18."""
        net = _make_bottleneck_network()
        flow = compute_max_flow(net)
        assert flow == pytest.approx(18.0), (
            f"Очікувалось 18.0, отримано {flow}"
        )

    def test_all_four_topologies_positive_flow(self):
        """Усі 4 топології N=50: F(0) > 0 (Math Spec 11.2)."""
        for topology in ["hierarchical", "hub", "reserved", "dense"]:
            net = Network.create(topology, N=50, seed=42)
            flow = compute_max_flow(net)
            assert flow > 0.0, (
                f"Топологія '{topology}': F(0)={flow} ≤ 0 — мережа не функціонує"
            )

    def test_flow_non_negative(self):
        """max_flow завжди ≥ 0."""
        for topology in ["hierarchical", "hub", "reserved", "dense"]:
            net = Network.create(topology, N=30, seed=7)
            assert compute_max_flow(net) >= 0.0

    def test_fully_failed_network_zero_flow(self):
        """Якщо всі постачальники failed — F(t) = 0."""
        net = _make_line_network()
        for s in list(net.suppliers):
            net.set_node_state(s, "failed")
        flow = compute_max_flow(net)
        assert flow == pytest.approx(0.0)

    def test_zero_capacity_edge_zero_flow(self):
        """Якщо всі ребра degraded до 0 — F(t) = 0."""
        net = _make_line_network()
        net.set_edge_capacity(0, 1, 0.0)
        flow = compute_max_flow(net)
        assert flow == pytest.approx(0.0)


# ─────────────────────────────────────────────
# Тести compute_Q
# ─────────────────────────────────────────────

class TestComputeQ:

    def test_q_equals_one_for_intact_network(self):
        """Q(0) = 1.0 для нерушеної мережі (Math Spec 5.1)."""
        net = _make_line_network()
        F0 = compute_max_flow(net)
        Q = compute_Q(net, F0)
        assert Q == pytest.approx(1.0)

    def test_q_zero_when_F0_zero(self):
        """Q = 0 якщо F0 = 0 (граничний випадок, Math Spec 3.3)."""
        net = _make_line_network()
        assert compute_Q(net, F0=0.0) == pytest.approx(0.0)

    def test_q_decreases_after_disruption(self):
        """Q(t) < 1.0 після часткового збою (якщо F(t) < F(0))."""
        net = _make_line_network()
        F0 = compute_max_flow(net)  # = 3.0
        # Деградуємо ребро A→B: c = 1.0 (замість 3.0)
        net.set_edge_capacity(0, 1, 1.0)
        Q = compute_Q(net, F0)
        assert Q < 1.0, f"Q={Q} має бути < 1.0 після деградації"
        assert Q == pytest.approx(1.0 / 3.0)

    def test_q_zero_after_full_disruption(self):
        """Q(t) = 0.0 якщо всі постачальники failed."""
        net = _make_line_network()
        F0 = compute_max_flow(net)
        for s in list(net.suppliers):
            net.set_node_state(s, "failed")
        Q = compute_Q(net, F0)
        assert Q == pytest.approx(0.0)

    def test_q_all_topologies(self):
        """Q(0) = 1.0 для всіх 4 топологій."""
        for topology in ["hierarchical", "hub", "reserved", "dense"]:
            net = Network.create(topology, N=50, seed=42)
            F0 = compute_max_flow(net)
            assert compute_Q(net, F0) == pytest.approx(1.0), (
                f"Топологія '{topology}': Q(0) ≠ 1.0"
            )


# ─────────────────────────────────────────────
# Тести compute_demand_satisfaction
# ─────────────────────────────────────────────

class TestComputeDemandSatisfaction:

    def test_intact_network(self):
        """D_sat для нерушеної мережі: F / Σd_j."""
        net = _make_line_network()
        # F = 3.0, base demand = 7.0
        D_sat = compute_demand_satisfaction(net)
        assert D_sat == pytest.approx(3.0 / 7.0)

    def test_disconnected_zero(self):
        """D_sat = 0 для розʼєднаної мережі."""
        net = _make_disconnected_network()
        assert compute_demand_satisfaction(net) == pytest.approx(0.0)

    def test_d_sat_range(self):
        """D_sat ∈ [0, 1] для всіх топологій."""
        for topology in ["hierarchical", "hub", "reserved", "dense"]:
            net = Network.create(topology, N=50, seed=1)
            D_sat = compute_demand_satisfaction(net)
            assert 0.0 <= D_sat <= 1.0 + 1e-9, (
                f"Топологія '{topology}': D_sat={D_sat} виходить за [0, 1]"
            )


# ─────────────────────────────────────────────
# Тести compute_delta_C
# ─────────────────────────────────────────────

class TestComputeDeltaC:

    def test_intact_network_zero(self):
        """ΔC(0) = 0.0 для нерушеної мережі."""
        net = _make_line_network()
        assert compute_delta_C(net) == pytest.approx(0.0)

    def test_degraded_edge(self):
        """ΔC > 0 після деградації ребра."""
        net = _make_line_network()
        # Деградуємо єдине ребро A→B з 3.0 до 1.0
        net.set_edge_capacity(0, 1, 1.0)
        delta = compute_delta_C(net)
        # (3 - 1) / 3 ≈ 0.667
        assert delta == pytest.approx(2.0 / 3.0)

    def test_all_edges_failed(self):
        """ΔC = 1.0 коли всі ребра повністю деградовані."""
        net = _make_line_network()
        net.set_edge_capacity(0, 1, 0.0)
        assert compute_delta_C(net) == pytest.approx(1.0)

    def test_delta_c_range(self):
        """ΔC ∈ [0, 1]."""
        net = Network.create("dense", N=50, seed=10)
        assert 0.0 <= compute_delta_C(net) <= 1.0 + 1e-9

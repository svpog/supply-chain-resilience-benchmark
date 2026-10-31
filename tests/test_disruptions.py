"""
Unit-тести сценаріїв збоїв і стратегій відновлення.

Реалізує: розділ 11.3 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Тести охоплюють:
    - NodeFailureDisruption: статистика Bernoulli за 10 000 прогонів ≈ p_v ± 2σ.
    - EdgeDegradationDisruption: фактор (1 − γ) застосований правильно.
    - ClusterDisruption: всі failed вузли мають dist ≤ r_clust від епіцентру.
    - RecoveryStrategy: повне відновлення → Q(t) ≈ 1.0.
    - Irreversible (w > 0): частка w позначена як незворотна.
    - Оригінальна мережа не змінюється після apply().
"""

import math

import networkx as nx
import numpy as np
import pytest

from src.network.base import Network
from src.network.flow import compute_max_flow, compute_Q
from src.disruptions.node_failure import NodeFailureDisruption
from src.disruptions.fixed_budget_node import FixedBudgetNodeDisruption
from src.disruptions.edge_degradation import EdgeDegradationDisruption
from src.disruptions.cluster import ClusterDisruption
from src.recovery.random_strategy import RandomRecoveryStrategy


# ─────────────────────────────────────────────
# Допоміжні фабрики
# ─────────────────────────────────────────────

def _make_line_network() -> Network:
    """Лінія A→B з відомими параметрами (аналог test_flow.py)."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, capacity=3.0)
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([1]),
        base_capacities={(0, 1): 3.0},
        node_supply={0: 5.0, 1: 0.0},
        node_demand={0: 0.0, 1: 7.0},
    )


def _make_grid_network(n: int = 4) -> Network:
    """
    Проста решітка n×n з ребрами →праворуч і →вниз.
    Постачальники = перший рядок, споживачі = останній рядок.
    Використовується для тесту ClusterDisruption.
    """
    G = nx.DiGraph()
    nodes = list(range(n * n))
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(n):
            node = i * n + j
            if j + 1 < n:
                G.add_edge(node, node + 1, capacity=10.0)
            if i + 1 < n:
                G.add_edge(node, node + n, capacity=10.0)

    suppliers = frozenset(range(n))           # перший рядок
    consumers = frozenset(range(n * (n - 1), n * n))  # останній рядок
    base_cap = {(u, v): 10.0 for u, v in G.edges()}
    supply = {nd: (50.0 if nd in suppliers else 0.0) for nd in nodes}
    demand = {nd: (50.0 if nd in consumers else 0.0) for nd in nodes}
    return Network(
        graph=G,
        suppliers=suppliers,
        consumers=consumers,
        base_capacities=base_cap,
        node_supply=supply,
        node_demand=demand,
    )


# ─────────────────────────────────────────────
# Тести NodeFailureDisruption
# ─────────────────────────────────────────────

class TestNodeFailureDisruption:

    def test_failed_nodes_in_network(self):
        """Після apply() failed_nodes ≠ порожня при p_v = 1.0."""
        net = Network.create("hierarchical", N=30, seed=1)
        d = NodeFailureDisruption(p_v=1.0)
        disrupted = d.apply(net, seed=0)
        assert len(disrupted.failed_nodes) == net.N

    def test_no_failures_when_p_zero(self):
        """p_v = 0 → жодного збою."""
        net = Network.create("hierarchical", N=30, seed=1)
        d = NodeFailureDisruption(p_v=0.0)
        disrupted = d.apply(net, seed=0)
        assert len(disrupted.failed_nodes) == 0

    def test_bernoulli_statistics(self):
        """
        За 10 000 прогонів середня частка failed вузлів ≈ p_v ± 2σ.

        Math Spec 11.3: σ = sqrt(p_v·(1-p_v) / (n_runs · N)).
        """
        net = Network.create("hierarchical", N=30, seed=42)
        p_v = 0.25
        d = NodeFailureDisruption(p_v=p_v)
        n_runs = 10_000
        fractions = []
        for seed in range(n_runs):
            disrupted = d.apply(net, seed=seed)
            fractions.append(len(disrupted.failed_nodes) / net.N)

        avg = float(np.mean(fractions))
        sigma = math.sqrt(p_v * (1 - p_v) / (n_runs * net.N))
        assert abs(avg - p_v) < 2 * sigma, (
            f"Середня частка failed={avg:.4f} відхиляється від p_v={p_v} "
            f"більш ніж на 2σ={2*sigma:.4f}"
        )

    def test_original_network_unchanged(self):
        """apply() не змінює оригінальну мережу."""
        net = Network.create("hub", N=20, seed=5)
        original_failed = len(net.failed_nodes)
        d = NodeFailureDisruption(p_v=0.5)
        d.apply(net, seed=1)
        assert len(net.failed_nodes) == original_failed

    def test_failed_node_supply_zeroed(self):
        """Після збою постачальника його node_supply = 0."""
        net = _make_line_network()
        d = NodeFailureDisruption(p_v=1.0)
        disrupted = d.apply(net, seed=0)
        for s in disrupted.suppliers:
            assert disrupted.node_supply[s] == 0.0

    def test_failed_node_capacity_zero(self):
        """get_current_capacity → 0 для ребер, суміжних із failed вузлом."""
        net = _make_line_network()
        d = NodeFailureDisruption(p_v=1.0)
        disrupted = d.apply(net, seed=0)
        assert disrupted.get_current_capacity(0, 1) == 0.0

    def test_invalid_p_v_raises(self):
        """p_v поза [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            NodeFailureDisruption(p_v=1.5)
        with pytest.raises(ValueError):
            NodeFailureDisruption(p_v=-0.1)

    def test_irreversible_nodes_marked_when_w_positive(self):
        """При w > 0 частина failed вузлів позначена як irreversible."""
        net = Network.create("hierarchical", N=30, seed=7)
        d = NodeFailureDisruption(p_v=1.0, w=0.5)
        disrupted = d.apply(net, seed=0)
        n_failed = len(disrupted.failed_nodes)
        n_irr = len(disrupted.irreversible_nodes)
        # Очікуємо ≈ 50% незворотних
        assert n_irr > 0, "Незворотних вузлів немає, хоча w=0.5"
        assert n_irr <= n_failed, "Незворотних не може бути більше ніж failed"

    def test_irreversible_nodes_empty_when_w_zero(self):
        """При w = 0 (за замовчуванням) — irreversible_nodes порожня."""
        net = Network.create("hierarchical", N=30, seed=3)
        d = NodeFailureDisruption(p_v=0.5)
        disrupted = d.apply(net, seed=1)
        assert len(disrupted.irreversible_nodes) == 0


class TestFixedBudgetNodeDisruption:
    """S3 fixed-budget disruptions: fair random vs targeted comparison."""

    def test_random_fixed_k_fails_exactly_k_transit_nodes(self):
        """random_fixed_k пошкоджує рівно k_attack вузлів з C_attack."""
        net = Network.create("hub", N=50, h=3, seed=42)
        d = FixedBudgetNodeDisruption(k_attack=5, mode="random_fixed_k")
        disrupted = d.apply(net, seed=1)
        assert len(disrupted.failed_nodes) == 5
        assert disrupted.disruption_metadata["candidate_set_size"] == 30
        assert disrupted.disruption_metadata["k_attack"] == 5
        assert disrupted.failed_nodes.isdisjoint(net.suppliers)
        assert disrupted.failed_nodes.isdisjoint(net.consumers)

    def test_attack_fraction_resolves_to_ceil_budget(self):
        """attack_fraction q → k_attack=max(1, ceil(q·|C_attack|))."""
        net = Network.create("hub", N=50, h=3, seed=42)
        d = FixedBudgetNodeDisruption(attack_fraction=0.10, mode="random_fixed_k")
        disrupted = d.apply(net, seed=2)
        assert disrupted.disruption_metadata["candidate_set_size"] == 30
        assert disrupted.disruption_metadata["k_attack"] == 3
        assert len(disrupted.failed_nodes) == 3

    def test_targeted_betweenness_does_not_use_h_hubs_as_budget(self):
        """
        k_attack не дорівнює автоматично h_hubs.

        Для h=8 і q=0.02 маємо C_attack=30 → k_attack=1, не 8.
        """
        net = Network.create("hub", N=50, h=8, seed=42)
        d = FixedBudgetNodeDisruption(
            attack_fraction=0.02,
            mode="targeted",
            attack_rule="betweenness",
        )
        disrupted = d.apply(net, seed=3)
        assert disrupted.disruption_metadata["k_attack"] == 1
        assert len(disrupted.failed_nodes) == 1

    def test_targeted_hub_role_prefers_hubs(self):
        """hub_role rule спершу вибирає вузли з role='hub'."""
        net = Network.create("hub", N=50, h=3, seed=42)
        d = FixedBudgetNodeDisruption(k_attack=2, mode="targeted", attack_rule="hub_role")
        disrupted = d.apply(net, seed=4)
        assert disrupted.disruption_metadata["failed_hubs_count"] == 2

    def test_invalid_fixed_budget_params_raise(self):
        """Некоректні параметри fixed-budget disruption → ValueError."""
        with pytest.raises(ValueError):
            FixedBudgetNodeDisruption()
        with pytest.raises(ValueError):
            FixedBudgetNodeDisruption(attack_fraction=0.0)
        with pytest.raises(ValueError):
            FixedBudgetNodeDisruption(k_attack=0)
        with pytest.raises(ValueError):
            FixedBudgetNodeDisruption(k_attack=1, mode="random")


# ─────────────────────────────────────────────
# Тести EdgeDegradationDisruption
# ─────────────────────────────────────────────

class TestEdgeDegradationDisruption:

    def test_gamma_factor_applied(self):
        """Деградоване ребро має capacity = (1 − γ) · c_base."""
        net = _make_line_network()
        gamma = 0.4
        d = EdgeDegradationDisruption(p_e=1.0, gamma=gamma)
        disrupted = d.apply(net, seed=0)
        base_cap = net._base_capacities[(0, 1)]
        current_cap = disrupted.edge_capacities[(0, 1)]
        expected = (1.0 - gamma) * base_cap
        assert current_cap == pytest.approx(expected), (
            f"capacity={current_cap} ≠ (1−γ)·c_base={expected}"
        )

    def test_no_degradation_when_p_e_zero(self):
        """p_e = 0 → жодне ребро не деградує."""
        net = Network.create("hierarchical", N=30, seed=1)
        d = EdgeDegradationDisruption(p_e=0.0, gamma=0.5)
        disrupted = d.apply(net, seed=0)
        for edge, cap in disrupted.edge_capacities.items():
            base = disrupted._base_capacities[edge]
            assert cap == pytest.approx(base), (
                f"Ребро {edge}: capacity змінилась до {cap} при p_e=0"
            )

    def test_all_degraded_when_p_e_one(self):
        """p_e = 1 → всі ребра деградують."""
        net = _make_line_network()
        d = EdgeDegradationDisruption(p_e=1.0, gamma=0.3)
        disrupted = d.apply(net, seed=0)
        for edge, cap in disrupted.edge_capacities.items():
            base = disrupted._base_capacities[edge]
            assert cap == pytest.approx((1.0 - 0.3) * base)

    def test_gamma_one_zeroes_capacity(self):
        """γ = 1.0 → capacity = 0 для всіх деградованих ребер."""
        net = _make_line_network()
        d = EdgeDegradationDisruption(p_e=1.0, gamma=1.0)
        disrupted = d.apply(net, seed=0)
        assert disrupted.edge_capacities[(0, 1)] == pytest.approx(0.0)

    def test_original_network_unchanged(self):
        """apply() не змінює оригінальну мережу."""
        net = Network.create("hub", N=20, seed=5)
        original_caps = dict(net.edge_capacities)
        d = EdgeDegradationDisruption(p_e=0.8, gamma=0.5)
        d.apply(net, seed=2)
        assert net.edge_capacities == original_caps

    def test_bernoulli_edge_statistics(self):
        """
        За 5 000 прогонів середня частка деградованих ребер ≈ p_e ± 2σ.
        """
        net = Network.create("dense", N=20, seed=10)
        p_e = 0.3
        d = EdgeDegradationDisruption(p_e=p_e, gamma=0.5)
        n_runs = 5_000
        L = net.L
        fractions = []
        for seed in range(n_runs):
            disrupted = d.apply(net, seed=seed)
            degraded = sum(
                1 for (u, v), cap in disrupted.edge_capacities.items()
                if cap < disrupted._base_capacities[(u, v)]
            )
            fractions.append(degraded / L)

        avg = float(np.mean(fractions))
        sigma = math.sqrt(p_e * (1 - p_e) / (n_runs * L))
        assert abs(avg - p_e) < 2 * sigma + 1e-3, (
            f"Середня частка деградованих={avg:.4f} відхиляється від p_e={p_e}"
        )

    def test_invalid_params_raise(self):
        """Некоректні параметри → ValueError."""
        with pytest.raises(ValueError):
            EdgeDegradationDisruption(p_e=1.5, gamma=0.5)
        with pytest.raises(ValueError):
            EdgeDegradationDisruption(p_e=0.5, gamma=-0.1)

    def test_irreversible_edges_marked_when_w_positive(self):
        """При w > 0 частина деградованих ребер позначена як irreversible."""
        net = Network.create("dense", N=20, seed=4)
        d = EdgeDegradationDisruption(p_e=1.0, gamma=0.5, w=0.5)
        disrupted = d.apply(net, seed=0)
        assert len(disrupted.irreversible_edges) > 0, (
            "Незворотних ребер немає, хоча w=0.5"
        )

    def test_target_all_degrades_all_reserved_edges(self):
        """target='all' лишає legacy-поведінку: деградують усі ребра."""
        net = Network.create("reserved", N=50, H=4, b=3, K_res=2, seed=42)
        d = EdgeDegradationDisruption(p_e=1.0, gamma=0.5, target="all")
        disrupted = d.apply(net, seed=0)

        for edge, cap in disrupted.edge_capacities.items():
            base = disrupted._base_capacities[edge]
            assert cap == pytest.approx(0.5 * base)

    def test_target_trunk_only_keeps_backup_edges_intact(self):
        """target='trunk_only' деградує магістральні ребра, але не backup."""
        net = Network.create("reserved", N=50, H=4, b=3, K_res=2, seed=42)
        d = EdgeDegradationDisruption(p_e=1.0, gamma=0.5, target="trunk_only")
        disrupted = d.apply(net, seed=0)

        assert any(
            data.get("edge_type") == "backup"
            for _, _, data in net.graph.edges(data=True)
        )
        for edge, cap in disrupted.edge_capacities.items():
            base = disrupted._base_capacities[edge]
            edge_type = disrupted.graph.edges[edge].get("edge_type")
            if edge_type == "backup":
                assert cap == pytest.approx(base)
            else:
                assert cap == pytest.approx(0.5 * base)

    def test_target_backup_only_keeps_trunk_edges_intact(self):
        """target='backup_only' деградує лише резервні ребра."""
        net = Network.create("reserved", N=50, H=4, b=3, K_res=2, seed=42)
        d = EdgeDegradationDisruption(p_e=1.0, gamma=0.5, target="backup_only")
        disrupted = d.apply(net, seed=0)

        for edge, cap in disrupted.edge_capacities.items():
            base = disrupted._base_capacities[edge]
            edge_type = disrupted.graph.edges[edge].get("edge_type")
            if edge_type == "backup":
                assert cap == pytest.approx(0.5 * base)
            else:
                assert cap == pytest.approx(base)

    def test_invalid_target_raises(self):
        """Некоректний target для edge_degradation → ValueError."""
        with pytest.raises(ValueError):
            EdgeDegradationDisruption(p_e=0.2, gamma=0.5, target="mainline")


# ─────────────────────────────────────────────
# Тести ClusterDisruption
# ─────────────────────────────────────────────

class TestClusterDisruption:

    def test_epicenter_in_metadata(self):
        """Епіцентр зберігається у disruption_metadata['epicenter']."""
        net = _make_grid_network(n=5)
        d = ClusterDisruption(r_clust=1)
        disrupted = d.apply(net, seed=0)
        assert "epicenter" in disrupted.disruption_metadata

    def test_all_failed_within_radius(self):
        """
        Всі failed вузли мають dist_undirected(epicenter, v) ≤ r_clust.

        Math Spec 11.3: перевірка геометричної коректності BFS-кулі.
        """
        n = 6
        net = _make_grid_network(n=n)
        r_clust = 2
        d = ClusterDisruption(r_clust=r_clust)

        for seed in range(20):
            disrupted = d.apply(net, seed=seed)
            epicenter = disrupted.disruption_metadata["epicenter"]
            undirected = disrupted.graph.to_undirected()
            lengths = nx.single_source_shortest_path_length(
                undirected, epicenter
            )
            for node in disrupted.failed_nodes:
                dist = lengths.get(node, float("inf"))
                assert dist <= r_clust, (
                    f"seed={seed}: вузол {node} на відстані {dist} > r_clust={r_clust}"
                )

    def test_epicenter_always_failed(self):
        """Епіцентр сам завжди входить до failed_nodes (r_clust ≥ 0)."""
        net = Network.create("hierarchical", N=30, seed=1)
        d = ClusterDisruption(r_clust=0)
        for seed in range(10):
            disrupted = d.apply(net, seed=seed)
            epicenter = disrupted.disruption_metadata["epicenter"]
            assert epicenter in disrupted.failed_nodes

    def test_r_zero_fails_only_epicenter(self):
        """r_clust = 0 → падає лише епіцентр."""
        net = _make_grid_network(n=4)
        d = ClusterDisruption(r_clust=0)
        disrupted = d.apply(net, seed=7)
        assert len(disrupted.failed_nodes) == 1

    def test_large_radius_fails_all(self):
        """r_clust ≥ діаметр графу → всі вузли failed."""
        net = _make_grid_network(n=3)
        d = ClusterDisruption(r_clust=100)
        disrupted = d.apply(net, seed=0)
        assert len(disrupted.failed_nodes) == net.N

    def test_original_network_unchanged(self):
        """apply() не змінює оригінальну мережу."""
        net = Network.create("dense", N=20, seed=2)
        original_failed = set(net.failed_nodes)
        d = ClusterDisruption(r_clust=2)
        d.apply(net, seed=0)
        assert net.failed_nodes == original_failed

    def test_invalid_r_clust_raises(self):
        """r_clust < 0 → ValueError."""
        with pytest.raises(ValueError):
            ClusterDisruption(r_clust=-1)

    def test_no_params_raises(self):
        """Жоден параметр не задано → ValueError."""
        with pytest.raises(ValueError):
            ClusterDisruption()

    def test_invalid_target_fraction_raises(self):
        """target_fraction поза (0, 1] → ValueError."""
        with pytest.raises(ValueError):
            ClusterDisruption(target_fraction=0.0)
        with pytest.raises(ValueError):
            ClusterDisruption(target_fraction=1.5)

    def test_irreversible_nodes_marked_when_w_positive(self):
        """При w > 0 частина failed вузлів є irreversible."""
        net = _make_grid_network(n=4)
        d = ClusterDisruption(r_clust=3, w=0.5)
        disrupted = d.apply(net, seed=0)
        if len(disrupted.failed_nodes) > 0:
            assert len(disrupted.irreversible_nodes) > 0

    def test_target_fraction_exact_count(self):
        """
        target_fraction=0.20 при N=50 → рівно 10 failed вузлів
        (або менше, якщо граф має < 10 вузлів від епіцентра).
        """
        net = Network.create("hierarchical", N=50, seed=42)
        target = 0.20
        d = ClusterDisruption(target_fraction=target)
        for seed in range(10):
            disrupted = d.apply(net, seed=seed)
            expected = max(1, int(target * net.N))
            # Може бути менше, якщо граф вичерпано, але не більше
            assert len(disrupted.failed_nodes) <= expected + 1
            # Ніколи не більше target_n + 1 (допуск для крайнього рівня)
            assert len(disrupted.failed_nodes) >= min(expected, net.N)

    def test_target_fraction_topology_independence(self):
        """
        target_fraction дає приблизно однаковий % failed вузлів
        для різних топологій (на відміну від r_clust).
        """
        target = 0.20
        d = ClusterDisruption(target_fraction=target)

        for topo in ["hierarchical", "dense", "hub"]:
            net = Network.create(topo, N=50, seed=7)
            counts = []
            for seed in range(30):
                disrupted = d.apply(net, seed=seed)
                counts.append(len(disrupted.failed_nodes) / net.N)
            avg = float(np.mean(counts))
            assert abs(avg - target) < 0.05, (
                f"{topo}: avg failed={avg:.3f}, очікувалось {target} ± 0.05"
            )

    def test_target_fraction_epicenter_in_failed(self):
        """Епіцентр завжди входить до failed_nodes при target_fraction."""
        net = Network.create("dense", N=50, seed=1)
        d = ClusterDisruption(target_fraction=0.10)
        for seed in range(10):
            disrupted = d.apply(net, seed=seed)
            epicenter = disrupted.disruption_metadata["epicenter"]
            assert epicenter in disrupted.failed_nodes

    def test_target_fraction_priority_over_r_clust(self):
        """
        Якщо задані обидва параметри, target_fraction має пріоритет.
        Результат має відповідати target_fraction, не r_clust=0 (лише епіцентр).
        """
        net = Network.create("hierarchical", N=50, seed=3)
        d = ClusterDisruption(r_clust=0, target_fraction=0.20)
        disrupted = d.apply(net, seed=0)
        # Якби діяв r_clust=0 → лише 1 вузол; target_fraction=0.20 → 10 вузлів
        assert len(disrupted.failed_nodes) > 1


# ─────────────────────────────────────────────
# Тести RandomRecoveryStrategy
# ─────────────────────────────────────────────

class TestRandomRecoveryStrategy:

    def test_returns_none_on_intact_network(self):
        """Якщо мережа ціла — select_element_to_repair → None."""
        net = _make_line_network()
        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)
        assert strategy.select_element_to_repair(net, t=0, rng=rng) is None

    def test_returns_node_element_format(self):
        """Якщо є failed вузол — повертає ("node", node_id)."""
        net = _make_line_network()
        net.set_node_state(0, "failed")
        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)
        element = strategy.select_element_to_repair(net, t=0, rng=rng)
        assert element is not None
        kind, target = element
        assert kind == "node"
        assert target == 0

    def test_returns_edge_element_format(self):
        """Якщо є деградоване ребро (без failed вузлів) — повертає ("edge", (u,v))."""
        net = _make_line_network()
        net.set_edge_capacity(0, 1, 1.0)
        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)
        element = strategy.select_element_to_repair(net, t=0, rng=rng)
        assert element is not None
        kind, target = element
        assert kind == "edge"
        assert target == (0, 1)

    def test_repair_node_restores_supply(self):
        """Після repair("node") node_supply відновлюється до базового значення."""
        net = _make_line_network()
        net.set_node_state(0, "failed")
        strategy = RandomRecoveryStrategy()
        strategy.repair(net, ("node", 0))
        assert net.node_supply[0] == pytest.approx(net._base_supply[0])
        assert 0 not in net.failed_nodes

    def test_repair_edge_restores_capacity(self):
        """Після repair("edge") capacity ребра повертається до базового."""
        net = _make_line_network()
        net.set_edge_capacity(0, 1, 0.5)
        strategy = RandomRecoveryStrategy()
        strategy.repair(net, ("edge", (0, 1)))
        assert net.edge_capacities[(0, 1)] == pytest.approx(
            net._base_capacities[(0, 1)]
        )

    def test_full_recovery_restores_flow(self):
        """
        Повне відновлення після збою → Q(t) ≈ 1.0.

        Math Spec 11.3: recovery повертає мережу до початкового стану.
        """
        net = Network.create("hierarchical", N=30, seed=42)
        F0 = compute_max_flow(net)

        d = NodeFailureDisruption(p_v=0.3)
        disrupted = d.apply(net, seed=7)

        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)

        # Відновлюємо по одному елементу поки є що відновлювати
        max_steps = net.N * 2
        for _ in range(max_steps):
            element = strategy.select_element_to_repair(disrupted, t=0, rng=rng)
            if element is None:
                break
            strategy.repair(disrupted, element)

        Q = compute_Q(disrupted, F0)
        assert Q == pytest.approx(1.0), (
            f"Після повного відновлення Q={Q:.4f} ≠ 1.0"
        )

    def test_full_recovery_after_edge_degradation(self):
        """Повне відновлення після деградації ребер → Q ≈ 1.0."""
        net = Network.create("dense", N=20, seed=10)
        F0 = compute_max_flow(net)

        d = EdgeDegradationDisruption(p_e=0.5, gamma=0.8)
        disrupted = d.apply(net, seed=3)

        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(1)

        max_steps = net.L * 2
        for _ in range(max_steps):
            element = strategy.select_element_to_repair(disrupted, t=0, rng=rng)
            if element is None:
                break
            strategy.repair(disrupted, element)

        Q = compute_Q(disrupted, F0)
        assert Q == pytest.approx(1.0), (
            f"Після повного відновлення ребер Q={Q:.4f} ≠ 1.0"
        )

    def test_irreversible_nodes_not_repaired(self):
        """Незворотно пошкоджені вузли не потрапляють до списку на відновлення."""
        net = _make_line_network()
        net.set_node_state(0, "failed")
        net.irreversible_nodes.add(0)

        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)
        # Вузол 0 є irreversible — не повинен бути вибраним
        element = strategy.select_element_to_repair(net, t=0, rng=rng)
        if element is not None:
            kind, target = element
            assert not (kind == "node" and target == 0), (
                "Незворотний вузол не повинен обиратися для відновлення"
            )

    def test_invalid_repair_element_raises(self):
        """repair() з невідомим типом елемента → ValueError."""
        net = _make_line_network()
        strategy = RandomRecoveryStrategy()
        with pytest.raises(ValueError):
            strategy.repair(net, ("unknown_type", 0))

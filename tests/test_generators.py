"""
Unit-тести генераторів топологій мережі поставок.

Реалізує: розділ 11.1 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Тести охоплюють:
    - Ієрархічна: дерево (N−1 ребер для V0_count=1); умови дерева.
    - Хабова: рівно h хабів; кожен non-hub з'єднаний з ≥ 1 хабом; зв'язність.
    - Резервована: більше ребер, ніж у базовому дереві; альтернативні шляхи.
    - Щільна: середній ступінь ≈ k_avg; зв'язність.
    - Усі генератори: V_s ∩ V_c = ∅; N вузлів; F(0) > 0 (перевіряється в test_flow.py).
"""

import math

import networkx as nx
import pytest

from src.network.base import Network
from src.network.generators import hierarchical, hub, reserved, dense
from src.network.flow import compute_max_flow, compute_demand_satisfaction


# ─────────────────────────────────────────────
# Допоміжні функції
# ─────────────────────────────────────────────

def _is_weakly_connected(net: Network) -> bool:
    """Повертає True, якщо граф слабко зв'язний."""
    return nx.is_weakly_connected(net.graph)


def _avg_degree(net: Network) -> float:
    """Середній повний ступінь (in + out) вузла."""
    degrees = [net.graph.in_degree(n) + net.graph.out_degree(n)
               for n in net.graph.nodes()]
    return sum(degrees) / len(degrees) if degrees else 0.0


# ─────────────────────────────────────────────
# Ієрархічна топологія
# ─────────────────────────────────────────────

class TestHierarchical:

    def test_single_root_is_tree(self):
        """При V0_count=1 граф — дерево з рівно N−1 ребрами."""
        net = hierarchical.generate(N=50, H=4, V0_count=1, seed=0)
        assert net.L == net.N - 1, (
            f"Дерево має мати N−1={net.N - 1} ребер, отримано {net.L}"
        )

    def test_correct_node_count(self):
        """Кількість вузлів дорівнює N (N ≥ 30 для H=4, V0_count=2)."""
        for N in [30, 50, 80]:
            net = hierarchical.generate(N=N, H=4, seed=1)
            assert net.N == N, f"Очікувалось N={N}, отримано {net.N}"

    def test_non_root_has_one_parent(self):
        """Кожен вузол λ > 0 має рівно одного батька (in-degree = 1)."""
        net = hierarchical.generate(N=50, H=4, V0_count=1, seed=2)
        for node, data in net.graph.nodes(data=True):
            level = data.get("level", 0)
            if level > 0:
                assert net.graph.in_degree(node) == 1, (
                    f"Вузол {node} рівня {level} має in-degree "
                    f"{net.graph.in_degree(node)}, очікувалось 1"
                )

    def test_non_leaf_has_children(self):
        """Кожен вузол λ < H−1 має щонайменше одного нащадка (out-degree ≥ 1)."""
        H = 4
        net = hierarchical.generate(N=50, H=H, V0_count=2, seed=3)
        for node, data in net.graph.nodes(data=True):
            level = data.get("level", 0)
            if level < H - 1:
                assert net.graph.out_degree(node) >= 1, (
                    f"Вузол {node} рівня {level} має out-degree 0, "
                    f"але має мати ≥ 1 нащадків"
                )

    def test_vs_vc_disjoint(self):
        """V_s ∩ V_c = ∅ (Math Spec 1.2)."""
        net = hierarchical.generate(N=50, H=4, seed=4)
        assert net.suppliers & net.consumers == frozenset(), (
            "V_s і V_c перетинаються — порушення Math Spec 1.2"
        )

    def test_vs_is_level_0(self):
        """V_s = всі вузли рівня 0."""
        net = hierarchical.generate(N=50, H=4, V0_count=2, seed=5)
        level_0 = frozenset(
            n for n, d in net.graph.nodes(data=True) if d.get("level") == 0
        )
        assert net.suppliers == level_0

    def test_vc_is_last_level(self):
        """V_c = всі вузли останнього рівня."""
        H = 4
        net = hierarchical.generate(N=50, H=H, seed=6)
        level_last = frozenset(
            n for n, d in net.graph.nodes(data=True) if d.get("level") == H - 1
        )
        assert net.consumers == level_last

    def test_supply_surplus_20_percent(self):
        """Σs_i = 1.2 · Σd_j (Math Spec 1.3)."""
        net = hierarchical.generate(N=50, H=4, seed=7)
        total_supply = sum(net.node_supply.values())
        total_demand = sum(net.node_demand.values())
        assert abs(total_supply - 1.2 * total_demand) < 1e-6, (
            f"total_supply={total_supply:.2f} ≠ 1.2·total_demand={1.2*total_demand:.2f}"
        )

    def test_reproducibility(self):
        """Однаковий seed → однакова мережа."""
        net1 = hierarchical.generate(N=50, H=4, seed=42)
        net2 = hierarchical.generate(N=50, H=4, seed=42)
        assert net1.L == net2.L
        assert set(net1.edges) == set(net2.edges)

    def test_different_seeds_differ(self):
        """Різні seed → різні мережі (майже завжди)."""
        net1 = hierarchical.generate(N=50, H=4, seed=0)
        net2 = hierarchical.generate(N=50, H=4, seed=99)
        assert set(net1.edges) != set(net2.edges)


# ─────────────────────────────────────────────
# Хабова топологія
# ─────────────────────────────────────────────

class TestHub:

    def test_correct_hub_count(self):
        """Граф містить рівно h хабових вузлів."""
        for h in [1, 2, 3, 5]:
            net = hub.generate(N=50, h=h, seed=10)
            hub_count = sum(
                1 for n, d in net.graph.nodes(data=True) if d.get("role") == "hub"
            )
            assert hub_count == h, f"Очікувалось h={h} хабів, отримано {hub_count}"

    def test_every_non_hub_connected_to_hub(self):
        """Кожен non-hub-вузол з'єднаний щонайменше з одним хабом."""
        net = hub.generate(N=50, h=3, seed=11)
        hub_node_ids = {
            n for n, d in net.graph.nodes(data=True) if d.get("role") == "hub"
        }
        for node, data in net.graph.nodes(data=True):
            if data.get("role") == "non_hub":
                neighbors = set(net.graph.successors(node)) | set(net.graph.predecessors(node))
                connected_to_hub = neighbors & hub_node_ids
                assert connected_to_hub, (
                    f"Non-hub вузол {node} не з'єднаний жодним хабом"
                )

    def test_weakly_connected(self):
        """Граф слабко зв'язний."""
        net = hub.generate(N=50, h=3, seed=12)
        assert _is_weakly_connected(net), "Хабовий граф має бути слабко зв'язним"

    def test_vs_vc_disjoint(self):
        """V_s ∩ V_c = ∅."""
        net = hub.generate(N=50, h=3, seed=13)
        assert net.suppliers & net.consumers == frozenset()

    def test_hubs_not_in_vs_or_vc(self):
        """Хаби не є постачальниками і не є споживачами (вони транзитні)."""
        net = hub.generate(N=50, h=3, seed=14)
        hub_node_ids = {
            n for n, d in net.graph.nodes(data=True) if d.get("role") == "hub"
        }
        assert not (hub_node_ids & net.suppliers), "Хаби не мають бути в V_s"
        assert not (hub_node_ids & net.consumers), "Хаби не мають бути в V_c"

    def test_correct_node_count(self):
        """Кількість вузлів дорівнює N."""
        net = hub.generate(N=50, h=3, seed=15)
        assert net.N == 50

    def test_supply_surplus_20_percent(self):
        """Σs_i = 1.2 · Σd_j."""
        net = hub.generate(N=50, h=3, seed=16)
        total_supply = sum(net.node_supply.values())
        total_demand = sum(net.node_demand.values())
        assert abs(total_supply - 1.2 * total_demand) < 1e-6

    def test_reproducibility(self):
        """Однаковий seed → однакова мережа."""
        net1 = hub.generate(N=50, h=3, seed=42)
        net2 = hub.generate(N=50, h=3, seed=42)
        assert set(net1.edges) == set(net2.edges)


# ─────────────────────────────────────────────
# Резервована топологія
# ─────────────────────────────────────────────

class TestReserved:

    def test_more_edges_than_tree(self):
        """Резервована топологія має більше ребер, ніж базове дерево."""
        net_hier = hierarchical.generate(N=50, H=4, V0_count=2, seed=20)
        net_res = reserved.generate(N=50, H=4, V0_count=2, K_res=2, seed=20)
        assert net_res.L > net_hier.L, (
            f"Резервована ({net_res.L} ребер) має більше ребер, "
            f"ніж базове дерево ({net_hier.L})"
        )

    def test_alternative_paths_exist(self):
        """Між деякими парами V_s × V_c існує > 1 ребронезалежних шляхи."""
        # K_res=4 дає достатньо додаткових ребер для гарантованих альтернатив
        net = reserved.generate(N=50, H=4, V0_count=2, K_res=4, seed=21)

        # Перевіряємо, що хоча б одна пара V_s × V_c має шлях взагалі
        has_path = False
        for s in net.suppliers:
            for c in net.consumers:
                if nx.has_path(net.graph, s, c):
                    has_path = True
                    break
            if has_path:
                break
        assert has_path, "Немає жодного шляху між V_s і V_c у резервованій мережі"

        # Шукаємо пару з ≥ 2 ребронезалежними шляхами (серед усіх пар V_s × V_c)
        found_multiple = False
        for s in net.suppliers:
            for c in net.consumers:
                try:
                    paths = list(nx.edge_disjoint_paths(net.graph, s, c))
                    if len(paths) >= 2:
                        found_multiple = True
                        break
                except nx.NetworkXNoPath:
                    pass
            if found_multiple:
                break
        assert found_multiple, (
            "Для K_res=4 серед усіх пар V_s×V_c має існувати ≥ 1 пара "
            "з ≥ 2 ребронезалежними шляхами"
        )

    def test_vs_vc_disjoint(self):
        """V_s ∩ V_c = ∅."""
        net = reserved.generate(N=50, H=4, K_res=2, seed=22)
        assert net.suppliers & net.consumers == frozenset()

    def test_correct_node_count(self):
        """Кількість вузлів дорівнює N."""
        net = reserved.generate(N=50, H=4, K_res=2, seed=23)
        assert net.N == 50

    def test_weakly_connected(self):
        """Граф слабко зв'язний."""
        net = reserved.generate(N=50, H=4, K_res=2, seed=24)
        assert _is_weakly_connected(net)

    def test_higher_kres_more_edges(self):
        """Більший K_res → більше ребер (але не завжди строго монотонно)."""
        net1 = reserved.generate(N=50, H=4, K_res=1, seed=25)
        net3 = reserved.generate(N=50, H=4, K_res=3, seed=25)
        assert net3.L >= net1.L, (
            f"K_res=3 ({net3.L} ребер) має бути ≥ K_res=1 ({net1.L} ребер)"
        )

    def test_supply_surplus_20_percent(self):
        """Σs_i = 1.2 · Σd_j."""
        net = reserved.generate(N=50, H=4, K_res=2, seed=26)
        total_supply = sum(net.node_supply.values())
        total_demand = sum(net.node_demand.values())
        assert abs(total_supply - 1.2 * total_demand) < 1e-6

    def test_reproducibility(self):
        """Однаковий seed → однакова мережа."""
        net1 = reserved.generate(N=50, H=4, K_res=2, seed=42)
        net2 = reserved.generate(N=50, H=4, K_res=2, seed=42)
        assert net1.L == net2.L
        assert set(net1.edges) == set(net2.edges)

    def test_c_backup_default_is_backward_compatible(self):
        """c_backup=100 має давати ті самі ребра і capacity, що default."""
        default = reserved.generate(
            N=50, H=4, b=3, K_res=3, c_base=100.0, seed=42
        )
        explicit = reserved.generate(
            N=50, H=4, b=3, K_res=3, c_base=100.0, c_backup=100.0, seed=42
        )

        assert set(default.edges) == set(explicit.edges)
        assert default._base_capacities == explicit._base_capacities
        assert default.node_supply == explicit.node_supply
        assert default.node_demand == explicit.node_demand

    def test_c_backup_changes_only_backup_edges(self):
        """R2': c_backup масштабує лише backup-ребра, trunk лишається сталим."""
        default = reserved.generate(
            N=50, H=4, b=3, K_res=3, c_base=100.0, c_backup=100.0, seed=42
        )
        high = reserved.generate(
            N=50, H=4, b=3, K_res=3, c_base=100.0, c_backup=500.0, seed=42
        )

        assert set(default.edges) == set(high.edges)
        backup_edges = [
            (u, v)
            for u, v, data in high.graph.edges(data=True)
            if data.get("edge_type") == "backup"
        ]
        trunk_edges = [
            (u, v)
            for u, v, data in high.graph.edges(data=True)
            if data.get("edge_type") == "trunk"
        ]
        assert backup_edges, "K_res=3 має створити backup-ребра"
        assert trunk_edges, "Базове дерево має trunk-ребра"

        for edge in backup_edges:
            assert high._base_capacities[edge] == 500.0
            assert default._base_capacities[edge] == 100.0
        for edge in trunk_edges:
            assert high._base_capacities[edge] == default._base_capacities[edge]


# ─────────────────────────────────────────────
# Щільна децентралізована топологія
# ─────────────────────────────────────────────

class TestDense:

    def test_correct_node_count(self):
        """Кількість вузлів дорівнює N."""
        net = dense.generate(N=50, k_avg=5, seed=30)
        assert net.N == 50

    def test_average_degree_in_range(self):
        """Середній ступінь ≈ k_avg (у межах ±3 через стохастичність)."""
        # Для великої вибірки: середній out-degree ≈ k_avg
        net = dense.generate(N=100, k_avg=5.0, seed=31)
        out_degrees = [net.graph.out_degree(n) for n in net.graph.nodes()]
        avg_out = sum(out_degrees) / len(out_degrees)
        # Очікуємо avg_out ≈ 5 з допуском ±3
        assert 2.0 <= avg_out <= 8.0, (
            f"Середній out-degree={avg_out:.2f} виходить за межі [2, 8] "
            f"при k_avg=5"
        )

    def test_weakly_connected(self):
        """Граф гарантовано слабко зв'язний."""
        for seed in range(10):
            net = dense.generate(N=50, k_avg=5, seed=seed)
            assert _is_weakly_connected(net), (
                f"Щільний граф (seed={seed}) не є слабко зв'язним"
            )

    def test_vs_vc_disjoint(self):
        """V_s ∩ V_c = ∅."""
        net = dense.generate(N=50, k_avg=5, seed=32)
        assert net.suppliers & net.consumers == frozenset()

    def test_vs_size(self):
        """|V_s| ≈ ceil(0.2·N)."""
        N = 50
        net = dense.generate(N=N, k_avg=5, seed=33)
        expected_k_s = math.ceil(0.2 * N)
        assert len(net.suppliers) == expected_k_s, (
            f"|V_s|={len(net.suppliers)}, очікувалось {expected_k_s}"
        )

    def test_vc_size(self):
        """|V_c| ≈ ceil(0.2·N)."""
        N = 50
        net = dense.generate(N=N, k_avg=5, seed=34)
        expected_k_c = math.ceil(0.2 * N)
        assert len(net.consumers) == expected_k_c, (
            f"|V_c|={len(net.consumers)}, очікувалось {expected_k_c}"
        )

    def test_supply_surplus_20_percent(self):
        """Σs_i = 1.2 · Σd_j."""
        net = dense.generate(N=50, k_avg=5, seed=35)
        total_supply = sum(net.node_supply.values())
        total_demand = sum(net.node_demand.values())
        assert abs(total_supply - 1.2 * total_demand) < 1e-6

    def test_reproducibility(self):
        """Однаковий seed → однакова мережа."""
        net1 = dense.generate(N=50, k_avg=5, seed=42)
        net2 = dense.generate(N=50, k_avg=5, seed=42)
        assert set(net1.edges) == set(net2.edges)


# ─────────────────────────────────────────────
# Фабричний метод Network.create
# ─────────────────────────────────────────────

class TestNetworkCreate:

    @pytest.mark.parametrize("topology", ["hierarchical", "hub", "reserved", "dense"])
    def test_create_all_topologies(self, topology):
        """Network.create(topology, N=50, seed=42) повертає валідну мережу."""
        kwargs = {"N": 50, "seed": 42}
        net = Network.create(topology, **kwargs)
        assert net.N == 50
        assert len(net.suppliers) > 0
        assert len(net.consumers) > 0
        assert net.suppliers & net.consumers == frozenset()

    def test_invalid_topology_raises(self):
        """Невідома топологія — ValueError з зрозумілим повідомленням."""
        with pytest.raises(ValueError, match="Невідомий тип топології"):
            Network.create("unknown_topology", N=50)


# ─────────────────────────────────────────────
# Network.copy і мутація
# ─────────────────────────────────────────────

class TestNetworkCopy:

    def test_copy_is_independent(self):
        """Зміна копії не впливає на оригінал."""
        original = hierarchical.generate(N=20, H=3, seed=0)
        copy = original.copy()
        node = next(iter(original.suppliers))
        copy.set_node_state(node, "failed")
        assert node not in original.failed_nodes, (
            "Зміна копії вплинула на оригінал"
        )

    def test_set_node_state_failed(self):
        """set_node_state('failed') обнуляє supply/demand вузла."""
        net = hierarchical.generate(N=20, H=3, seed=1)
        s = next(iter(net.suppliers))
        net.set_node_state(s, "failed")
        assert net.node_supply[s] == 0.0
        assert s in net.failed_nodes

    def test_set_node_state_active_restores(self):
        """set_node_state('active') відновлює supply/demand до базових значень."""
        net = hierarchical.generate(N=20, H=3, seed=2)
        s = next(iter(net.suppliers))
        base_supply = net._base_supply[s]
        net.set_node_state(s, "failed")
        net.set_node_state(s, "active")
        assert net.node_supply[s] == base_supply
        assert s not in net.failed_nodes

    def test_get_current_capacity_zero_for_failed_node(self):
        """get_current_capacity повертає 0 для ребра з failed-вузлом."""
        net = hierarchical.generate(N=20, H=3, V0_count=1, seed=3)
        # Знаходимо будь-яке ребро
        u, v = next(iter(net.edges))
        net.set_node_state(u, "failed")
        assert net.get_current_capacity(u, v) == 0.0

    def test_get_local_subgraph(self):
        """get_local_subgraph повертає вузли у радіусі radius."""
        net = hierarchical.generate(N=50, H=4, V0_count=1, seed=4)
        center = next(iter(net.graph.nodes()))
        sub = net.get_local_subgraph(center, radius=2)
        assert center in sub.nodes()
        # Усі вузли підграфу мають бути в радіусі 2 від центру
        undirected = net.graph.to_undirected()
        for node in sub.nodes():
            dist = nx.shortest_path_length(undirected, center, node)
            assert dist <= 2


# ─────────────────────────────────────────────
# Patch v1.2: capacity-aware topology tests
# ─────────────────────────────────────────────

class TestPatchV12CapacityPatterns:
    """Перевірка геометричних capacity і D_sat(0) ≈ 1.0 (Math Spec Patch v1.2)."""

    def test_hierarchical_geometric_capacities(self):
        """
        Capacity ребер hierarchical: c(h→h+1) = c_base * b^(H-1-h).
        Для H=4, b=3, c_base=100: [2700, 900, 300] за рівнями.
        """
        H, b, c_base = 4, 3, 100.0
        net = hierarchical.generate(N=50, H=H, b=b, c_base=c_base, seed=42)
        expected = {
            (0, 1): c_base * b ** (H - 1),   # 2700
            (1, 2): c_base * b ** (H - 2),   # 900
            (2, 3): c_base * b ** (H - 3),   # 300
        }
        for (u, v), cap in net._base_capacities.items():
            src_level = net.graph.nodes[u].get("level", -1)
            dst_level = net.graph.nodes[v].get("level", -1)
            pair = (src_level, dst_level)
            if pair in expected:
                assert cap == expected[pair], (
                    f"Ребро ({u}→{v}) рівні {pair}: capacity={cap}, "
                    f"очікувалось {expected[pair]}"
                )

    def test_hierarchical_dsat_approx_one(self):
        """D_sat(0) ∈ [0.99, 1.01] для hierarchical (seed=42, N=50, H=4)."""
        net = hierarchical.generate(N=50, H=4, b=3, c_base=100.0, seed=42)
        d_sat = compute_demand_satisfaction(net)
        assert 0.99 <= d_sat <= 1.01, (
            f"hierarchical D_sat(0)={d_sat:.4f} ∉ [0.99, 1.01]"
        )

    def test_reserved_dsat_approx_one(self):
        """D_sat(0) ∈ [0.99, 1.01] для reserved (seed=42, N=50, H=4)."""
        net = reserved.generate(N=50, H=4, b=3, K_res=2, c_base=100.0, seed=42)
        d_sat = compute_demand_satisfaction(net)
        assert 0.99 <= d_sat <= 1.01, (
            f"reserved D_sat(0)={d_sat:.4f} ∉ [0.99, 1.01]"
        )

    def test_hub_dsat_approx_one(self):
        """D_sat(0) ∈ [0.99, 1.01] для hub (seed=42, N=50, h=3)."""
        net = hub.generate(N=50, h=3, c_base=100.0, seed=42)
        d_sat = compute_demand_satisfaction(net)
        assert 0.99 <= d_sat <= 1.01, (
            f"hub D_sat(0)={d_sat:.4f} ∉ [0.99, 1.01]"
        )

    def test_dense_dsat_approx_one(self):
        """D_sat(0) ∈ [0.99, 1.01] для dense (seed=42, N=50)."""
        net = dense.generate(N=50, k_avg=5.0, c_base=100.0, seed=42)
        d_sat = compute_demand_satisfaction(net)
        assert 0.99 <= d_sat <= 1.01, (
            f"dense D_sat(0)={d_sat:.4f} ∉ [0.99, 1.01]"
        )

    def test_hub_edge_cap_scaled(self):
        """Hub↔non-hub capacity = c_base * ceil((N-h)/h)."""
        N, h, c_base = 50, 3, 100.0
        expected_cap = c_base * math.ceil((N - h) / h)
        net = hub.generate(N=N, h=h, c_base=c_base, seed=42)
        hub_ids = {n for n, d in net.graph.nodes(data=True) if d.get("role") == "hub"}
        for (u, v), cap in net._base_capacities.items():
            u_is_hub = u in hub_ids
            v_is_hub = v in hub_ids
            if u_is_hub != v_is_hub:   # рівно одна сторона — хаб
                assert cap == expected_cap, (
                    f"Hub↔non-hub ребро ({u}→{v}): capacity={cap}, "
                    f"очікувалось {expected_cap}"
                )

    def test_dense_all_edges_c_base(self):
        """Dense: усі ребра мають capacity = c_base (без масштабування)."""
        c_base = 100.0
        net = dense.generate(N=50, k_avg=5.0, c_base=c_base, seed=42)
        for (u, v), cap in net._base_capacities.items():
            assert cap == c_base, (
                f"Dense ребро ({u}→{v}): capacity={cap}, очікувалось {c_base}"
            )

    def test_reserved_tree_edges_geometric(self):
        """Reserved: ребра базового дерева мають геометричну capacity."""
        H, b, c_base = 4, 3, 100.0
        net = reserved.generate(N=50, H=H, b=b, K_res=2, c_base=c_base, seed=42)
        for (u, v), cap in net._base_capacities.items():
            src_level = net.graph.nodes[u].get("level", -1)
            dst_level = net.graph.nodes[v].get("level", -1)
            if dst_level == src_level + 1:
                expected = c_base * (b ** (H - dst_level))
                # Основні ребра дерева мають geometric capacity;
                # резервні ребра (теж adjacent рівні) — або c_base, або geometric.
                # Перевіряємо, що capacity ≥ c_base і ≤ geometric (включно).
                assert c_base <= cap <= expected + 1e-9, (
                    f"Reserved ребро ({u}→{v}) рівні {src_level}→{dst_level}: "
                    f"capacity={cap} ∉ [{c_base}, {expected}]"
                )

    @pytest.mark.parametrize("topology,kwargs", [
        ("hierarchical", {"N": 50, "H": 4, "seed": 0}),
        ("hierarchical", {"N": 50, "H": 4, "seed": 1}),
        ("reserved",     {"N": 50, "H": 4, "K_res": 2, "seed": 0}),
        ("hub",          {"N": 50, "h": 3, "seed": 0}),
        ("dense",        {"N": 50, "k_avg": 5.0, "seed": 0}),
    ])
    def test_flow_positive(self, topology, kwargs):
        """F(0) > 0 для всіх топологій (базова перевірка зв'язності)."""
        net = Network.create(topology, **kwargs)
        f0 = compute_max_flow(net)
        assert f0 > 0, f"{topology} F(0)={f0} — очікувалось > 0"

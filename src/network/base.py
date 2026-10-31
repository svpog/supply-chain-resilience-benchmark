"""
Базовий клас Network — обгортка над графом NetworkX.

Реалізує: розділи 1.1–1.3 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Зберігає стан мережі (вузли, ребра, пропозиція/попит) і підтримує
мутацію при збоях та відновленні.

Приклад використання:
    >>> from src.network.base import Network
    >>> net = Network.create("hierarchical", N=50, seed=42)
    >>> print(len(net.nodes), len(net.edges))
"""

from __future__ import annotations

import networkx as nx


def _assign_uniform_demand(network: "Network", c_base: float) -> "Network":
    """
    Uniform supply/demand assignment — Math Spec Patch v1.2 (для hub, dense).

    d_j = c_base для всіх j ∈ V_c (uniform demand)
    s_i = 1.20 * Σd_j / |V_s| для всіх i ∈ V_s (uniform supply з 20% запасом)

    Підходить для hub і dense топологій, де routing симетричний і
    D_sat(0) = 1.0 навіть при рівномірному розподілі пропозиції.
    Для деревоподібних топологій (hierarchical, reserved) використовується
    _assign_tree_demand (підрахунок піддерева без max flow).
    """
    if not network.consumers or not network.suppliers:
        return network

    d_j = c_base
    total_demand = d_j * len(network.consumers)
    s_i = 1.20 * total_demand / len(network.suppliers)

    for node in network.graph.nodes():
        sup = s_i if node in network.suppliers else 0.0
        dem = d_j if node in network.consumers else 0.0
        network.node_supply[node] = sup
        network.node_demand[node] = dem
        network._base_supply[node] = sup
        network._base_demand[node] = dem

    return network


def _assign_tree_demand(
    network: "Network",
    c_base: float,
    tree_graph: "nx.DiGraph | None" = None,
) -> "Network":
    """
    Non-uniform supply assignment для деревоподібних топологій (Patch v1.2).

    d_j = c_base для всіх j ∈ V_c.
    s_i = 1.20 * subtree_consumers_i * d_j, де subtree_consumers_i —
    кількість споживачів у піддереві постачальника i.

    МОТИВАЦІЯ
    ---------
    У hierarchical і reserved постачальники мають нерівні піддерева
    (напр. 24 vs 9 споживачів). Рівномірне supply s_i = 1.2*Σd_j/|V_s|
    < subtree_demand_i для великого постачальника → D_sat(0) < 1.
    Підрахунок піддерева (BFS O(N)) усуває цей bottleneck без max flow.

    Для reserved: tree_graph — граф базового ДЕРЕВА (до додавання
    резервних ребер), щоб правильно визначити приналежність споживачів.

    Args:
        network: мережа з уже побудованою топологією.
        c_base: базова capacity (= d_j для споживачів).
        tree_graph: граф базового дерева (для reserved); якщо None —
                    використовується network.graph (для hierarchical).
    """
    if not network.consumers or not network.suppliers:
        return network

    G_tree = tree_graph if tree_graph is not None else network.graph

    # BFS від постачальників вниз по дереву:
    # кожному вузлу призначається «власний» постачальник-предок.
    from collections import deque
    supplier_of: dict = {s: s for s in network.suppliers}
    queue: deque = deque(network.suppliers)
    while queue:
        node = queue.popleft()
        for child in G_tree.successors(node):
            if child not in supplier_of:
                supplier_of[child] = supplier_of[node]
                queue.append(child)

    subtree_consumers: dict = {s: 0 for s in network.suppliers}
    for c in network.consumers:
        root = supplier_of.get(c)
        if root is not None:
            subtree_consumers[root] += 1

    d_j = c_base
    for node in network.graph.nodes():
        if node in network.suppliers:
            sup = 1.20 * subtree_consumers.get(node, 0) * d_j
        else:
            sup = 0.0
        dem = d_j if node in network.consumers else 0.0
        network.node_supply[node] = sup
        network.node_demand[node] = dem
        network._base_supply[node] = sup
        network._base_demand[node] = dem

    return network


class Network:
    """
    Мережа поставок: орієнтований зважений граф із підмножинами V_s і V_c.

    Атрибути:
        graph: орієнтований граф NetworkX (поточна топологія).
        suppliers: V_s — фіксована множина вузлів-постачальників.
        consumers: V_c — фіксована множина вузлів-споживачів.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        suppliers: frozenset,
        consumers: frozenset,
        base_capacities: dict[tuple, float],
        node_supply: dict,
        node_demand: dict,
    ) -> None:
        self.graph = graph
        self.suppliers: frozenset = frozenset(suppliers)
        self.consumers: frozenset = frozenset(consumers)

        # Базові (незмінні) значення — потрібні для відновлення
        self._base_capacities: dict[tuple, float] = dict(base_capacities)
        self._base_supply: dict = dict(node_supply)
        self._base_demand: dict = dict(node_demand)

        # Поточні (змінні) значення — модифікуються збоями і відновленням
        self.edge_capacities: dict[tuple, float] = dict(base_capacities)
        self.node_supply: dict = dict(node_supply)
        self.node_demand: dict = dict(node_demand)

        # Стан збою
        self.failed_nodes: set = set()
        # Незворотно пошкоджені елементи (параметр w, серія S6)
        self.irreversible_nodes: set = set()
        self.irreversible_edges: set = set()
        # Мета-дані збою (наприклад, епіцентр кластерного збою для тестів)
        self.disruption_metadata: dict = {}

    # ------------------------------------------------------------------
    # Властивості зручного доступу
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list:
        """Список усіх вузлів мережі."""
        return list(self.graph.nodes())

    @property
    def edges(self) -> list:
        """Список усіх ребер мережі."""
        return list(self.graph.edges())

    @property
    def N(self) -> int:
        """Кількість вузлів."""
        return self.graph.number_of_nodes()

    @property
    def L(self) -> int:
        """Кількість ребер."""
        return self.graph.number_of_edges()

    # ------------------------------------------------------------------
    # Доступ до поточних capacity
    # ------------------------------------------------------------------

    def get_current_capacity(self, u, v) -> float:
        """
        Повертає поточну capacity ребра (u, v).

        Якщо u або v — failed-вузол, capacity = 0 (Math Spec 2.1).
        Якщо ребро деградоване — повертає знижену capacity.
        """
        if u in self.failed_nodes or v in self.failed_nodes:
            return 0.0
        return self.edge_capacities.get((u, v), 0.0)

    def get_total_base_capacity(self) -> float:
        """Сума всіх базових capacity ребер (потрібна для ΔC(t))."""
        return sum(self._base_capacities.values())

    def get_total_current_capacity(self) -> float:
        """Сума поточних capacity (з урахуванням збоїв і деградації)."""
        total = 0.0
        for (u, v), cap in self.edge_capacities.items():
            total += self.get_current_capacity(u, v)
        return total

    # ------------------------------------------------------------------
    # Мутація стану вузлів і ребер
    # ------------------------------------------------------------------

    def set_node_state(self, node, state: str) -> None:
        """
        Встановлює стан вузла: 'active' або 'failed'.

        При 'failed': пропозиція і попит обнуляються (Math Spec 2.1).
        При 'active': відновлюються до базових значень.
        """
        if state == "failed":
            self.failed_nodes.add(node)
            self.node_supply[node] = 0.0
            self.node_demand[node] = 0.0
        elif state == "active":
            self.failed_nodes.discard(node)
            self.node_supply[node] = self._base_supply.get(node, 0.0)
            self.node_demand[node] = self._base_demand.get(node, 0.0)
        else:
            raise ValueError(f"Невідомий стан вузла: '{state}'. Має бути 'active' або 'failed'.")

    def set_edge_capacity(self, u, v, capacity: float) -> None:
        """Встановлює поточну capacity ребра (для деградації, Math Spec 2.2)."""
        if (u, v) not in self._base_capacities:
            raise KeyError(f"Ребро ({u}, {v}) не існує в базовому графі.")
        self.edge_capacities[(u, v)] = max(0.0, capacity)

    def restore_edge(self, u, v) -> None:
        """Відновлює capacity ребра до базового значення (recovery, Math Spec 4.2)."""
        if (u, v) in self._base_capacities:
            self.edge_capacities[(u, v)] = self._base_capacities[(u, v)]

    # ------------------------------------------------------------------
    # Запити про стан збоїв
    # ------------------------------------------------------------------

    def get_failed_elements(self) -> dict:
        """
        Повертає словник усіх пошкоджених елементів, придатних до відновлення.

        Потрібен для recovery-стратегій (Math Spec 4.2):
            {"nodes": set, "edges": set}
        де edges — ребра з capacity < базової (не пов'язані з failed вузлами).
        Незворотно пошкоджені елементи (irreversible_*) виключаються.
        """
        repairable_nodes = self.failed_nodes - self.irreversible_nodes
        degraded_edges = {
            (u, v)
            for (u, v), cap in self.edge_capacities.items()
            if cap < self._base_capacities.get((u, v), 0.0)
            and u not in self.failed_nodes
            and v not in self.failed_nodes
            and (u, v) not in self.irreversible_edges
        }
        return {"nodes": repairable_nodes, "edges": degraded_edges}

    # ------------------------------------------------------------------
    # Hook для 2-ї статті: локальний підграф для агентів
    # ------------------------------------------------------------------

    def get_local_subgraph(self, node, radius: int) -> nx.DiGraph:
        """
        Повертає підграф у BFS-кулі радіуса radius навколо вузла.

        Цей метод не використовується у 1-й статті, але є обов'язковим
        hook-ом для мультиагентних алгоритмів 2-ї статті.
        Агенти використовують локальну інформацію для прийняття рішень.

        Args:
            node: центральний вузол BFS-кулі.
            radius: максимальна відстань (у ребрах) від центру.

        Returns:
            Підграф NetworkX DiGraph (копія).
        """
        undirected = self.graph.to_undirected()
        reachable = nx.single_source_shortest_path_length(
            undirected, node, cutoff=radius
        )
        return self.graph.subgraph(reachable.keys()).copy()

    # ------------------------------------------------------------------
    # Копіювання і фабричний метод
    # ------------------------------------------------------------------

    def copy(self) -> "Network":
        """Повертає незалежну копію мережі (для застосування збоїв без мутації оригіналу)."""
        net = Network(
            graph=self.graph.copy(),
            suppliers=self.suppliers,
            consumers=self.consumers,
            base_capacities=self._base_capacities,
            node_supply=self._base_supply,
            node_demand=self._base_demand,
        )
        net.edge_capacities = dict(self.edge_capacities)
        net.node_supply = dict(self.node_supply)
        net.node_demand = dict(self.node_demand)
        net.failed_nodes = set(self.failed_nodes)
        net.irreversible_nodes = set(self.irreversible_nodes)
        net.irreversible_edges = set(self.irreversible_edges)
        net.disruption_metadata = dict(self.disruption_metadata)
        return net

    @classmethod
    def create(cls, topology: str, **kwargs) -> "Network":
        """
        Фабричний метод: створює мережу за назвою типу топології.

        Args:
            topology: 'hierarchical' | 'hub' | 'reserved' | 'dense'
            **kwargs: параметри генератора (N, H, seed, c_base, тощо).

        Returns:
            Готова Network.

        Raises:
            ValueError: якщо topology не розпізнана.
        """
        from src.network.generators import hierarchical, hub, reserved, dense

        generators = {
            "hierarchical": hierarchical.generate,
            "hub": hub.generate,
            "reserved": reserved.generate,
            "dense": dense.generate,
        }
        if topology not in generators:
            raise ValueError(
                f"Невідомий тип топології: '{topology}'. "
                f"Доступні: {sorted(generators.keys())}"
            )
        return generators[topology](**kwargs)

    # ------------------------------------------------------------------
    # Repr для зручності налагодження
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Network(N={self.N}, L={self.L}, "
            f"|V_s|={len(self.suppliers)}, |V_c|={len(self.consumers)}, "
            f"failed={len(self.failed_nodes)})"
        )

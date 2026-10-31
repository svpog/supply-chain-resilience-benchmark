"""
Модель потоку: single-commodity max flow з super-source та super-sink.

Реалізує: розділ 3 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Ключова точка розширення: ЛИШЕ ЦЕЙ МОДУЛЬ викликає networkx.maximum_flow.
Якщо у 2-й статті буде потрібен multi-commodity solver — замінюємо
тільки цей файл, решта коду залишається незмінною (Math Spec 3.4).

Приклад використання:
    >>> from src.network.flow import compute_max_flow, compute_demand_satisfaction
    >>> net = Network.create("hierarchical", N=50, seed=42)
    >>> F0 = compute_max_flow(net)
    >>> D_sat = compute_demand_satisfaction(net)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from src.network.base import Network

# Унікальні мітки для super-source і super-sink (не перетинаються з int ID вузлів)
_SUPER_SOURCE = "__SUPER_SOURCE__"
_SUPER_SINK = "__SUPER_SINK__"


def build_extended_graph(network: "Network") -> nx.DiGraph:
    """
    Будує розширений граф G* = (V*, E*, c*) (Math Spec 3.1).

    G* включає:
        - усі вузли оригінальної мережі;
        - super-source S, з'єднаний з кожним v ∈ V_s ребром capacity = s_v;
        - super-sink T, з'єднаний з кожним v ∈ V_c ребром capacity = d_v;
        - усі ребра оригінальної мережі з поточними capacity
          (враховуються failed вузли та деградовані ребра).

    Args:
        network: стан мережі в момент t (може бути після disruption).

    Returns:
        Орієнтований граф NetworkX з атрибутом 'capacity' на ребрах.
    """
    G_ext = nx.DiGraph()

    # Копіюємо всі вузли
    G_ext.add_nodes_from(network.graph.nodes())
    G_ext.add_node(_SUPER_SOURCE)
    G_ext.add_node(_SUPER_SINK)

    # Ребра оригінальної мережі з поточними capacity
    # (get_current_capacity повертає 0 якщо вузол failed або ребро деградоване)
    for u, v in network.graph.edges():
        cap = network.get_current_capacity(u, v)
        G_ext.add_edge(u, v, capacity=cap)

    # Super-source → постачальники (capacity = поточна пропозиція)
    for s in network.suppliers:
        supply = network.node_supply[s]  # = 0 якщо вузол failed
        G_ext.add_edge(_SUPER_SOURCE, s, capacity=supply)

    # Споживачі → super-sink (capacity = поточний попит)
    for c in network.consumers:
        demand = network.node_demand[c]  # = 0 якщо вузол failed
        G_ext.add_edge(c, _SUPER_SINK, capacity=demand)

    return G_ext


def compute_max_flow(network: "Network") -> float:
    """
    Обчислює максимальний потік F(t) від super-source до super-sink (Math Spec 3.2).

    Використовує алгоритм Dinic (networkx.maximum_flow).
    Якщо шлях від super-source до super-sink відсутній — повертає 0.0.

    Args:
        network: стан мережі в момент t.

    Returns:
        F(t) — максимальний потік ≥ 0.
    """
    G_ext = build_extended_graph(network)

    try:
        flow_value, _ = nx.maximum_flow(
            G_ext, _SUPER_SOURCE, _SUPER_SINK, capacity="capacity"
        )
        return float(flow_value)
    except nx.NetworkXError:
        # Граф некоректний або немає шляху в граничному випадку
        return 0.0


def compute_node_flow_loads(network: "Network") -> dict:
    """
    Обчислює потокове навантаження вузлів для S3 centralization metrics.

    Навантаження вузла — сума потоку по всіх інцидентних оригінальних ребрах
    (без super-source/super-sink). Функція повертає лише вузли початкової
    мережі, не службові вузли.

    Returns:
        dict node -> nonnegative flow load.
    """
    G_ext = build_extended_graph(network)
    try:
        _, flow_dict = nx.maximum_flow(
            G_ext, _SUPER_SOURCE, _SUPER_SINK, capacity="capacity"
        )
    except nx.NetworkXError:
        return {node: 0.0 for node in network.graph.nodes()}

    loads = {node: 0.0 for node in network.graph.nodes()}
    for u, v in network.graph.edges():
        flow_uv = float(flow_dict.get(u, {}).get(v, 0.0))
        loads[u] += flow_uv
        loads[v] += flow_uv
    return loads


def compute_demand_satisfaction(network: "Network") -> float:
    """
    Обчислює D_sat(t) = F(t) / Σ_{j ∈ V_c} d_j (Math Spec 5.2).

    Знаменник — БАЗОВИЙ попит (до disruption), щоб D_sat відображав,
    яку частку оригінального попиту мережа задовольняє в момент t.

    Args:
        network: стан мережі в момент t.

    Returns:
        D_sat ∈ [0, 1]. Повертає 0.0 якщо базовий попит = 0.
    """
    # Базовий (оригінальний) сумарний попит
    total_base_demand = sum(network._base_demand.values())
    if total_base_demand <= 0.0:
        return 0.0

    flow = compute_max_flow(network)
    return flow / total_base_demand


def compute_Q(network: "Network", F0: float) -> float:
    """
    Обчислює Q(t) = F(t) / F(0) — нормовану продуктивність (Math Spec 5.1).

    Q(t) ∈ [0, 1]. Q(0) = 1 за визначенням.

    Args:
        network: стан мережі в момент t.
        F0: базовий потік F(0) (обчислений до disruption).

    Returns:
        Q(t). Повертає 0.0 якщо F0 = 0 (граничний випадок — прогін виключається,
        Math Spec 3.3).
    """
    if F0 <= 0.0:
        return 0.0
    return compute_max_flow(network) / F0


def compute_delta_C(network: "Network") -> float:
    """
    Обчислює ΔC(t) = (Σc_ij(0) − Σc_ij(t)) / Σc_ij(0) (Math Spec 5.2).

    Нормована сумарна втрата capacity ребер.

    Args:
        network: стан мережі в момент t.

    Returns:
        ΔC(t) ∈ [0, 1]. Повертає 0.0 якщо базова capacity = 0.
    """
    total_base = network.get_total_base_capacity()
    if total_base <= 0.0:
        return 0.0
    total_current = network.get_total_current_capacity()
    return (total_base - total_current) / total_base

"""
Структурні метрики мережі: LCC, N_cc, I_iso, APL, B_c_lost.

Реалізує: розділ 5.3 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Всі метрики обчислюються на «активному підграфі» — вузлах та ребрах,
що не є failed. Нормування — на загальну кількість вузлів N.

Приклад використання:
    >>> from src.metrics.structural import compute_structural_metrics, compute_base_bc_sum
    >>> base_bc = compute_base_bc_sum(intact_net)
    >>> metrics = compute_structural_metrics(disrupted_net, base_bc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from src.network.base import Network


# ─────────────────────────────────────────────
# Внутрішній помічник
# ─────────────────────────────────────────────

def _active_undirected(network: "Network") -> nx.Graph:
    """
    Повертає undirected підграф лише з активних (non-failed) вузлів.

    Використовується як основа для всіх структурних метрик, щоб
    коректно відображати ефект відмов вузлів на топологію мережі.
    """
    active = [n for n in network.graph.nodes() if n not in network.failed_nodes]
    return network.graph.subgraph(active).to_undirected()


# ─────────────────────────────────────────────
# Базовий розрахунок (для intact мережі, один раз)
# ─────────────────────────────────────────────

def compute_base_bc_sum(network: "Network") -> float:
    """
    Обчислює суму betweenness centrality для НЕЗМІННОЇ (intact) мережі.

    Повинна викликатися один раз перед симуляцією і передаватися
    як base_bc_sum у compute_B_c_lost() на кожному кроці t.

    Returns:
        Сума BC усіх вузлів у undirected-проєкції ≥ 0.
    """
    G_und = network.graph.to_undirected()
    bc = nx.betweenness_centrality(G_und, normalized=True)
    return sum(bc.values())


# ─────────────────────────────────────────────
# Окремі структурні метрики
# ─────────────────────────────────────────────

def compute_LCC(network: "Network") -> float:
    """
    LCC(t) = |найбільша слабко зв'язна компонента активного підграфу| / N.

    Для нерушеної зв'язної мережі LCC = 1.0.
    Для повністю пошкодженої мережі LCC = 0.0.
    """
    G = _active_undirected(network)
    if G.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)
    return largest / network.N


def compute_N_cc(network: "Network") -> int:
    """
    N_cc(t) = загальна кількість компонент (активних СКК + кожен failed вузол).

    Failed вузли рахуються як окремі ізольовані компоненти (відповідно до
    фізичного сенсу: вузол у стані 'failed' відрізаний від решти мережі).
    """
    G = _active_undirected(network)
    n_active_cc = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    return n_active_cc + len(network.failed_nodes)


def compute_I_iso(network: "Network") -> float:
    """
    I_iso(t) = частка ізольованих вузлів (failed + активних без сусідів) / N.

    Для нерушеної зв'язної мережі I_iso = 0.0.
    """
    G = _active_undirected(network)
    n_active_isolated = sum(1 for v in G.nodes() if G.degree(v) == 0)
    return (len(network.failed_nodes) + n_active_isolated) / network.N


def compute_APL(network: "Network") -> float:
    """
    APL(t) = середня довжина найкоротшого шляху у найбільшій активній компоненті.

    Якщо LCC має < 2 вузлів — повертає 0.0 (не визначено).
    """
    G = _active_undirected(network)
    if G.number_of_nodes() == 0:
        return 0.0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc_subgraph = G.subgraph(components[0])
    if lcc_subgraph.number_of_nodes() < 2:
        return 0.0
    return float(nx.average_shortest_path_length(lcc_subgraph))


def compute_B_c_lost(network: "Network", base_bc_sum: float) -> float:
    """
    B_c_lost(t) = 1 − Σbc_t / Σbc_0 ∈ [0, 1].

    Нормована втрата betweenness centrality відносно intact мережі.
    base_bc_sum обчислюється один раз через compute_base_bc_sum().

    Args:
        network: поточний стан мережі.
        base_bc_sum: сума BC для intact мережі (≥ 0).

    Returns:
        0.0 для нерушеної мережі; 1.0 якщо повністю пошкоджена.
    """
    if base_bc_sum <= 0.0:
        return 0.0
    G = _active_undirected(network)
    if G.number_of_nodes() == 0:
        return 1.0
    current_bc_sum = sum(nx.betweenness_centrality(G, normalized=True).values())
    return float(max(0.0, (base_bc_sum - current_bc_sum) / base_bc_sum))


def compute_degree_centralization(network: "Network") -> float:
    """
    Freeman degree centralization for the undirected network projection.

    S3 uses this as a structural indicator of hub concentration. The value is
    normalized to [0, 1] for undirected graphs: a star approaches 1, a regular
    graph approaches 0.
    """
    G = network.graph.to_undirected()
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    numerator = sum(max_degree - degree for degree in degrees.values())
    denominator = (n - 1) * (n - 2)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def compute_betweenness_centralization(network: "Network") -> float:
    """
    Freeman betweenness centralization for the undirected network projection.

    Uses normalized betweenness centrality. For the project scale (N≤120 in
    current protocols) exact NetworkX betweenness is sufficient.
    """
    G = network.graph.to_undirected()
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    bc = nx.betweenness_centrality(G, normalized=True)
    max_bc = max(bc.values()) if bc else 0.0
    numerator = sum(max_bc - value for value in bc.values())
    # For normalized betweenness centrality in undirected graphs, this keeps
    # the star graph close to 1.0 and regular graphs close to 0.0.
    denominator = n - 1
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


# ─────────────────────────────────────────────
# Уніфікована функція для runner.py
# ─────────────────────────────────────────────

def compute_structural_metrics(
    network: "Network",
    base_bc_sum: float | None = None,
) -> dict[str, float]:
    """
    Обчислює всі структурні метрики для стану мережі в момент t.

    Args:
        network: поточний стан мережі.
        base_bc_sum: сума BC для intact мережі. Якщо None — обчислюється
                     з поточної мережі (не рекомендується у циклі).

    Returns:
        dict з ключами: "LCC", "N_cc", "I_iso", "APL", "B_c_lost".
    """
    if base_bc_sum is None:
        base_bc_sum = compute_base_bc_sum(network)

    return {
        "LCC": compute_LCC(network),
        "N_cc": float(compute_N_cc(network)),
        "I_iso": compute_I_iso(network),
        "APL": compute_APL(network),
        "B_c_lost": compute_B_c_lost(network, base_bc_sum),
    }

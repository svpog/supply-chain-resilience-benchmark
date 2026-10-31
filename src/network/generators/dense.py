"""
Генератор щільної децентралізованої топології (Erdős–Rényi).

Реалізує: розділ 1.4, рядок «Щільна» Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Орієнтований граф Erdős–Rényi G(N, p) з p = k_avg / (N − 1).
Гарантується слабка зв'язність (якщо потрібно — додаються ребра).

V_s = ceil(0.2·N) вузлів із найвищим out-degree.
V_c = ceil(0.2·N) вузлів із V без V_s.

Приклад використання:
    >>> from src.network.generators.dense import generate
    >>> net = generate(N=50, k_avg=5, seed=42)
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from src.network.base import Network, _assign_uniform_demand


def generate(
    N: int = 50,
    k_avg: float = 5.0,
    k_s_frac: float = 0.2,
    k_c_frac: float = 0.2,
    c_base: float = 100.0,
    seed: int | None = None,
) -> Network:
    """
    Генерує щільну децентралізовану топологію (directed Erdős–Rényi).

    Args:
        N: кількість вузлів.
        k_avg: цільовий середній ступінь (in + out) вузла. Рекомендовано 4–6.
        k_s_frac: частка вузлів → V_s (за найвищим out-degree).
        k_c_frac: частка вузлів → V_c (з V без V_s, за найвищим in-degree).
        c_base: базова capacity кожного ребра.
        seed: random seed для відтворюваності.

    Returns:
        Network зі щільною децентралізованою топологією.

    Raises:
        ValueError: якщо N < 2 або k_avg ≤ 0.
    """
    if N < 2:
        raise ValueError(f"N={N} має бути ≥ 2.")
    if k_avg <= 0:
        raise ValueError(f"k_avg={k_avg} має бути > 0.")

    rng = np.random.default_rng(seed)

    # 1. Ймовірність кожного спрямованого ребра (без петель)
    # Для орієнтованого графа: очікуваний out-degree = p * (N-1)
    # Очікуваний повний ступінь (in + out) = 2 * p * (N-1)
    # Тому p = k_avg / (2 * (N-1)) — але Math Spec пише p = k_avg / (N-1)
    # Інтерпретуємо k_avg як середній out-degree → p = k_avg / (N - 1)
    p = min(k_avg / (N - 1), 1.0)

    # 2. Генерація directed ER графу
    G = nx.gnp_random_graph(N, p, directed=True, seed=int(rng.integers(0, 2**31)))

    # 3. Гарантуємо слабку зв'язність: якщо кілька компонент — з'єднуємо
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    if len(components) > 1:
        # З'єднуємо компоненти послідовно випадковими ребрами
        for i in range(1, len(components)):
            u = int(rng.choice(list(components[i - 1])))
            v = int(rng.choice(list(components[i])))
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            if not G.has_edge(v, u):
                G.add_edge(v, u)

    # 4. Встановлюємо capacity на всіх ребрах
    base_capacities: dict[tuple, float] = {}
    for u, v in G.edges():
        G[u][v]["capacity"] = c_base
        base_capacities[(u, v)] = c_base

    # 5. Вибір V_s і V_c (Math Spec 1.4)
    k_s = max(1, math.ceil(k_s_frac * N))
    k_c = max(1, math.ceil(k_c_frac * N))

    # V_s = ceil(0.2·N) вузлів із найвищим out-degree
    sorted_by_out = sorted(G.nodes(), key=lambda n: (-G.out_degree(n), n))
    suppliers = frozenset(sorted_by_out[:k_s])

    # V_c = ceil(0.2·N) вузлів із V \ V_s із найвищим in-degree
    remaining = [n for n in G.nodes() if n not in suppliers]
    sorted_remaining = sorted(remaining, key=lambda n: (-G.in_degree(n), n))
    consumers = frozenset(sorted_remaining[:k_c])

    # 6. Supply і demand (Math Spec Patch v1.2): uniform через _assign_uniform_demand
    zeros = {n: 0.0 for n in G.nodes()}
    net = Network(
        graph=G,
        suppliers=suppliers,
        consumers=consumers,
        base_capacities=base_capacities,
        node_supply=zeros,
        node_demand=zeros,
    )
    return _assign_uniform_demand(net, c_base)

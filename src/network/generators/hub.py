"""
Генератор хабової топології мережі поставок.

Реалізує: розділ 1.4, рядок «Хабова» Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

h хабових (транзитних) вузлів; non-hub вузли приєднуються до хабів
за preferential attachment (∝ deg + 1). Ребра — двоспрямовані між
non-hub і їхніми хабами; хаби з'єднані між собою.

V_s = перші ceil(0.2·N) non-hub (за спадом out-degree після побудови).
V_c = наступні ceil(0.2·N) non-hub (не з V_s).
Хаби — транзитні: supply = 0, demand = 0.

Приклад використання:
    >>> from src.network.generators.hub import generate
    >>> net = generate(N=50, h=3, seed=42)
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from src.network.base import Network, _assign_uniform_demand


def generate(
    N: int = 50,
    h: int = 3,
    k_s_frac: float = 0.2,
    k_c_frac: float = 0.2,
    c_base: float = 100.0,
    seed: int | None = None,
) -> Network:
    """
    Генерує хабову топологію мережі поставок.

    Args:
        N: загальна кількість вузлів (включно з хабами).
        h: кількість хабових (транзитних) вузлів. h < N.
        k_s_frac: частка non-hub вузлів, що стають постачальниками (≈ 0.2).
        k_c_frac: частка non-hub вузлів, що стають споживачами (≈ 0.2).
        c_base: базова capacity кожного ребра.
        seed: random seed для відтворюваності.

    Returns:
        Network із хабовою топологією, що має рівно h хабів,
        кожен non-hub зʼєднаний щонайменше з одним хабом,
        граф слабко зв'язний.

    Raises:
        ValueError: якщо параметри некоректні.
    """
    if h >= N:
        raise ValueError(f"h={h} має бути < N={N}.")
    if h < 1:
        raise ValueError("h має бути ≥ 1.")

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()

    # 1. Хабові вузли (транзитні: supply=0, demand=0)
    hub_nodes = list(range(h))
    for node in hub_nodes:
        G.add_node(node, role="hub")

    # 2. Non-hub вузли
    non_hub_nodes = list(range(h, N))
    for node in non_hub_nodes:
        G.add_node(node, role="non_hub")

    base_capacities: dict[tuple, float] = {}

    # Масштабована capacity для ребер hub ↔ non-hub (Patch v1.2):
    # кожен хаб обслуговує ≈ (N-h)/h non-hub вузлів, тому capacity
    # ребра = c_base * ceil((N-h)/h), що гарантує D_sat(0) ≈ 1.0.
    n_non_hub = N - h
    hub_edge_cap = c_base * math.ceil(n_non_hub / h)

    # 3. Preferential attachment: кожен non-hub → 1 hub (∝ deg + 1)
    hub_degree = [0] * h  # кількість вже приєднаних non-hub-вузлів
    for node in non_hub_nodes:
        weights = np.array([d + 1.0 for d in hub_degree])
        probs = weights / weights.sum()
        chosen_hub = int(rng.choice(hub_nodes, p=probs))

        # Двоспрямовані ребра: non-hub ↔ hub з масштабованою capacity
        G.add_edge(node, chosen_hub, capacity=hub_edge_cap)
        G.add_edge(chosen_hub, node, capacity=hub_edge_cap)
        base_capacities[(node, chosen_hub)] = hub_edge_cap
        base_capacities[(chosen_hub, node)] = hub_edge_cap

        hub_degree[chosen_hub] += 2

    # 4. Хаб-до-хабових з'єднань (повний кліг між хабами), capacity = c_base
    for i in range(h):
        for j in range(i + 1, h):
            G.add_edge(hub_nodes[i], hub_nodes[j], capacity=c_base)
            G.add_edge(hub_nodes[j], hub_nodes[i], capacity=c_base)
            base_capacities[(hub_nodes[i], hub_nodes[j])] = c_base
            base_capacities[(hub_nodes[j], hub_nodes[i])] = c_base

    # 5. Вибір V_s і V_c: non-hub за спадним out-degree, потім за ID
    k_s = max(1, math.ceil(k_s_frac * N))
    k_c = max(1, math.ceil(k_c_frac * N))

    # Гарантуємо, що k_s + k_c ≤ кількість non-hub
    n_non_hub = len(non_hub_nodes)
    k_s = min(k_s, n_non_hub - 1)
    k_c = min(k_c, n_non_hub - k_s)

    # Сортуємо non-hub за спадом out-degree, при рівності — за ID
    sorted_non_hub = sorted(
        non_hub_nodes,
        key=lambda n: (-G.out_degree(n), n),
    )
    suppliers = frozenset(sorted_non_hub[:k_s])
    consumers = frozenset(sorted_non_hub[k_s: k_s + k_c])

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

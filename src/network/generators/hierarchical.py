"""
Генератор ієрархічної (деревоподібної) топології мережі поставок.

Реалізує: розділ 1.4, рядок «Ієрархічна» Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Вузли розподіляються за H рівнями з геометричним коефіцієнтом b.
Ребра — лише між сусідніми рівнями. Кожен вузол λ > 0 має рівно одного
батька → N − V0_count ребер (дерево або ліс).

V_s = всі вузли рівня 0; V_c = всі вузли рівня H−1.

Приклад використання:
    >>> from src.network.generators.hierarchical import generate
    >>> net = generate(N=50, H=4, seed=42)
    >>> print(net.N, net.L)  # 50, 49 (дерево з V0_count=1)
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from src.network.base import Network, _assign_tree_demand


def _distribute_nodes(N: int, H: int, b: int, V0_count: int) -> list[int]:
    """
    Розподіляє N вузлів між H рівнями з геометричним коефіцієнтом b.

    Рівень 0 має рівно V0_count вузлів. Решта розподіляється пропорційно
    геометричному ряду b^1, b^2, ..., b^(H-1).
    """
    if H == 1:
        return [N]

    # Геометричний розподіл для рівнів 1..H-1
    raw = [b ** h for h in range(1, H)]
    raw_sum = sum(raw)
    remaining = N - V0_count

    counts = [V0_count]
    for i, r in enumerate(raw):
        if i < len(raw) - 1:
            cnt = max(1, round(remaining * r / raw_sum))
        else:
            # Останній рівень добирає решту
            cnt = remaining - sum(counts[1:])
        counts.append(max(1, cnt))

    # Підганяємо суму до N (через округлення може відрізнятись)
    diff = N - sum(counts)
    counts[-1] = max(1, counts[-1] + diff)

    # Якщо все ще не рівно — ітеруємо корекцію
    while sum(counts) != N:
        diff = N - sum(counts)
        if diff > 0:
            counts[-1] += 1
        else:
            # Зменшуємо найбільший рівень (не перший)
            idx = max(range(1, H), key=lambda i: counts[i])
            if counts[idx] > 1:
                counts[idx] -= 1
            else:
                counts[-1] -= 1

    return counts


def generate(
    N: int = 50,
    H: int = 4,
    b: int = 3,
    V0_count: int = 2,
    c_base: float = 100.0,
    seed: int | None = None,
    _assign_demand: bool = True,
) -> Network:
    """
    Генерує ієрархічну (деревоподібну) топологію мережі поставок.

    Args:
        N: загальна кількість вузлів.
           Необхідна умова коректної ієрархії (Math Spec 1.4):
               ∀h ∈ {1,...,H−1}: nodes_per_level[h] ≥ ⌈nodes_per_level[h−1] / b⌉
           Тобто кожен батьківський вузол має шанс отримати хоча б одну дитину.
           Якщо умова порушена хоча б для одного h → ValueError.
           При дефолтних H=4, b=3, V0_count=2 безпечне N ≥ 30.
        H: кількість ієрархічних рівнів.
        b: branching factor (кількість дітей на вузол у «ідеальному» дереві).
        V0_count: кількість вузлів на рівні 0 (постачальники).
                  При V0_count=1 граф — дерево з N−1 ребрами.
        c_base: базова capacity кожного ребра.
        seed: random seed для відтворюваності.

    Returns:
        Network з ієрархічною топологією.

    Raises:
        ValueError: якщо параметри некоректні або умова ієрархії порушена.
    """
    if N < H:
        raise ValueError(f"N={N} має бути ≥ H={H}.")
    if V0_count >= N:
        raise ValueError(f"V0_count={V0_count} має бути < N={N}.")

    rng = np.random.default_rng(seed)

    # 1. Розподіл вузлів між рівнями
    nodes_per_level = _distribute_nodes(N, H, b, V0_count)

    # Перевірка коректності ієрархії для всіх суміжних рівнів (Math Spec 1.4):
    #   ∀h ∈ {1,...,H−1}: nodes_per_level[h] ≥ ceil(nodes_per_level[h−1] / b)
    # ceil гарантує, що навіть при непарному розподілі кожен батьківський вузол
    # має шанс отримати дитину. Порушення → деякі батьки залишаться ізольованими.
    for h in range(1, H):
        threshold = math.ceil(nodes_per_level[h - 1] / b)
        if nodes_per_level[h] < threshold:
            raise ValueError(
                f"N={N} порушує умову ієрархії на рівні h={h}: "
                f"nodes_per_level[{h}]={nodes_per_level[h]} < "
                f"ceil(nodes_per_level[{h-1}]/b) = ceil({nodes_per_level[h-1]}/{b}) = {threshold}. "
                f"Кожен батьківський вузол має шанс отримати дитину "
                f"(∀h: nodes[h] ≥ ceil(nodes[h−1] / b)). "
                f"Збільшіть N або зменшіть H, b чи V0_count."
            )

    # 2. Будуємо граф і присвоюємо вузли до рівнів
    G = nx.DiGraph()
    level_nodes: list[list[int]] = []
    node_id = 0

    for level, count in enumerate(nodes_per_level):
        level_group = list(range(node_id, node_id + count))
        for n in level_group:
            G.add_node(n, level=level)
        level_nodes.append(level_group)
        node_id += count

    # 3. Кожен вузол λ > 0 отримує рівно одного батька з рівня λ−1.
    # Гарантія: кожен батьківський вузол отримує щонайменше одного нащадка
    # (Math Spec 11.1: «всі вузли λ < H−1 мають дітей»).
    #
    # Геометрична capacity (Patch v1.2): c(level-1 → level) = c_base * b^(H-level)
    # Забезпечує F(0) ≈ Σd_j без flow-aware корекцій: кожен рівень має
    # достатню «пропускну ширину» для всього трафіку нижчих рівнів.
    base_capacities: dict[tuple, float] = {}
    for level in range(1, H):
        parents = level_nodes[level - 1]
        children = list(level_nodes[level])
        edge_capacity = c_base * (b ** (H - level))

        # Спочатку перемішуємо дітей і призначаємо кожному батьку по одній дитині
        # (щоб кожен батько гарантовано мав хоча б одного нащадка)
        rng.shuffle(children)
        parent_of: dict[int, int] = {}

        if len(children) >= len(parents):
            # Перший прохід: round-robin — кожен батько отримує одну дитину
            for i, parent in enumerate(parents):
                child = children[i]
                parent_of[child] = parent

            # Другий прохід: решта дітей — випадковий батько
            for child in children[len(parents):]:
                parent = int(rng.choice(parents))
                parent_of[child] = parent
        else:
            # Дітей менше, ніж батьків — кожна дитина випадково вибирає батька
            # (деякі батьки залишаться без дітей у цьому граничному випадку)
            for child in children:
                parent = int(rng.choice(parents))
                parent_of[child] = parent

        for child, parent in parent_of.items():
            G.add_edge(parent, child, capacity=edge_capacity)
            base_capacities[(parent, child)] = edge_capacity

    # 4. V_s = рівень 0; V_c = рівень H−1
    suppliers = frozenset(level_nodes[0])
    consumers = frozenset(level_nodes[-1])

    zeros = {n: 0.0 for n in G.nodes()}
    net = Network(
        graph=G,
        suppliers=suppliers,
        consumers=consumers,
        base_capacities=base_capacities,
        node_supply=zeros,
        node_demand=zeros,
    )
    # _assign_demand=False використовується reserved.py при внутрішньому
    # виклику: demand виставить reserved.generate() на фінальному графі.
    return _assign_tree_demand(net, c_base) if _assign_demand else net

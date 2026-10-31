"""
Генератор резервованої багаторівневої топології мережі поставок.

Реалізує: розділ 1.4, рядок «Резервована» Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Починається з ієрархічного дерева, потім додаються K_res×(H-1) додаткових
«резервних» ребер між вузлами сусідніх рівнів і K_res shortcut-ребер
між рівнями через один.

V_s = рівень 0; V_c = рівень H−1.

Приклад використання:
    >>> from src.network.generators.reserved import generate
    >>> net = generate(N=50, H=4, K_res=2, seed=42)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.network.base import Network, _assign_tree_demand
from src.network.generators.hierarchical import generate as _hier_generate


def generate(
    N: int = 50,
    H: int = 4,
    b: int = 3,
    V0_count: int = 2,
    K_res: int = 2,
    c_base: float = 100.0,
    c_backup: float | None = None,
    seed: int | None = None,
) -> Network:
    """
    Генерує резервовану багаторівневу топологію мережі поставок.

    Args:
        N: загальна кількість вузлів.
        H: кількість ієрархічних рівнів.
        b: branching factor базового дерева.
        V0_count: кількість вузлів на рівні 0.
        K_res: кількість додаткових резервних ребер між кожною парою
               сусідніх рівнів (загалом ≈ K_res × (H-1) + K_res shortcuts).
        c_base: базова capacity кожного ребра.
        c_backup: capacity резервних ребер. Якщо None, використовується
                  c_base — повна зворотна сумісність із R2/статтею 1.
        seed: random seed для відтворюваності.

    Returns:
        Network із резервованою топологією: більше ребер, ніж у дереві.
    """
    rng = np.random.default_rng(seed)
    backup_capacity = c_base if c_backup is None else float(c_backup)

    # 1. Будуємо базове ієрархічне дерево з геометричними capacity,
    #    але без встановлення demand (_assign_demand=False):
    #    _assign_uniform_demand буде викликана нижче вже на фінальному
    #    reserved-графі з додатковими ребрами (резервними і shortcut).
    base_net = _hier_generate(
        N=N, H=H, b=b, V0_count=V0_count,
        c_base=c_base, seed=int(rng.integers(0, 2**31)),
        _assign_demand=False,
    )
    G = base_net.graph.copy()
    for u, v in G.edges():
        G.edges[u, v]["edge_type"] = "trunk"
    # Зберігаємо копію БАЗОВОГО ДЕРЕВА для підрахунку піддерев постачальників.
    # Резервні ребра не мають впливати на визначення «власного» постачальника
    # кожного споживача — воно ґрунтується на ієрархічній структурі.
    base_tree_graph = base_net.graph.copy()
    base_capacities: dict[tuple, float] = dict(base_net._base_capacities)
    suppliers = base_net.suppliers
    consumers = base_net.consumers

    # Відновлюємо інформацію про рівні з атрибутів вузлів
    level_nodes: list[list[int]] = [[] for _ in range(H)]
    for node, data in G.nodes(data=True):
        lvl = data.get("level", 0)
        level_nodes[lvl].append(node)

    # 2. Додаємо K_res × (H-1) резервних ребер між сусідніми рівнями
    # (альтернативні шляхи, тобто додаткові батьківські з'єднання)
    for level in range(1, H):
        parents = level_nodes[level - 1]
        children = level_nodes[level]
        added = 0
        # Перемішуємо можливі пари (parent, child) без наявного ребра
        candidates = [
            (p, c) for p in parents for c in children if not G.has_edge(p, c)
        ]
        rng.shuffle(candidates)
        for p, c in candidates:
            if added >= K_res:
                break
            G.add_edge(p, c, capacity=backup_capacity, edge_type="backup")
            base_capacities[(p, c)] = backup_capacity
            added += 1

    # 3. Додаємо K_res shortcut-ребер через рівень (λ → λ+2)
    for level in range(0, H - 2):
        sources = level_nodes[level]
        targets = level_nodes[level + 2]
        candidates = [
            (s, t) for s in sources for t in targets if not G.has_edge(s, t)
        ]
        rng.shuffle(candidates)
        for i, (s, t) in enumerate(candidates):
            if i >= K_res:
                break
            G.add_edge(s, t, capacity=backup_capacity, edge_type="backup")
            base_capacities[(s, t)] = backup_capacity

    # 4. Supply і demand (Patch v1.2): non-uniform supply на основі підрахунку
    #    піддерев у BASE TREE (без резервних ребер) → D_sat(0) ≈ 1.0
    zeros = {n: 0.0 for n in G.nodes()}
    net = Network(
        graph=G,
        suppliers=suppliers,
        consumers=consumers,
        base_capacities=base_capacities,
        node_supply=zeros,
        node_demand=zeros,
    )
    return _assign_tree_demand(net, c_base, tree_graph=base_tree_graph)

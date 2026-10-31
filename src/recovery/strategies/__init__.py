"""
Реєстр recovery-стратегій для статті 3.

Реалізує: розділ 6.2 Math Spec Article 3
(див. docs/Math_Spec_Article_3.docx).

Приклад використання:
    >>> from src.recovery.strategies import get_recovery_strategy
    >>> strategy = get_recovery_strategy("flow_greedy")
"""

from __future__ import annotations

from collections.abc import Callable

from src.recovery.base import RecoveryStrategy
from src.recovery.strategies.centrality import CentralityStrategy
from src.recovery.strategies.flow_greedy import FlowGreedyStrategy
from src.recovery.strategies.flow_greedy_local import FlowGreedyLocalStrategy
from src.recovery.strategies.random import RandomStrategy


STRATEGY_REGISTRY: dict[str, Callable[[], RecoveryStrategy]] = {
    "random": RandomStrategy,
    "centrality_degree": lambda: CentralityStrategy("degree"),
    "centrality_betweenness": lambda: CentralityStrategy("betweenness"),
    "centrality_pagerank": lambda: CentralityStrategy("pagerank"),
    "flow_greedy": FlowGreedyStrategy,
    "flow_greedy_local_r0": lambda: FlowGreedyLocalStrategy(r=0),
    "flow_greedy_local_r1": lambda: FlowGreedyLocalStrategy(r=1),
    "flow_greedy_local_r2": lambda: FlowGreedyLocalStrategy(r=2),
    "flow_greedy_local_r3": lambda: FlowGreedyLocalStrategy(r=3),
    "flow_greedy_local_inf": lambda: FlowGreedyLocalStrategy(r=None),
}


def get_recovery_strategy(name: str) -> RecoveryStrategy:
    """
    Створює recovery-стратегію за назвою з конфігу.

    Args:
        name: random | centrality_degree | centrality_betweenness |
              centrality_pagerank | flow_greedy.

    Returns:
        Екземпляр RecoveryStrategy.
    """
    try:
        factory = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Невідома recovery strategy '{name}'. Дозволені: {allowed}.") from exc
    return factory()

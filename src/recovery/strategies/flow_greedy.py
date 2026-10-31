"""
Flow-greedy recovery strategy для статті 3.

Реалізує: розділ 3.5 Math Spec Article 3
(див. docs/Math_Spec_Article_3.docx).

На кожному кроці стратегія тимчасово відновлює кожен кандидатний елемент,
рахує приріст max-flow ΔF і обирає елемент з найбільшим приростом.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from src.network.flow import compute_max_flow
from src.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from src.network.base import Network


class FlowGreedyStrategy(RecoveryStrategy):
    """Жадібна стратегія за приростом max-flow."""

    def _restore_on_copy(self, network: "Network", element: tuple) -> "Network":
        candidate = network.copy()
        kind, target = element
        if kind == "node":
            candidate.set_node_state(target, "active")
        elif kind == "edge":
            u, v = target
            candidate.restore_edge(u, v)
        else:
            raise ValueError(f"Unknown failed element kind: {kind}")
        return candidate

    def _rank_with_seeded_ties(
        self,
        gains: dict[tuple, float],
        rng: np.random.Generator,
    ) -> list[tuple]:
        groups: dict[float, list[tuple]] = defaultdict(list)
        for element, gain in gains.items():
            groups[float(gain)].append(element)

        ranked: list[tuple] = []
        for gain in sorted(groups.keys(), reverse=True):
            group = sorted(groups[gain], key=repr)
            if len(group) > 1:
                order = rng.permutation(len(group))
                group = [group[int(i)] for i in order]
            ranked.extend(group)
        return ranked

    def select_element_to_repair(
        self,
        network: "Network",
        t: int,
        rng: np.random.Generator | None = None,
    ) -> tuple | None:
        """Обирає елемент із максимальним Δmax_flow після відновлення."""
        if rng is None:
            rng = np.random.default_rng()

        failed = network.get_failed_elements()
        elements = [("node", node) for node in sorted(failed["nodes"])]
        elements.extend(("edge", edge) for edge in sorted(failed["edges"]))

        if not elements:
            return None
        if len(elements) == 1:
            return elements[0]

        current_flow = compute_max_flow(network)
        gains: dict[tuple, float] = {}
        for element in elements:
            restored = self._restore_on_copy(network, element)
            gains[element] = compute_max_flow(restored) - current_flow

        return self._rank_with_seeded_ties(gains, rng)[0]

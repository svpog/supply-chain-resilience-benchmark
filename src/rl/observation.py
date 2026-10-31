"""
Set-encoder спостереження для M5-ceiling RL.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §4.3
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    >>> encoder = SetObservationEncoder(max_candidates=128)
    >>> vector, candidates = encoder.encode(agent, observation, locked=set(), F0=100.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SetObservationEncoder:
    """
    Фіксований DeepSets-style encoder для видимих failed-кандидатів.

    Для сумісності з SB3 PPO змінна множина кандидатів пакується у
    фіксований масив max_candidates. Симетричний mean-pool додається як
    глобальний контекст, а slot-mask відмічає доступні дії. Це зберігає
    set-семантику без прив'язки політики до конкретного node_id.
    """

    max_candidates: int = 128
    feature_dim: int = 6
    global_dim: int = 4

    @property
    def obs_dim(self) -> int:
        """Розмір flattened observation vector."""
        return self.max_candidates * (self.feature_dim + 1) + self.global_dim

    def encode(self, agent, observation: dict, locked: set, F0: float) -> tuple[np.ndarray, list]:
        """Кодує observation у фіксований float32 vector і список candidate nodes."""
        network = observation["network"]
        G_sub = observation["G_sub"].to_undirected()
        locked = set(locked)
        candidates = sorted(set(observation["visible_failed"]) - locked)[: self.max_candidates]

        features = np.zeros((self.max_candidates, self.feature_dim), dtype=np.float32)
        mask = np.zeros((self.max_candidates, 1), dtype=np.float32)
        H = max(1.0, max(
            float(data.get("level", 0.0))
            for _, data in network.graph.nodes(data=True)
        ))
        N = max(1.0, float(network.N - 1))
        active_suppliers = [node for node in network.suppliers if node not in network.failed_nodes and node in G_sub]
        active_consumers = [node for node in network.consumers if node not in network.failed_nodes and node in G_sub]
        dist_to_supplier = _multi_source_distances(G_sub, active_suppliers)
        dist_to_consumer = _multi_source_distances(G_sub, active_consumers)

        for idx, node in enumerate(candidates):
            delta = float(agent.estimate_gain(node, observation)) / max(F0, 1.0)
            level = float(network.graph.nodes[node].get("level", 0.0)) / H
            degree = float(G_sub.degree(node)) / N if node in G_sub else 0.0
            d_sup = _normalized_distance(dist_to_supplier.get(node), N)
            d_con = _normalized_distance(dist_to_consumer.get(node), N)
            features[idx] = np.asarray([delta, level, degree, 0.0, d_sup, d_con], dtype=np.float32)
            mask[idx, 0] = 1.0

        if candidates:
            pooled = features[: len(candidates)].mean(axis=0)
            global_features = np.asarray(
                [
                    float(len(candidates)) / max(1.0, float(self.max_candidates)),
                    float(len(network.failed_nodes)) / max(1.0, float(network.N)),
                    float(mask.sum()) / max(1.0, float(self.max_candidates)),
                    float(pooled[0]),
                ],
                dtype=np.float32,
            )
        else:
            global_features = np.zeros(self.global_dim, dtype=np.float32)

        encoded = np.concatenate([features.reshape(-1), mask.reshape(-1), global_features])
        return encoded.astype(np.float32), candidates


def _multi_source_distances(G: nx.Graph, sources: list) -> dict:
    """Shortest-path distances from many sources, empty-safe."""
    if not sources:
        return {}
    return nx.multi_source_dijkstra_path_length(G, sources)


def _normalized_distance(distance: float | None, N: float) -> float:
    """Нормалізує distance; недосяжні вузли отримують 1.0."""
    if distance is None:
        return 1.0
    return min(1.0, float(distance) / max(1.0, N))

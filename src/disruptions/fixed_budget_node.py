"""
Fixed-budget node disruptions for S3 targeted hub experiments.

Реалізує: протокол TZ_S3_targeted_hub_disruptions.docx, розділи 4.1 та 8.

На відміну від Bernoulli node_failure, цей сценарій пошкоджує рівно
k_attack вузлів із множини кандидатів C_attack. Це дозволяє чесно
порівнювати random_fixed_k і targeted режими при однаковому бюджеті атаки.

Приклад використання:
    >>> from src.disruptions.fixed_budget_node import FixedBudgetNodeDisruption
    >>> d = FixedBudgetNodeDisruption(attack_fraction=0.10, mode="targeted")
    >>> disrupted = d.apply(network, seed=42)
"""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np

from src.disruptions.base import Disruption
from src.network.base import Network
from src.network.flow import compute_max_flow


class FixedBudgetNodeDisruption(Disruption):
    """
    Fixed-budget node disruption for S3.

    Args:
        attack_fraction: q ∈ (0, 1]; k_attack = ceil(q·|C_attack|).
        k_attack: optional explicit number of nodes to fail. If set, it has
            priority over attack_fraction.
        mode: "random_fixed_k" or "targeted".
        attack_rule: scoring rule for targeted mode:
            "betweenness", "degree", "hub_role", or "flow_impact".
        candidate_set: currently "transit_only" = V \\ (V_s ∪ V_c) or "all".
        w: irreversible fraction, inherited from Disruption.
    """

    VALID_MODES = {"random_fixed_k", "targeted"}
    VALID_RULES = {"betweenness", "degree", "hub_role", "flow_impact"}
    VALID_CANDIDATE_SETS = {"transit_only", "all"}

    def __init__(
        self,
        attack_fraction: float | None = None,
        k_attack: int | None = None,
        mode: str = "random_fixed_k",
        attack_rule: str = "betweenness",
        candidate_set: str = "transit_only",
        w: float = 0.0,
    ) -> None:
        super().__init__(w=w)
        if attack_fraction is None and k_attack is None:
            raise ValueError("Потрібно задати attack_fraction або k_attack.")
        if attack_fraction is not None and not (0.0 < attack_fraction <= 1.0):
            raise ValueError("attack_fraction має бути у (0, 1].")
        if k_attack is not None and k_attack < 1:
            raise ValueError("k_attack має бути >= 1.")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode='{mode}' має бути одним із {sorted(self.VALID_MODES)}.")
        if attack_rule not in self.VALID_RULES:
            raise ValueError(
                f"attack_rule='{attack_rule}' має бути одним із {sorted(self.VALID_RULES)}."
            )
        if candidate_set not in self.VALID_CANDIDATE_SETS:
            raise ValueError(
                f"candidate_set='{candidate_set}' має бути одним із "
                f"{sorted(self.VALID_CANDIDATE_SETS)}."
            )
        self.attack_fraction = attack_fraction
        self.k_attack = k_attack
        self.mode = mode
        self.attack_rule = attack_rule
        self.candidate_set = candidate_set

    def apply(self, network: Network, seed: int | None = None) -> Network:
        """Applies fixed-budget node failure and returns a disrupted copy."""
        rng = np.random.default_rng(seed)
        net = network.copy()
        candidates = _candidate_nodes(net, self.candidate_set)
        if not candidates:
            net.disruption_metadata.update({
                "mode": self.mode,
                "attack_rule": self.attack_rule,
                "candidate_set": self.candidate_set,
                "candidate_set_size": 0,
                "k_attack": 0,
                "targeted_nodes": [],
                "failed_hubs_count": 0,
            })
            return net

        k_attack = self._resolve_k(len(candidates))
        if self.mode == "random_fixed_k":
            selected = _sample_nodes(candidates, k_attack, rng)
        else:
            scores = _score_candidates(net, candidates, self.attack_rule)
            selected = _top_k_by_score(candidates, scores, k_attack)

        for node in selected:
            net.set_node_state(node, "failed")

        if self.w > 0.0 and selected:
            n_irreversible = max(1, round(self.w * len(selected)))
            shuffled = list(selected)
            rng.shuffle(shuffled)
            for node in shuffled[:n_irreversible]:
                net.irreversible_nodes.add(node)

        failed_hubs = sum(1 for node in selected if net.graph.nodes[node].get("role") == "hub")
        net.disruption_metadata.update({
            "mode": self.mode,
            "attack_rule": self.attack_rule,
            "candidate_set": self.candidate_set,
            "candidate_set_size": len(candidates),
            "k_attack": k_attack,
            "attack_fraction": self.attack_fraction,
            "targeted_nodes": list(selected),
            "failed_nodes_count": len(selected),
            "failed_hubs_count": failed_hubs,
        })
        return net

    def _resolve_k(self, candidate_count: int) -> int:
        """Computes k_attack from explicit k or attack_fraction."""
        if self.k_attack is not None:
            return min(self.k_attack, candidate_count)
        assert self.attack_fraction is not None
        return max(1, min(candidate_count, math.ceil(self.attack_fraction * candidate_count)))


def _candidate_nodes(network: Network, candidate_set: str) -> list:
    """Returns candidate nodes for attack."""
    if candidate_set == "all":
        return sorted(network.graph.nodes())
    return sorted(
        node for node in network.graph.nodes()
        if node not in network.suppliers and node not in network.consumers
    )


def _sample_nodes(candidates: list, k_attack: int, rng: np.random.Generator) -> list:
    """Uniformly samples exactly k_attack candidate nodes."""
    selected = rng.choice(np.array(candidates, dtype=object), size=k_attack, replace=False)
    return sorted(selected.tolist())


def _top_k_by_score(candidates: list, scores: dict, k_attack: int) -> list:
    """Returns top-k candidates by decreasing score and deterministic tie-break."""
    return sorted(candidates, key=lambda n: (-scores.get(n, 0.0), n))[:k_attack]


def _score_candidates(network: Network, candidates: Iterable, attack_rule: str) -> dict:
    """Computes targeted attack scores for candidate nodes."""
    if attack_rule == "betweenness":
        graph = network.graph.to_undirected()
        scores = nx.betweenness_centrality(graph, normalized=True)
        return {node: float(scores.get(node, 0.0)) for node in candidates}

    if attack_rule == "degree":
        graph = network.graph.to_undirected()
        return {node: float(graph.degree(node)) for node in candidates}

    if attack_rule == "hub_role":
        graph = network.graph.to_undirected()
        return {
            node: (1.0 if network.graph.nodes[node].get("role") == "hub" else 0.0)
            + 1e-6 * float(graph.degree(node))
            for node in candidates
        }

    if attack_rule == "flow_impact":
        return _flow_impact_scores(network, candidates)

    raise ValueError(f"Невідоме attack_rule='{attack_rule}'.")


def _flow_impact_scores(network: Network, candidates: Iterable) -> dict:
    """
    Scores nodes by ΔF_v = F(G) - F(G\\v).

    Uses compute_max_flow() so the NetworkX max-flow call remains isolated
    inside src.network.flow, as required by the project architecture.
    """
    F0 = compute_max_flow(network)
    scores = {}
    for node in candidates:
        net_copy = network.copy()
        net_copy.set_node_state(node, "failed")
        scores[node] = max(0.0, F0 - compute_max_flow(net_copy))
    return scores

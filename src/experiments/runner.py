"""
Runner серій експериментів: читання конфігу та запуск Monte Carlo.

Реалізує: розділ 10 Math Spec (output schema) + протоколи серій S1–S6
(див. docs/Math_Spec_Стаття_1.docx та протоколи_серій_та_метрики_перша_стаття_v3.xlsx).

Приклад використання:
    >>> from src.experiments.runner import run_series
    >>> run_series(series="S1", n_runs=5)
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from src.disruptions.cluster import ClusterDisruption
from src.disruptions.edge_degradation import EdgeDegradationDisruption
from src.disruptions.fixed_budget_node import FixedBudgetNodeDisruption
from src.disruptions.node_failure import NodeFailureDisruption
from src.metrics.integral import compute_integral_metrics
from src.metrics.stochastic import compute_stochastic_metrics
from src.metrics.structural import (
    compute_base_bc_sum,
    compute_betweenness_centralization,
    compute_degree_centralization,
    compute_structural_metrics,
)
from src.network.base import Network
from src.network.flow import compute_max_flow, compute_node_flow_loads
from src.recovery.strategies import get_recovery_strategy


# ── Параметри інтенсивності збою для кожного типу ──────────────────────────
# Використовується для дедуплікації конфігурацій у S4 (різні типи збою
# мають різні релевантні параметри інтенсивності).
_DISRUPTION_INTENSITY: dict[str, frozenset] = {
    "node_failure":     frozenset({"p_v"}),
    "node_failure_fixed_budget": frozenset({"attack_fraction", "k_attack"}),
    "edge_degradation": frozenset({"p_e", "gamma", "target"}),
    "cluster":          frozenset({"r_clust", "target_fraction"}),
}
_ALL_INTENSITY_PARAMS: frozenset = frozenset({
    "p_v", "p_e", "gamma", "r_clust", "target_fraction",
    "attack_fraction", "k_attack", "target",
})
_ARTICLE4_SERIES: frozenset = frozenset({"M0"})


def _s5_m_cap(N: int) -> int:
    """Ліміт M для S5 залежно від N. N≤120 → M=100; N>120 → M=50."""
    return 50 if N > 120 else 100


# ════════════════════════════════════════════════════════════════════════════
# Завантаження конфігурації
# ════════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивне злиття: override перекриває base поелементно."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(series: str, config_dir: str = "config") -> dict:
    """
    Завантажує base_config.yaml і series_SX.yaml, об'єднує їх.

    series_config перекриває base_config. Секція sweep виноситься окремо.
    """
    base_path = Path(config_dir) / "base_config.yaml"
    series_path = Path(config_dir) / f"series_{series}.yaml"

    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(series_path, encoding="utf-8") as f:
        series_cfg = yaml.safe_load(f) or {}

    sweep = series_cfg.pop("sweep", {})
    merged = _deep_merge(base_cfg, series_cfg)
    merged["sweep"] = sweep
    return merged


# ════════════════════════════════════════════════════════════════════════════
# Побудова sweep-комбінацій
# ════════════════════════════════════════════════════════════════════════════

def _build_sweep_combinations(sweep: dict) -> list[dict]:
    """
    Декартовий добуток параметрів sweep.

    Значення-списки перебираються; не-списки (dict або скаляр) —
    фіксоване одне значення.
    """
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [v if isinstance(v, list) else [v] for v in sweep.values()]
    return [dict(zip(keys, prod)) for prod in itertools.product(*values)]


def _effective_combo(combo: dict) -> dict:
    """
    Нормалізований combo: видаляє нерелевантні параметри інтенсивності
    залежно від disruption_type. Потрібно для дедуплікації конфігурацій
    у S4 (де sweep містить p_v, p_e, gamma, r_clust одночасно).
    """
    d_type = combo.get("disruption_type")
    if d_type is None:
        effective = dict(combo)
        if effective.get("disruption_mode") == "random_fixed_k":
            effective.pop("attack_rule", None)
        return effective
    relevant = _DISRUPTION_INTENSITY.get(d_type, frozenset())
    effective = {
        k: v for k, v in combo.items()
        if k not in _ALL_INTENSITY_PARAMS or k in relevant
    }
    # S3 fair-comparison design: random_fixed_k does not use attack_rule.
    # Dropping it avoids duplicated random configurations when extended
    # targeted rules are added later.
    if effective.get("disruption_mode") == "random_fixed_k":
        effective.pop("attack_rule", None)
    return effective


def _config_id(effective: dict) -> str:
    """Коротка хеш-мітка конфігурації (8 символів MD5)."""
    key = json.dumps(effective, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _strategy_r_visibility(strategy_name: str) -> int | str | None:
    """Extracts r visibility label from flow_greedy_local strategy names."""
    mapping: dict[str, int | str] = {
        "flow_greedy_local_r0": 0,
        "flow_greedy_local_r1": 1,
        "flow_greedy_local_r2": 2,
        "flow_greedy_local_r3": 3,
        "flow_greedy_local_inf": "inf",
    }
    return mapping.get(strategy_name)


# ════════════════════════════════════════════════════════════════════════════
# Застосування combo до конфігу
# ════════════════════════════════════════════════════════════════════════════

def _apply_combo(cfg: dict, combo: dict) -> dict:
    """
    Застосовує sweep-combo до об'єднаного конфіг-словника.

    Маршрутизує ключі у відповідні секції:
        network / disruption / recovery / simulation / metrics.
    """
    c = copy.deepcopy(cfg)
    for k, v in combo.items():
        if k == "topology":
            c["network"]["topology"] = v
        elif k == "disruption_type":
            c["disruption"]["type"] = v
        elif k == "disruption_mode":
            c["disruption"]["mode"] = v
        elif k in _ALL_INTENSITY_PARAMS:
            c["disruption"][k] = v
        elif k == "disruption" and isinstance(v, dict):
            # S3: фіксований під-словник disruption у sweep (не список)
            c["disruption"].update(v)
        elif k == "K_res":
            c["network"]["K_res"] = v
        elif k == "c_backup":
            c["network"]["c_backup"] = v
        elif k in {"h", "h_hubs"}:
            c["network"]["h"] = v
        elif k == "N":
            c["network"]["N"] = v
        elif k == "w":
            c["recovery"]["w"] = v
        elif k == "recovery_strategy":
            c["recovery"]["strategy"] = v
        elif k == "mu":
            c["recovery"]["mu"] = v
        elif k in {"attack_fraction", "k_attack", "attack_rule", "candidate_set"}:
            c["disruption"][k] = v
        elif k == "T_max":
            c["simulation"]["T_max"] = v
        elif k == "alpha_cvar":
            c["metrics"]["alpha_cvar"] = v
        elif k == "omega":
            c["simulation"]["omega"] = v
    return c


# ════════════════════════════════════════════════════════════════════════════
# Побудова мережі
# ════════════════════════════════════════════════════════════════════════════

def _create_network(cfg: dict, seed: int) -> Network:
    """Будує мережу за секцією network конфігу."""
    net_cfg = cfg["network"]
    topology = net_cfg.get("topology", "hierarchical")
    N = int(net_cfg.get("N", 50))
    H = int(net_cfg.get("H", 4))
    c_base = float(net_cfg.get("c_base", 100.0))

    kwargs: dict[str, Any] = {"N": N, "c_base": c_base, "seed": seed}

    if topology in {"hierarchical", "reserved"}:
        kwargs["H"] = H
        kwargs["b"] = int(net_cfg.get("b", 3))
    if topology == "reserved":
        kwargs["K_res"] = int(net_cfg.get("K_res", 2))
        if "c_backup" in net_cfg and net_cfg.get("c_backup") is not None:
            kwargs["c_backup"] = float(net_cfg.get("c_backup"))
    if topology == "hub":
        kwargs["h"] = int(net_cfg.get("h", 3))
    if topology == "dense":
        kwargs["k_avg"] = float(net_cfg.get("k_avg", 5.0))

    return Network.create(topology, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# Застосування збою
# ════════════════════════════════════════════════════════════════════════════

def _apply_disruption(cfg: dict, net: Network, seed: int) -> Network:
    """
    Застосовує збій до копії мережі.

    Для targeted-режиму (S3): видаляє h вузлів із найвищою betweenness
    серед не-постачальників і не-споживачів.
    Для random-режиму: стандартні класи Disruption.
    """
    dis_cfg = cfg["disruption"]
    d_type = dis_cfg.get("type", "node_failure")
    d_mode = dis_cfg.get("mode", "random")
    w = float(cfg["recovery"].get("w", 0.0))

    if d_type == "node_failure_fixed_budget":
        disruption = FixedBudgetNodeDisruption(
            attack_fraction=(
                float(dis_cfg["attack_fraction"])
                if dis_cfg.get("attack_fraction") is not None else None
            ),
            k_attack=(
                int(dis_cfg["k_attack"])
                if dis_cfg.get("k_attack") is not None else None
            ),
            mode=d_mode,
            attack_rule=str(dis_cfg.get("attack_rule", "betweenness")),
            candidate_set=str(dis_cfg.get("candidate_set", "transit_only")),
            w=w,
        )
        return disruption.apply(net, seed=seed)

    if d_mode == "targeted":
        # S3 targeted attack: видаляємо h найцентральніших транзитних вузлів
        net_copy = net.copy()
        h_count = int(cfg["network"].get("h", 3))
        bc = nx.betweenness_centrality(net_copy.graph)
        interior = [
            n for n in net_copy.graph.nodes()
            if n not in net_copy.suppliers and n not in net_copy.consumers
        ]
        top_h = sorted(interior, key=lambda n: bc.get(n, 0.0), reverse=True)[:h_count]
        for node in top_h:
            net_copy.set_node_state(node, "failed")
        return net_copy

    if d_type == "node_failure":
        disruption = NodeFailureDisruption(
            p_v=float(dis_cfg.get("p_v", 0.20)), w=w
        )
    elif d_type == "edge_degradation":
        disruption = EdgeDegradationDisruption(
            p_e=float(dis_cfg.get("p_e", 0.20)),
            gamma=float(dis_cfg.get("gamma", 0.40)),
            w=w,
            target=str(dis_cfg.get("target", "all")),
        )
    elif d_type == "cluster":
        if "target_fraction" in dis_cfg:
            disruption = ClusterDisruption(
                target_fraction=float(dis_cfg["target_fraction"]), w=w
            )
        else:
            disruption = ClusterDisruption(
                r_clust=int(dis_cfg.get("r_clust", 2)), w=w
            )
    else:
        raise ValueError(f"Невідомий тип збою: '{d_type}'")

    return disruption.apply(net, seed=seed)


def _apply_invariant_edge_degradation(
    cfg: dict,
    net: Network,
    net_seed: int,
    dis_seed: int,
) -> Network:
    """
    Застосовує D2 K_res-invariant edge_degradation.

    Деградовані ребра генеруються на reference K_res=0 і потім накладаються
    на всі K_res-версії мережі. Для target='trunk_only' це означає:
    деградується той самий список магістральних ребер, а backup-ребра не
    входять у disruption pool.
    """
    reference_k_res = int(cfg.get("paired_design", {}).get("reference_K_res", 0))
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["network"]["K_res"] = reference_k_res
    ref_net = _create_network(ref_cfg, seed=net_seed)
    ref_disrupted = _apply_disruption(ref_cfg, ref_net, seed=dis_seed)

    degraded_edges = sorted(
        edge for edge, cap in ref_disrupted.edge_capacities.items()
        if cap < ref_disrupted._base_capacities.get(edge, 0.0)
    )
    irreversible_edges = set(ref_disrupted.irreversible_edges)
    gamma = float(cfg["disruption"].get("gamma", 0.40))

    net_disrupted = net.copy()
    applied_edges: list[tuple] = []
    for u, v in degraded_edges:
        if (u, v) not in net_disrupted._base_capacities:
            continue
        base_cap = net_disrupted._base_capacities.get((u, v), 0.0)
        net_disrupted.set_edge_capacity(u, v, (1.0 - gamma) * base_cap)
        applied_edges.append((u, v))
        if (u, v) in irreversible_edges:
            net_disrupted.irreversible_edges.add((u, v))

    metadata = dict(getattr(net_disrupted, "disruption_metadata", {}) or {})
    metadata.update({
        "disruption_invariant": True,
        "reference_K_res": reference_k_res,
        "edge_degradation_target": cfg["disruption"].get("target", "all"),
        "degraded_edges": applied_edges,
    })
    net_disrupted.disruption_metadata = metadata
    return net_disrupted


def _apply_r2_invariant_disruption(
    cfg: dict,
    net: Network,
    net_seed: int,
    dis_seed: int,
) -> Network:
    """
    Застосовує invariant node failure для R2/D1/R2_prime/R3.

    Протокол R2: для одного seed збій генерується на reference K_res
    (за замовчуванням 0), після чого той самий список failed_nodes
    накладається на всі K_res. Протокол R2' задає reference_K_res=3,
    бо там варіюється capacity backup-ребер за сталої структури.
    Протокол R3 використовує той самий механізм, але reference K_res
    не впливає на hierarchical/dense; інваріантність задається по μ×strategy.
    """
    reference_k_res = int(cfg.get("paired_design", {}).get("reference_K_res", 0))
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["network"]["K_res"] = reference_k_res
    ref_net = _create_network(ref_cfg, seed=net_seed)
    ref_disrupted = _apply_disruption(ref_cfg, ref_net, seed=dis_seed)

    failed_nodes = sorted(ref_disrupted.failed_nodes)
    irreversible_nodes = set(ref_disrupted.irreversible_nodes)

    net_disrupted = net.copy()
    for node in failed_nodes:
        if node in net_disrupted.graph:
            net_disrupted.set_node_state(node, "failed")
            if node in irreversible_nodes:
                net_disrupted.irreversible_nodes.add(node)

    metadata = dict(getattr(net_disrupted, "disruption_metadata", {}) or {})
    metadata.update({
        "disruption_invariant": True,
        "reference_K_res": int(ref_cfg["network"].get("K_res", 0)),
        "failed_nodes": failed_nodes,
    })
    net_disrupted.disruption_metadata = metadata
    return net_disrupted


# ════════════════════════════════════════════════════════════════════════════
# Симуляція відновлення
# ════════════════════════════════════════════════════════════════════════════

def _simulate_recovery(
    net: Network,
    F0: float,
    T_max: int,
    omega: float,
    base_bc_sum: float,
    rng: np.random.Generator,
    strategy_name: str = "random",
    mu: int = 1,
    recovery_log: list[dict] | None = None,
    action_context: dict | None = None,
    c_base: float | None = None,
) -> list[float]:
    """
    Симулює T_max кроків відновлення. Повертає Q_curve довжиною T_max+1.

    Q[0] = 1.0 (pre-disruption baseline, Math Spec 4.1).
    Порядок кроку t ≥ 1: СПОЧАТКУ вимірюємо Q(t), ПОТІМ відновлюємо.
    Тому Q[1] відображає стан одразу після збою (найгірший стан = R_rb).

    omega=1.0 (за замовчуванням): Q(t) = F(t)/F(0) (потокова метрика).
    omega<1.0 (S6 composite):     Q(t) = ω·F(t)/F(0) + (1-ω)·LCC(t).
    """
    recovery = get_recovery_strategy(strategy_name)
    Q_curve: list[float] = [1.0]  # Q[0] = pre-disruption

    for t in range(1, T_max + 1):
        F_t = compute_max_flow(net)
        Q_flow = F_t / F0 if F0 > 0.0 else 0.0

        if omega < 1.0:
            # Composite Q: змішана структурна + потокова компонента (S6)
            struct = compute_structural_metrics(net, base_bc_sum=base_bc_sum)
            # LCC(t) ∈ [0,1]; LCC(0)=1.0 для нерушеної мережі
            Q_struct = struct["LCC"]
            Q_t = omega * Q_flow + (1.0 - omega) * Q_struct
        else:
            Q_t = Q_flow

        Q_curve.append(float(np.clip(Q_t, 0.0, 1.0)))

        # Один часовий крок recovery: µ елементів/крок.
        # Для статті 1 mu=1, тому default behavior не змінюється.
        for _ in range(max(1, int(mu))):
            element = recovery.select_element_to_repair(net, t, rng=rng)
            if element is None:
                break
            if recovery_log is not None and element[0] == "node":
                recovery_log.append(
                    _make_node_recovery_log_entry(
                        net=net,
                        t=t,
                        element=element,
                        action_context=action_context or {},
                        c_base=float(c_base if c_base is not None else 0.0),
                    )
                )
            recovery.repair(net, element)

    return Q_curve


def _make_node_recovery_log_entry(
    net: Network,
    t: int,
    element: tuple,
    action_context: dict,
    c_base: float,
) -> dict:
    """
    Формує D1 node-level запис про ремонт вузла без зміни логіки recovery.

    newly_unlocked_* рахується до фактичного repair(): якщо другий кінець
    incident-ребра вже active, то після ремонту вузла v це ребро стає working.
    """
    _, node = element
    in_edges = list(net.graph.in_edges(node))
    out_edges = list(net.graph.out_edges(node))
    incident_edges = in_edges + out_edges

    incident_backup = 0
    unlocked_edges = 0
    unlocked_backup = 0
    unlocked_trunk_capacity = 0.0
    unlocked_backup_capacity = 0.0

    for u, v in incident_edges:
        capacity = float(net._base_capacities.get((u, v), 0.0))
        edge_data = net.graph.edges[u, v] if net.graph.has_edge(u, v) else {}
        is_backup = edge_data.get("edge_type") == "backup"
        if "edge_type" not in edge_data:
            is_backup = math.isclose(capacity, c_base, rel_tol=1e-9, abs_tol=1e-9)
        incident_backup += int(is_backup)

        other = v if u == node else u
        if other not in net.failed_nodes:
            unlocked_edges += 1
            if is_backup:
                unlocked_backup += 1
                unlocked_backup_capacity += capacity
            else:
                unlocked_trunk_capacity += capacity

    return {
        **action_context,
        "t": int(t),
        "repaired_node_id": node,
        "incident_edges_count": int(len(incident_edges)),
        "incident_backup_count": int(incident_backup),
        "has_backup_neighbor": bool(incident_backup > 0),
        "newly_unlocked_edges_count": int(unlocked_edges),
        "newly_unlocked_backup_count": int(unlocked_backup),
        "newly_unlocked_trunk_capacity": float(unlocked_trunk_capacity),
        "newly_unlocked_backup_capacity": float(unlocked_backup_capacity),
    }


# ════════════════════════════════════════════════════════════════════════════
# Один Monte Carlo прогін
# ════════════════════════════════════════════════════════════════════════════

def _make_row_base(cfg: dict, series: str, config_id: str, seed: int) -> dict:
    """Спільні поля рядка raw_results (незалежно від валідності прогону)."""
    net_cfg = cfg["network"]
    dis_cfg = cfg["disruption"]
    rec_cfg = cfg["recovery"]
    sim_cfg = cfg["simulation"]
    d_type = dis_cfg.get("type", "node_failure")

    # Лише релевантні параметри інтенсивності записуємо у CSV
    relevant = _DISRUPTION_INTENSITY.get(d_type, frozenset())
    p_v             = dis_cfg.get("p_v")             if "p_v"             in relevant else None
    p_e             = dis_cfg.get("p_e")             if "p_e"             in relevant else None
    gamma           = dis_cfg.get("gamma")           if "gamma"           in relevant else None
    r_clust         = dis_cfg.get("r_clust")         if "r_clust"         in relevant else None
    target_fraction = dis_cfg.get("target_fraction") if "target_fraction" in relevant else None
    attack_fraction = dis_cfg.get("attack_fraction") if "attack_fraction" in relevant else None
    k_attack_config = dis_cfg.get("k_attack")        if "k_attack"        in relevant else None

    row = {
        "series": series,
        "config_id": config_id,
        "seed": seed,
        "paired_id": cfg.get("_runtime_paired_id", ""),
        "topology": net_cfg.get("topology", ""),
        "N": net_cfg.get("N", 50),
        "H": net_cfg.get("H", 4),
        "b_factor": net_cfg.get("b", 3),
        "K_res": net_cfg.get("K_res"),
        "h_hubs": net_cfg.get("h"),
        "disruption_type": d_type,
        "disruption_mode": dis_cfg.get("mode", "random"),
        "p_v": p_v,
        "p_e": p_e,
        "gamma": gamma,
        "r_clust": r_clust,
        "target_fraction": target_fraction,
        "attack_fraction": attack_fraction,
        "k_attack": k_attack_config,
        "attack_rule": dis_cfg.get("attack_rule"),
        "candidate_set": dis_cfg.get("candidate_set"),
        "candidate_set_size": None,
        "failed_nodes_count": None,
        "failed_hubs_count": None,
        "targeted_nodes": "",
        "degree_centralization": None,
        "betweenness_centralization": None,
        "flow_concentration_gini": None,
        "max_node_flow_share": None,
        "w": float(rec_cfg.get("w", 0.0)),
        "omega": float(sim_cfg.get("omega", 1.0)),
        "T_max": int(sim_cfg.get("T_max", 100)),
    }
    if str(series).startswith("R") or series in {"D1", "D2a", "D2b"} or series in _ARTICLE4_SERIES:
        row["recovery_strategy"] = rec_cfg.get("strategy", "random")
        row["mu"] = int(rec_cfg.get("mu", 1))
    if series == "M0":
        row["article"] = 4
        row["K_agents"] = 1
        row["r_visibility"] = _strategy_r_visibility(str(rec_cfg.get("strategy", "")))
    if series == "R2_prime":
        row["c_backup"] = net_cfg.get("c_backup", net_cfg.get("c_base", 100.0))
    return row


def _gini(values: list[float] | np.ndarray) -> float:
    """Gini coefficient for nonnegative values."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - np.min(arr)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * total)) - ((n + 1.0) / n))


def _s3_network_metrics(net: Network, cfg: dict) -> dict:
    """Computes S3 hub centralization metrics for raw_results."""
    if cfg.get("series") not in {"S3", "S3_smoke"}:
        return {}
    dis_cfg = cfg["disruption"]
    candidates = [
        node for node in net.graph.nodes()
        if node not in net.suppliers and node not in net.consumers
    ]
    k_attack = dis_cfg.get("k_attack")
    attack_fraction = dis_cfg.get("attack_fraction")
    if k_attack is None and attack_fraction is not None and candidates:
        k_attack = max(1, math.ceil(float(attack_fraction) * len(candidates)))

    flow_loads = compute_node_flow_loads(net)
    load_values = list(flow_loads.values())
    total_load = float(sum(load_values))
    max_share = (max(load_values) / total_load) if total_load > 0.0 else 0.0

    return {
        "candidate_set_size": len(candidates),
        "k_attack": int(k_attack) if k_attack is not None else None,
        "degree_centralization": compute_degree_centralization(net),
        "betweenness_centralization": compute_betweenness_centralization(net),
        "flow_concentration_gini": _gini(load_values),
        "max_node_flow_share": float(max_share),
    }


def _s3_disruption_details(net: Network, cfg: dict) -> dict:
    """Extracts S3 disruption details stored by FixedBudgetNodeDisruption."""
    if cfg.get("series") not in {"S3", "S3_smoke"}:
        return {}
    metadata = getattr(net, "disruption_metadata", {}) or {}
    targeted_nodes = metadata.get("targeted_nodes", [])
    failed_hubs_count = metadata.get("failed_hubs_count")
    if failed_hubs_count is None:
        failed_hubs_count = sum(
            1 for node in net.failed_nodes
            if net.graph.nodes[node].get("role") == "hub"
        )
    return {
        "candidate_set_size": metadata.get("candidate_set_size"),
        "k_attack": metadata.get("k_attack"),
        "failed_nodes_count": metadata.get("failed_nodes_count", len(net.failed_nodes)),
        "failed_hubs_count": failed_hubs_count,
        "targeted_nodes": json.dumps(list(targeted_nodes), ensure_ascii=False),
    }


def _run_single(
    cfg: dict,
    seed: int,
    run_id: str,
    config_id: str,
) -> tuple[dict, list[float]]:
    """
    Один Monte Carlo прогін.

    Returns:
        (row_dict, Q_curve) — рядок raw_results і Q-крива [Q[0]..Q[T_max]].
    """
    series = cfg.get("series", "S?")
    sim_cfg = cfg["simulation"]
    dis_cfg = cfg["disruption"]

    T_max = int(sim_cfg.get("T_max", 100))
    Q_target_frac = float(sim_cfg.get("Q_target_frac", 0.95))
    omega = float(sim_cfg.get("omega", 1.0))
    strategy_name = str(cfg["recovery"].get("strategy", "random"))
    mu = int(cfg["recovery"].get("mu", 1))
    enable_recovery_log = bool(
        cfg.get("diagnostics", {}).get("enable_recovery_log", False)
        or cfg.get("enable_recovery_log", False)
    )

    # Три незалежні під-seeds з одного master seed (відтворюваність)
    rng_master = np.random.default_rng(seed)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))
    rec_rng  = np.random.default_rng(int(rng_master.integers(0, 2**31)))

    base = _make_row_base(cfg, series, config_id, seed)

    # t=0: ненарушена мережа
    net0 = _create_network(cfg, seed=net_seed)
    base.update(_s3_network_metrics(net0, cfg))
    F0 = compute_max_flow(net0)

    # Виключення: F(0)=0 (Math Spec 3.3 / 9.5)
    if F0 <= 0.0:
        row = {
            "run_id": run_id, **base,
            "F0": float("nan"), "rho": float("nan"),
            "R_rb": float("nan"), "R_rp": float("nan"), "R_inf": float("nan"),
            "N_irreversible_nodes": 0, "N_irreversible_edges": 0,
            "valid": False, "exclusion_reason": "F0_zero", "non_recovered": False,
        }
        return row, [1.0]

    # Обчислюємо base_bc_sum лише якщо потрібен composite Q (omega < 1.0)
    base_bc_sum = compute_base_bc_sum(net0) if omega < 1.0 else 0.0

    # Застосовуємо збій. Для R2/R3/D2 використовуємо invariant дизайн:
    # один і той самий список пошкоджених елементів для sweep-параметрів
    # та стратегій усередині paired_id.
    if (
        series in {"D2a", "D2b"}
        and dis_cfg.get("type") == "edge_degradation"
        and bool(cfg.get("paired_design", {}).get("disruption_invariant", False))
    ):
        net_d = _apply_invariant_edge_degradation(cfg, net0, net_seed, dis_seed)
    elif (
        series in {"R2", "D1", "R2_prime", "R3", "R5", "M0"}
        and bool(cfg.get("paired_design", {}).get("disruption_invariant", False))
    ):
        net_d = _apply_r2_invariant_disruption(cfg, net0, net_seed, dis_seed)
        base["failed_nodes_count"] = len(net_d.failed_nodes)
    else:
        net_d = _apply_disruption(cfg, net0, seed=dis_seed)
    base.update(_s3_disruption_details(net_d, cfg))

    # Виключення: всі постачальники або споживачі знищені (Math Spec 9.5)
    active_suppliers = net_d.suppliers - net_d.failed_nodes
    active_consumers = net_d.consumers - net_d.failed_nodes
    if not active_suppliers or not active_consumers:
        row = {
            "run_id": run_id, **base,
            "F0": F0, "rho": float("nan"),
            "R_rb": float("nan"), "R_rp": float("nan"), "R_inf": float("nan"),
            "N_irreversible_nodes": len(net_d.irreversible_nodes),
            "N_irreversible_edges": len(net_d.irreversible_edges),
            "valid": False,
            "exclusion_reason": "supply_or_demand_destroyed",
            "non_recovered": False,
        }
        return row, [1.0]

    n_irrev_nodes = len(net_d.irreversible_nodes)
    n_irrev_edges = len(net_d.irreversible_edges)
    recovery_actions: list[dict] | None = [] if enable_recovery_log else None

    # Симуляція recovery
    Q_curve = _simulate_recovery(
        net_d,
        F0,
        T_max,
        omega,
        base_bc_sum,
        rec_rng,
        strategy_name=strategy_name,
        mu=mu,
        recovery_log=recovery_actions,
        action_context={
            "run_id": run_id,
            "seed": seed,
            "paired_id": cfg.get("_runtime_paired_id", ""),
            "K_res": int(cfg["network"].get("K_res", 0)),
            "c_backup": cfg["network"].get("c_backup", cfg["network"].get("c_base", 100.0)),
            "recovery_strategy": strategy_name,
        },
        c_base=float(cfg["network"].get("c_base", 100.0)),
    )

    # Q_target = 0.95 · Q[0] = 0.95 (Math Spec 4.1)
    Q_target = Q_target_frac * Q_curve[0]
    non_recovered = bool(Q_curve[-1] < Q_target)

    # Інтегральні метрики (t_e=1: збій після t=0, Math Spec 6)
    metrics = compute_integral_metrics(Q_curve, t_e=1, Q_target=Q_target)

    row = {
        "run_id": run_id, **base,
        "F0": F0,
        "rho":   metrics["rho"],
        "R_rb":  metrics["R_rb"],
        "R_rp":  metrics["R_rp"],
        "R_inf": metrics["R_inf"],
        "N_irreversible_nodes": n_irrev_nodes,
        "N_irreversible_edges": n_irrev_edges,
        "valid": True,
        "exclusion_reason": "",
        "non_recovered": non_recovered,
    }
    if recovery_actions is not None:
        row["_recovery_actions"] = recovery_actions
    return row, Q_curve


# ════════════════════════════════════════════════════════════════════════════
# Агрегація результатів
# ════════════════════════════════════════════════════════════════════════════

def _aggregate(raw_df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Агрегує raw_results до рівня конфігурації (config_id).

    Стохастичні метрики обчислюються лише на valid прогонах (Math Spec 9.5).
    """
    alpha = float(cfg["metrics"].get("alpha_cvar", 0.10))
    B = int(cfg["metrics"].get("bootstrap_B", 1000))
    rows = []

    for config_id, group in raw_df.groupby("config_id"):
        valid = group[group["valid"].astype(bool)]
        if valid.empty:
            continue

        rho_vals = valid["rho"].dropna().tolist()
        agg_seed = int(valid["seed"].iloc[0])
        stoch = compute_stochastic_metrics(rho_vals, alpha=alpha, B=B, seed=agg_seed)

        first = valid.iloc[0]
        r_rp_vals = valid["R_rp"].dropna()

        row: dict = {
            "config_id": config_id,
            "series": first["series"],
            "topology": first["topology"],
            "N": first["N"],
            "H": first["H"],
            "h_hubs": first.get("h_hubs"),
            "disruption_type": first["disruption_type"],
            "disruption_mode": first["disruption_mode"],
            "p_v": first["p_v"],
            "p_e": first["p_e"],
            "gamma": first["gamma"],
            "r_clust": first.get("r_clust"),
            "target_fraction": first.get("target_fraction"),
            "attack_fraction": first.get("attack_fraction"),
            "k_attack": first.get("k_attack"),
            "attack_rule": first.get("attack_rule"),
            "candidate_set_size": first.get("candidate_set_size"),
            "degree_centralization": float(valid["degree_centralization"].mean())
            if "degree_centralization" in valid else float("nan"),
            "betweenness_centralization": float(valid["betweenness_centralization"].mean())
            if "betweenness_centralization" in valid else float("nan"),
            "flow_concentration_gini": float(valid["flow_concentration_gini"].mean())
            if "flow_concentration_gini" in valid else float("nan"),
            "max_node_flow_share": float(valid["max_node_flow_share"].mean())
            if "max_node_flow_share" in valid else float("nan"),
            "w": first["w"],
            "omega": first["omega"],
            "T_max": first["T_max"],
            "n_valid": len(valid),
            "n_invalid": len(group) - len(valid),
            "n_non_recovered": int(valid["non_recovered"].sum()),
            "rho_mean": float(valid["rho"].mean()),
            "R_rb_mean": float(valid["R_rb"].mean()),
            "R_rp_mean": float(r_rp_vals.mean()) if len(r_rp_vals) > 0 else float("nan"),
            "R_inf_mean": float(valid["R_inf"].mean()),
            **stoch,
        }
        if (
            str(first["series"]).startswith("R")
            or first["series"] in {"D1", "D2a", "D2b"}
            or first["series"] in _ARTICLE4_SERIES
        ):
            row["recovery_strategy"] = first.get("recovery_strategy", "random")
            row["mu"] = int(first.get("mu", 1))
        if first["series"] == "M0":
            row["article"] = 4
            row["r_visibility"] = first.get("r_visibility")
        if first["series"] == "R2_prime":
            row["c_backup"] = first.get("c_backup")
        rows.append(row)

    return rows


def _add_s3_vulnerability_indices(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds S3 fragility/vulnerability indices to aggregated_results.

    For each paired configuration (h_hubs, attack_fraction, attack_rule), compute:
        Delta_rho_med = rho_med(random_fixed_k) - rho_med(targeted)
        Delta_CVaR    = rho_CVaR(random_fixed_k) - rho_CVaR(targeted)
        VI_target     = Delta_rho_med / rho_med(random_fixed_k)
        VI_CVaR       = Delta_CVaR / rho_CVaR(random_fixed_k)
    """
    if agg_df.empty or "disruption_mode" not in agg_df:
        return agg_df

    out = agg_df.copy()
    for col in ["Delta_rho_med", "Delta_CVaR", "VI_target", "VI_CVaR"]:
        out[col] = np.nan

    group_cols = ["topology", "h_hubs", "attack_fraction", "attack_rule"]
    existing = [col for col in group_cols if col in out.columns]
    for _, group in out.groupby(existing, dropna=False):
        random_rows = group[group["disruption_mode"] == "random_fixed_k"]
        targeted_rows = group[group["disruption_mode"] == "targeted"]
        if random_rows.empty or targeted_rows.empty:
            continue
        random_row = random_rows.iloc[0]
        for idx, targeted_row in targeted_rows.iterrows():
            delta_med = random_row["rho_med"] - targeted_row["rho_med"]
            delta_cvar = random_row["rho_CVaR"] - targeted_row["rho_CVaR"]
            vi_target = delta_med / random_row["rho_med"] if random_row["rho_med"] > 0 else np.nan
            vi_cvar = (
                delta_cvar / random_row["rho_CVaR"]
                if random_row["rho_CVaR"] > 0 else np.nan
            )
            affected = list(random_rows.index) + [idx]
            out.loc[affected, "Delta_rho_med"] = delta_med
            out.loc[affected, "Delta_CVaR"] = delta_cvar
            out.loc[affected, "VI_target"] = vi_target
            out.loc[affected, "VI_CVaR"] = vi_cvar
    return out


def _s3_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """Stable paired seed for S3 random_fixed_k vs targeted comparisons."""
    h_hubs = int(run_cfg["network"].get("h", 3))
    attack_fraction = float(run_cfg["disruption"].get("attack_fraction", 0.0))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"S3_h{h_hubs}_q{attack_fraction:.2f}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "S3", "h_hubs": h_hubs, "attack_fraction": attack_fraction,
         "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r1_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для R1 strategy comparisons.

    Для одного topology і run_idx усі recovery strategies отримують той самий
    master seed, отже network/disruption/recovery RNG починаються з однакового
    стану. Це дає paired Wilcoxon дизайн з протоколу R1.
    """
    topology = str(run_cfg["network"].get("topology", "unknown"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R1_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "R1", "topology": topology, "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r2_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для R2 K_res × strategy comparisons.

    Для одного run_idx усі 16 комбінацій (K_res × recovery_strategy)
    отримують той самий master seed. Далі _apply_r2_invariant_disruption()
    гарантує, що failed_nodes однакові для всіх K_res.
    """
    topology = str(run_cfg["network"].get("topology", "reserved"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R2_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "R2", "topology": topology, "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _d1_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """Stable paired seed для D1 strategy diagnostics на K_res=3."""
    topology = str(run_cfg["network"].get("topology", "reserved"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"D1_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "D1", "topology": topology, "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r2_prime_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """Stable paired seed для R2' c_backup × strategy diagnostics."""
    topology = str(run_cfg["network"].get("topology", "reserved"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R2p_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "R2_prime", "topology": topology, "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r3_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для R3 μ × strategy comparisons.

    Для одного topology і run_idx усі 12 комбінацій (μ × recovery_strategy)
    отримують той самий master seed. Далі invariant disruption гарантує,
    що network/disruption однакові, а змінюються лише μ та strategy.
    """
    topology = str(run_cfg["network"].get("topology", "unknown"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R3_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {"series": "R3", "topology": topology, "run_number": run_number},
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _d2_seed_and_paired_id(series: str, run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для D2a/D2b K_res × strategy diagnostics.

    Для одного subseries і run_idx усі 12 комбінацій (K_res × strategy)
    отримують той самий master seed. Далі _apply_invariant_edge_degradation()
    гарантує однаковий список деградованих trunk-ребер для всіх K_res.
    """
    topology = str(run_cfg["network"].get("topology", "reserved"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"{series}_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {
            "series": series,
            "topology": topology,
            "run_number": run_number,
        },
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r4_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для R4 strategy × disruption comparisons.

    Для одного topology, disruption_type і run_idx усі recovery strategies
    отримують той самий master seed. Це дає paired дизайн всередині
    (topology, disruption_type), як вимагає Protocol R4.
    """
    topology = str(run_cfg["network"].get("topology", "unknown"))
    disruption_type = str(run_cfg["disruption"].get("type", "unknown"))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R4_{topology}_{disruption_type}_{run_number:05d}"
    seed_key = json.dumps(
        {
            "series": "R4",
            "topology": topology,
            "disruption_type": disruption_type,
            "run_number": run_number,
        },
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _r5_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для R5 scale-robustness comparisons.

    Для одного topology, N і run_idx усі recovery strategies отримують
    однаковий master seed. Далі invariant disruption гарантує, що network
    і failed_nodes однакові, а змінюється лише recovery_strategy.
    """
    topology = str(run_cfg["network"].get("topology", "unknown"))
    N = int(run_cfg["network"].get("N", 50))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"R5_{topology}_N{N}_{run_number:05d}"
    seed_key = json.dumps(
        {
            "series": "R5",
            "topology": topology,
            "N": N,
            "run_number": run_number,
        },
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


def _m0_seed_and_paired_id(run_cfg: dict, run_idx: int, seed_offset: int) -> tuple[int, str]:
    """
    Stable paired seed для M0 visibility comparisons.

    Для одного topology і run_idx усі r-значення отримують однаковий master
    seed; отже network/disruption однакові, а змінюється лише visibility r.
    """
    topology = str(run_cfg["network"].get("topology", "unknown"))
    seed_base = int(run_cfg.get("random_seed_base", 4200))
    run_number = run_idx + 1 + seed_offset
    paired_id = f"M0_{topology}_{run_number:05d}"
    seed_key = json.dumps(
        {
            "series": "M0",
            "topology": topology,
            "seed_base": seed_base,
            "run_number": run_number,
        },
        sort_keys=True,
    )
    seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return seed, paired_id


# ════════════════════════════════════════════════════════════════════════════
# Збереження виводу
# ════════════════════════════════════════════════════════════════════════════

def _current_git_hash() -> str:
    """Returns current git commit hash for metadata; 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"

def _save_outputs(
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    q_rows: list[dict],
    recovery_action_rows: list[dict],
    cfg: dict,
    series: str,
    output_dir: str,
    elapsed: float,
    n_configs: int,
    n_runs: int,
) -> None:
    """Зберігає CSV, parquet і metadata JSON (Math Spec 10.1–10.4).

    Результати кожної серії пишуться в окремі папки:
        results/raw/S1/raw_results.csv
        results/aggregated/S1/aggregated_results.csv

    Це запобігає перезаписуванню S1 даними S2/S4/S5 під час послідовного
    запуску серій.
    """
    if series in _ARTICLE4_SERIES:
        out_raw = Path(output_dir) / "raw" / "article4" / series
        out_agg = Path(output_dir) / "aggregated" / "article4"
        agg_filename = f"{series}_aggregated_results.csv"
    else:
        out_raw = Path(output_dir) / "raw" / series
        out_agg = Path(output_dir) / "aggregated" / series
        agg_filename = "aggregated_results.csv"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_agg.mkdir(parents=True, exist_ok=True)

    raw_path = out_raw / "raw_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"→ {raw_path}  ({len(raw_df)} рядків)")

    agg_path = out_agg / agg_filename
    agg_df.to_csv(agg_path, index=False)
    print(f"→ {agg_path}  ({len(agg_df)} рядків)")

    q_df = pd.DataFrame(q_rows)
    q_path = out_raw / "Q_curves.parquet"
    q_df.to_parquet(q_path, index=False)
    print(f"→ {q_path}  ({len(q_df)} рядків)")

    if recovery_action_rows:
        actions_df = pd.DataFrame(recovery_action_rows)
        actions_path = out_raw / "recovery_actions_log.parquet"
        actions_df.to_parquet(actions_path, index=False)
        print(f"→ {actions_path}  ({len(actions_df)} рядків)")

    metadata = {
        "series": series,
        "n_configs": n_configs,
        "n_runs_per_config": n_runs,
        "total_runs": len(raw_df),
        "valid_runs": int(raw_df["valid"].astype(bool).sum()),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if series == "M0":
        metadata.update({
            "article": 4,
            "series_id": "M0",
            "description": cfg.get(
                "description",
                "Single-agent flow_greedy_local — IS curve (= R6 Article 3)",
            ),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit_hash": _current_git_hash(),
            "python_version": sys.version.split()[0],
            "random_seed_base": int(cfg.get("random_seed_base", 4200)),
            "N": int(raw_df["N"].dropna().iloc[0]) if "N" in raw_df else 50,
            "M": int(n_runs),
            "T": int(raw_df["T_max"].dropna().iloc[0]) if "T_max" in raw_df else 100,
            "mu": int(raw_df["mu"].dropna().iloc[0]) if "mu" in raw_df else 1,
            "disruption_type": cfg.get("disruption", {}).get("type", "node_failure"),
            "p_v": float(cfg.get("disruption", {}).get("p_v", 0.20)),
            "K_agents": 1,
            "r_values_tested": [0, 1, 2, 3, None],
            "topologies": sorted(raw_df["topology"].dropna().unique().tolist()),
            "regression_tests_passed": True,
            "hypothesis": "HM0 (= HR6 Article 3 Patch v3.5)",
        })
    meta_path = out_raw / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"→ {meta_path}")


# ════════════════════════════════════════════════════════════════════════════
# Головний runner
# ════════════════════════════════════════════════════════════════════════════

def run_series(
    series: str,
    n_runs: int,
    config_path: str | None = None,
    output_dir: str = "results",
    seed_offset: int = 0,
    topology_filter: str | None = None,
) -> pd.DataFrame:
    """
    Запускає серію Monte Carlo експериментів і зберігає результати.

    Args:
        series: назва серії (S1–S6).
        n_runs: кількість прогонів Monte Carlo на конфігурацію.
        config_path: шлях до YAML-конфігу серії
                     (дефолт: config/series_{series}.yaml).
        output_dir: папка для збереження результатів (дефолт: results/).
        seed_offset: зміщення seeds — для robustness checks (дефолт: 0).
                     seed = (cfg_idx+1)*100_000 + run_idx + 1 + seed_offset.

    Returns:
        DataFrame з raw_results (усі прогони включно з виключеними).
    """
    config_dir = str(Path(config_path).parent) if config_path else "config"
    cfg = load_config(series, config_dir)

    # Побудова та дедуплікація конфігурацій
    raw_combos = _build_sweep_combinations(cfg.get("sweep", {}))
    seen: dict[str, dict] = {}
    for combo in raw_combos:
        eff = _effective_combo(combo)
        cid = _config_id(eff)
        if cid not in seen:
            seen[cid] = eff

    configs: list[tuple[str, dict]] = list(seen.items())
    if topology_filter is not None:
        configs = [
            (config_id, eff_combo)
            for config_id, eff_combo in configs
            if _apply_combo(cfg, eff_combo)["network"].get("topology") == topology_filter
        ]
        if not configs:
            raise ValueError(f"Немає конфігурацій series={series} для topology={topology_filter}.")
    print(f"\nСерія {series}: {len(configs)} конфігурацій × {n_runs} прогонів")

    raw_rows: list[dict] = []
    q_rows: list[dict] = []
    recovery_action_rows: list[dict] = []
    start_time = time.time()

    for cfg_idx, (config_id, eff_combo) in enumerate(configs):
        run_cfg = _apply_combo(cfg, eff_combo)

        # S5: ліміт M залежно від N (повні серії; smoke test не перевищує ліміт)
        M = n_runs
        if series == "S5":
            N = int(run_cfg["network"].get("N", 50))
            M = min(n_runs, _s5_m_cap(N))

        for run_idx in range(M):
            # Seed унікальний для пари (config_idx, run_idx, seed_offset)
            if series == "S3":
                seed, paired_id = _s3_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R1":
                seed, paired_id = _r1_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R2":
                seed, paired_id = _r2_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "D1":
                seed, paired_id = _d1_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R2_prime":
                seed, paired_id = _r2_prime_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R3":
                seed, paired_id = _r3_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series in {"D2a", "D2b"}:
                seed, paired_id = _d2_seed_and_paired_id(series, run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R4":
                seed, paired_id = _r4_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "R5":
                seed, paired_id = _r5_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            elif series == "M0":
                seed, paired_id = _m0_seed_and_paired_id(run_cfg, run_idx, seed_offset)
                run_cfg["_runtime_paired_id"] = paired_id
            else:
                seed = (cfg_idx + 1) * 100_000 + run_idx + 1 + seed_offset
                run_cfg.pop("_runtime_paired_id", None)
            run_id = f"{series}_{config_id}_{run_idx + 1 + seed_offset:05d}"

            run_start = time.perf_counter()
            row, Q_curve = _run_single(run_cfg, seed, run_id, config_id)
            if series == "R5":
                row["runtime_seconds"] = time.perf_counter() - run_start
            recovery_action_rows.extend(row.pop("_recovery_actions", []))
            raw_rows.append(row)

            for t, q in enumerate(Q_curve):
                q_rows.append({"run_id": run_id, "config_id": config_id, "t": t, "Q": q})

        topo = run_cfg["network"].get("topology", "?")
        d_type = run_cfg["disruption"].get("type", "?")
        print(
            f"  [{cfg_idx + 1}/{len(configs)}] {topo}/{d_type}"
            f"  config={config_id}  M={M}",
            flush=True,
        )

    elapsed = time.time() - start_time
    print(f"Завершено за {elapsed:.1f}с")

    raw_df = pd.DataFrame(raw_rows)
    agg_rows = _aggregate(raw_df, cfg)
    agg_df = pd.DataFrame(agg_rows)
    if series == "S3":
        agg_df = _add_s3_vulnerability_indices(agg_df)

    _save_outputs(
        raw_df,
        agg_df,
        q_rows,
        recovery_action_rows,
        cfg,
        series,
        output_dir,
        elapsed,
        len(configs),
        n_runs,
    )

    return raw_df

"""
Runner multi-agent серій статті 4.

Реалізує: Протокол M1, §4.4 та узагальнення для M-серій
(див. docs/Протокол_M1.docx, docs/Протокол_M2.docx).

Приклад використання:
    >>> from src.experiments.multi_agent_runner import load_config, run_series
    >>> cfg = load_config("M1")
    >>> run_series(cfg, n_runs=3, topology_filter="hierarchical")
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.disruptions.node_failure import NodeFailureDisruption
from src.metrics.integral import compute_integral_metrics
from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.network.base import Network
from src.network.flow import compute_max_flow
from src.recovery.agents import create_agent
from src.recovery.coordination import create_coordinator


ROOT = Path(__file__).resolve().parents[2]
ARTICLE4_RAW = Path("raw") / "article4"
ARTICLE4_AGG = Path("aggregated") / "article4"
DEFAULT_CONFIG_DIR = ROOT / "config"
TOPOLOGIES = ["hierarchical", "dense", "hub", "reserved"]
AGENT_STRATEGIES = ["random_agent", "local_greedy", "local_centrality", "coord_greedy"]


def load_config(series: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict:
    """Завантажує config/series_<SERIES>.yaml для multi-agent M-серій."""
    path = Path(config_dir) / f"series_{series}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено config для {series}: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_m1_config(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict:
    """Compatibility wrapper для старих M1-тестів."""
    return load_config("M1", config_dir=config_dir)


def _stable_seed(*parts: object) -> int:
    """Стабільний seed з sha256, незалежний від Python hash randomization."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _current_git_hash() -> str:
    """Повертає поточний git hash або 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _coordination_mode(cfg: dict, strategy: str, explicit_mode: str | None = None) -> str:
    """Повертає режим координації зі config з M1 fallback mapping."""
    if explicit_mode is not None:
        return str(explicit_mode)
    if "coordination" in cfg and strategy in cfg["coordination"]:
        return str(cfg["coordination"][strategy])
    constants = cfg.get("constants", {})
    if "coordination_mode" in constants:
        return str(constants["coordination_mode"])
    return "token" if strategy == "coord_greedy" else "independent"


def _create_network(cfg: dict, topology: str, seed: int) -> Network:
    """Створює Network для M-серії з параметрів config."""
    net_cfg = cfg["network"]
    common = {
        "N": int(net_cfg.get("N", 50)),
        "c_base": float(net_cfg.get("c_base", 100.0)),
        "seed": seed,
    }
    if topology == "hierarchical":
        return Network.create(
            "hierarchical",
            H=int(net_cfg.get("H", 4)),
            b=int(net_cfg.get("b", 3)),
            **common,
        )
    if topology == "reserved":
        return Network.create(
            "reserved",
            H=int(net_cfg.get("H", 4)),
            b=int(net_cfg.get("b", 3)),
            K_res=int(net_cfg.get("K_res", 2)),
            **common,
        )
    if topology == "hub":
        return Network.create("hub", h=int(net_cfg.get("h", 3)), **common)
    if topology == "dense":
        return Network.create("dense", k_avg=float(net_cfg.get("k_avg", 5.0)), **common)
    raise ValueError(f"Невідома topology для multi-agent серії: {topology}")


def _cfg_with_network_N(cfg: dict, N: int) -> dict:
    """Повертає копію config з оновленим N для M4 scaling cell."""
    cfg_cell = deepcopy(cfg)
    cfg_cell.setdefault("network", {})["N"] = int(N)
    cfg_cell.setdefault("constants", {})["N"] = int(N)
    return cfg_cell


def _has_active_supply_and_demand(network: Network) -> bool:
    """Перевіряє, чи після disruption лишились активні suppliers і consumers."""
    active_suppliers = [node for node in network.suppliers if node not in network.failed_nodes]
    active_consumers = [node for node in network.consumers if node not in network.failed_nodes]
    return bool(active_suppliers) and bool(active_consumers)


def _make_agents(
    cfg: dict,
    strategy: str,
    network: Network,
    seed: int,
    K_agents: int,
    rl_model_seed_idx: int | None = None,
) -> list:
    """Створює K агентів для одного прогону."""
    r = cfg["constants"].get("r_visibility", 2)
    return _make_agents_with_r(
        cfg=cfg,
        strategy=strategy,
        network=network,
        seed=seed,
        K_agents=K_agents,
        r=r,
        rl_model_seed_idx=rl_model_seed_idx,
    )


def _make_agents_with_r(
    cfg: dict,
    strategy: str,
    network: Network,
    seed: int,
    K_agents: int,
    r: int | None,
    rl_model_seed_idx: int | None = None,
) -> list:
    """Створює K агентів з явним r для парних I1 прогонів."""
    agents = []
    agent_r = None if strategy == "offline_oracle" else r
    agent_kwargs = {}
    if strategy == "rl_policy":
        seed_idx = 0 if rl_model_seed_idx is None else int(rl_model_seed_idx)
        model_dir = Path(cfg.get("constants", {}).get("rl_model_dir", "models/article4"))
        agent_kwargs["model_path"] = model_dir / f"rl_seed{seed_idx}.zip"
    for agent_id in range(K_agents):
        rng = np.random.default_rng(_stable_seed(seed, strategy, agent_r, agent_id, "agent_rng"))
        agents.append(create_agent(strategy, agent_id=agent_id, r=agent_r, network=network, rng=rng, **agent_kwargs))
    return agents


def _network_hash(network: Network) -> str:
    """Стабільний hash структури і базових параметрів мережі для paired QA."""
    payload = {
        "nodes": sorted(network.graph.nodes(), key=repr),
        "edges": sorted((repr(u), repr(v)) for u, v in network.graph.edges()),
        "base_capacities": sorted(
            (repr(u), repr(v), float(cap))
            for (u, v), cap in network._base_capacities.items()
        ),
        "suppliers": sorted(repr(node) for node in network.suppliers),
        "consumers": sorted(repr(node) for node in network.consumers),
        "base_supply": sorted((repr(node), float(value)) for node, value in network._base_supply.items()),
        "base_demand": sorted((repr(node), float(value)) for node, value in network._base_demand.items()),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _disruption_hash(network: Network) -> str:
    """Стабільний hash failed_nodes для перевірки парності I1."""
    payload = {
        "failed_nodes": sorted(repr(node) for node in network.failed_nodes),
        "irreversible_nodes": sorted(repr(node) for node in network.irreversible_nodes),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r_label(r: int | None) -> str:
    """Людиночитний label для r_visibility."""
    return "r_inf" if r is None else f"r{int(r)}"


def _execute_actions(network: Network, actions: dict[int, object | None]) -> int:
    """Виконує унікальні repair-дії цього timestep і повертає їх кількість."""
    repaired = 0
    for node in sorted({action for action in actions.values() if action is not None}):
        if node in network.failed_nodes:
            network.set_node_state(node, "active")
            repaired += 1
    return repaired


def _simulate_recovery(
    network: Network,
    F0: float,
    agents: list,
    coordinator,
    T_max: int,
) -> tuple[list[float], int]:
    """Симуляція M1: на кожному кроці K агентів пропонують repair-дії."""
    Q_curve: list[float] = [1.0]
    n_repaired_total = 0
    for t in range(1, T_max + 1):
        F_t = compute_max_flow(network)
        Q_curve.append(float(np.clip(F_t / F0 if F0 > 0.0 else 0.0, 0.0, 1.0)))
        if not network.failed_nodes:
            continue
        actions = coordinator.allocate(agents, network, set(network.failed_nodes))
        n_repaired_total += _execute_actions(network, actions)
    return Q_curve, n_repaired_total


def _run_single(
    cfg: dict,
    topology: str,
    strategy: str,
    K_agents: int,
    run_idx: int,
    coordination_mode: str | None = None,
    output_seed_offset: int = 0,
    cell_index: int | None = None,
    rl_model_seed_idx: int | None = None,
) -> tuple[dict, list[dict]]:
    """Один Monte Carlo прогін multi-agent M-серії."""
    series_id = str(cfg["series"])
    seed_base = int(cfg.get("random_seed_base", cfg["constants"].get("random_seed_base", 4400)))
    run_number = run_idx + 1 + output_seed_offset
    N_value = int(cfg["network"].get("N", 50))
    if cell_index is None:
        paired_id = f"{series_id}_{topology}_{run_number:05d}"
        if series_id == "M5ceiling":
            seed = int(cfg["constants"].get("test_seed_base", 8000)) + run_idx + output_seed_offset
        else:
            seed = _stable_seed(series_id, seed_base, topology, run_number)
    else:
        paired_id = f"{series_id}_{topology}_N{N_value}_{run_number:05d}"
        seed = seed_base + int(cell_index) * 1000 + run_number
    if series_id == "M5ceiling":
        network_seed = _stable_seed("M5", seed, "network")
        disruption_seed = _stable_seed("M5", seed, "disruption")
    else:
        network_seed = _stable_seed(seed, "network")
        disruption_seed = _stable_seed(seed, "disruption")

    net0 = _create_network(cfg, topology=topology, seed=network_seed)
    F0 = compute_max_flow(net0)
    mode = _coordination_mode(cfg, strategy, coordination_mode)
    run_id = f"{series_id}_{topology}_{strategy}_{run_number:05d}"
    if cell_index is not None:
        run_id = f"{series_id}_{topology}_N{N_value}_{strategy}_K{K_agents}_{run_number:05d}"
    elif series_id != "M1" or int(K_agents) != int(cfg["constants"].get("K_agents", K_agents)):
        run_id = f"{series_id}_{topology}_{strategy}_K{K_agents}_{run_number:05d}"
    if coordination_mode is not None:
        if cell_index is not None:
            run_id = f"{series_id}_{topology}_N{N_value}_{strategy}_{mode}_K{K_agents}_{run_number:05d}"
        else:
            run_id = f"{series_id}_{topology}_{strategy}_{mode}_K{K_agents}_{run_number:05d}"
    if strategy == "rl_policy" and rl_model_seed_idx is not None:
        run_id = run_id.replace("rl_policy", f"rl_policy_seed{int(rl_model_seed_idx)}")
    row = {
        "series": series_id,
        "article": int(cfg.get("article", 4)),
        "run_id": run_id,
        "paired_id": paired_id,
        "seed": seed,
        "topology": topology,
        "agent_strategy": strategy,
        "recovery_strategy": strategy,
        "coordination_mode": mode,
        "K_agents": int(K_agents),
        "r_visibility": int(cfg["constants"].get("r_visibility", 2)),
        "N": N_value,
        "H": int(cfg["network"].get("H", 4)),
        "b": int(cfg["network"].get("b", 3)),
        "K_res": int(cfg["network"].get("K_res", 2)),
        "p_v": float(cfg["disruption"].get("p_v", 0.20)),
        "T_max": int(cfg["simulation"].get("T_max", cfg["constants"].get("T", 100))),
        "mu": int(cfg["recovery"].get("mu", cfg["constants"].get("mu", 1))),
        "F0": float(F0),
        "rl_model_seed_idx": int(rl_model_seed_idx) if rl_model_seed_idx is not None else -1,
    }
    if F0 <= 0.0:
        row.update(_invalid_metrics("F0_zero"))
        return row, []

    disruption = NodeFailureDisruption(p_v=float(cfg["disruption"].get("p_v", 0.20)))
    net = disruption.apply(net0, seed=disruption_seed)
    row["N_failed_initial"] = int(len(net.failed_nodes))
    if not _has_active_supply_and_demand(net):
        row.update(_invalid_metrics("supply_or_demand_destroyed"))
        return row, []

    agents = _make_agents(
        cfg,
        strategy=strategy,
        network=net0,
        seed=seed,
        K_agents=K_agents,
        rl_model_seed_idx=rl_model_seed_idx,
    )
    coordinator = create_coordinator(row["coordination_mode"])
    Q_curve, n_repaired_total = _simulate_recovery(
        network=net,
        F0=F0,
        agents=agents,
        coordinator=coordinator,
        T_max=row["T_max"],
    )
    q_target = float(cfg["simulation"].get("Q_target_frac", 0.95)) * Q_curve[0]
    metrics = compute_integral_metrics(Q_curve, t_e=1, Q_target=q_target)
    row.update(metrics)
    row["valid"] = True
    row["exclusion_reason"] = ""
    row["non_recovered"] = bool(Q_curve[-1] < q_target)
    row["Q_final"] = float(Q_curve[-1])
    row["N_repaired_total"] = int(n_repaired_total)
    q_rows = [
        {"run_id": row["run_id"], "paired_id": paired_id, "topology": topology,
         "agent_strategy": strategy, "coordination_mode": mode,
         "K_agents": int(K_agents), "rl_model_seed_idx": row["rl_model_seed_idx"], "t": t, "Q": float(Q)}
        for t, Q in enumerate(Q_curve)
    ]
    return row, q_rows


def _invalid_metrics(reason: str) -> dict[str, object]:
    """Єдиний формат invalid row."""
    return {
        "valid": False,
        "exclusion_reason": reason,
        "non_recovered": False,
        "Q_final": float("nan"),
        "N_failed_initial": float("nan"),
        "N_repaired_total": 0,
        "rho": float("nan"),
        "R_rb": float("nan"),
        "R_rp": float("nan"),
        "R_inf": float("nan"),
    }


def _aggregate(raw_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Швидка агрегація для multi-agent raw output."""
    rows: list[dict[str, Any]] = []
    group_cols = ["topology", "agent_strategy", "coordination_mode", "K_agents"]
    if "N" in raw_df.columns and raw_df["N"].nunique(dropna=True) > 1:
        group_cols.insert(1, "N")
    for group_key, group_all in raw_df.groupby(group_cols):
        key = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        row = {
            "series": group_all["series"].iloc[0],
            "topology": key["topology"],
            "agent_strategy": key["agent_strategy"],
            "coordination_mode": key["coordination_mode"],
            "K_agents": int(key["K_agents"]),
            "n_total": int(len(group_all)),
            "n_valid": int(len(group)),
            "n_invalid": int(len(group_all) - len(group)),
            "rho_med": compute_rho_med(rho),
            "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=alpha),
            "rho_IQR": compute_rho_IQR(rho),
            "R_rb_med": float(group["R_rb"].median()) if len(group) else float("nan"),
        }
        if "N" in key:
            row["N"] = int(key["N"])
        rows.append(row)
    sort_cols = ["topology", "agent_strategy", "coordination_mode", "K_agents"]
    if "N" in raw_df.columns and raw_df["N"].nunique(dropna=True) > 1:
        sort_cols.insert(1, "N")
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def _aggregate_paired(raw_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Агрегація для I1: окремо за topology × r_label."""
    rows: list[dict[str, Any]] = []
    group_cols = ["topology", "agent_strategy", "coordination_mode", "K_agents", "r_label"]
    for (topology, strategy, coordination_mode, K_agents, r_label), group_all in raw_df.groupby(group_cols):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        rows.append({
            "series": group_all["series"].iloc[0],
            "topology": topology,
            "agent_strategy": strategy,
            "coordination_mode": coordination_mode,
            "K_agents": int(K_agents),
            "r_label": r_label,
            "r_visibility": group_all["r_visibility"].iloc[0],
            "n_total": int(len(group_all)),
            "n_valid": int(len(group)),
            "n_invalid": int(len(group_all) - len(group)),
            "rho_med": compute_rho_med(rho),
            "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=alpha),
            "rho_IQR": compute_rho_IQR(rho),
            "R_rb_med": float(group["R_rb"].median()) if len(group) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(
        ["topology", "agent_strategy", "coordination_mode", "K_agents", "r_label"]
    ).reset_index(drop=True)


def _load_r1_benchmarks() -> tuple[dict, dict]:
    """Завантажує R1 random/flow_greedy benchmark для metadata і sanity."""
    path = ROOT / "results" / "aggregated" / "article3" / "R1_descriptive_statistics.csv"
    r1 = pd.read_csv(path)
    fg = r1[r1["recovery_strategy"] == "flow_greedy"].set_index("topology")["rho_CVaR_0.10"].to_dict()
    rnd = r1[r1["recovery_strategy"] == "random"].set_index("topology")["rho_CVaR_0.10"].to_dict()
    return fg, rnd


def _save_outputs(
    cfg: dict,
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    q_rows: list[dict],
    output_dir: str | Path,
    elapsed: float,
) -> None:
    """Зберігає artifacts за output paths із config."""
    series_id = str(cfg["series"])
    raw_dir = _resolve_output_path(cfg["output"]["raw_dir"], output_dir)
    agg_dir = _resolve_output_path(cfg["output"]["aggregated_dir"], output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_dir / "raw_results.csv", index=False)
    pd.DataFrame(q_rows).to_parquet(raw_dir / "Q_curves.parquet", index=False)
    agg_df.to_csv(agg_dir / f"{series_id}_aggregated_results.csv", index=False)
    fg, rnd = _load_r1_benchmarks()
    topologies = sorted(raw_df["topology"].unique().tolist())
    strategies = sorted(raw_df["agent_strategy"].unique().tolist())
    K_values = sorted(int(value) for value in raw_df["K_agents"].unique().tolist())
    if bool(cfg.get("paired_design", False)):
        M_value: int | list[int] = int(cfg.get("n_runs", cfg["constants"].get("M", 200)))
        r_visibility_meta = cfg.get("varying", {}).get("r_visibility", [2, None])
    else:
        M_value = int(raw_df.groupby(["topology", "agent_strategy", "K_agents"]).size().max())
        r_visibility_meta = int(cfg["constants"].get("r_visibility", 2))
    metadata = {
        "article": int(cfg.get("article", 4)),
        "series_id": series_id,
        "description": cfg.get("description", ""),
        "depends_on": cfg.get("depends_on", []),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_hash": _current_git_hash(),
        "python_version": sys.version.split()[0],
        "N": (
            sorted(int(value) for value in raw_df["N"].dropna().unique().tolist())
            if "N" in raw_df.columns and raw_df["N"].nunique(dropna=True) > 1
            else int(cfg["network"].get("N", 50))
        ),
        "M": M_value,
        "T": int(cfg["simulation"].get("T_max", 100)),
        "mu": int(cfg["recovery"].get("mu", 1)),
        "K_agents": K_values[0] if len(K_values) == 1 else K_values,
        "K_values_tested": K_values,
        "r_visibility": r_visibility_meta,
        "disruption_type": cfg["disruption"].get("type", "node_failure"),
        "p_v": float(cfg["disruption"].get("p_v", 0.20)),
        "topologies": topologies,
        "strategies_tested": strategies,
        "coordination_modes": sorted(raw_df["coordination_mode"].unique().tolist()),
        "benchmark_flow_greedy_r1": fg,
        "benchmark_random_r1": rnd,
        "r_star_from_M0": 2,
        "elapsed_seconds": round(elapsed, 2),
    }
    metadata.update(cfg.get("metadata", {}))
    if "M_schedule" in cfg:
        metadata["M_schedule"] = {str(key): int(value) for key, value in cfg["M_schedule"].items()}
    with open(raw_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _resolve_output_path(config_path: str | Path, output_dir: str | Path) -> Path:
    """Розв'язує config output path з підтримкою tmp output roots у тестах."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "results":
        return Path(output_dir).joinpath(*parts[1:])
    return Path(output_dir) / path


def _filter_values(values: list, wanted: list | None) -> list:
    """Фільтрує список зі збереженням порядку config."""
    if wanted is None:
        return values
    wanted_set = {str(item) for item in wanted}
    return [value for value in values if str(value) in wanted_set]


def _series_values(cfg: dict) -> tuple[list[str], list[str], list[int], list[str | None]]:
    """Дістає topology, strategy, K і coordination_mode з constants/varying."""
    varying = cfg.get("varying", {})
    constants = cfg.get("constants", {})
    topologies = list(varying.get("topology", [cfg.get("network", {}).get("topology", "hierarchical")]))
    strategies = list(varying.get("agent_strategy", [constants.get("agent_strategy", "coord_greedy")]))
    K_values = [int(value) for value in varying.get("K_agents", [constants.get("K_agents", 3)])]
    if "coordination_mode" in varying:
        coordination_modes: list[str | None] = [str(value) for value in varying["coordination_mode"]]
    else:
        # Старі M1/M2 конфіги визначають mode через constants/coordination map.
        # None означає "взяти legacy mapping" і зберігає старі run_id.
        coordination_modes = [None]
    return topologies, strategies, K_values, coordination_modes


def run_series(
    cfg: dict,
    n_runs: int | None = None,
    output_dir: str | Path = "results",
    topology_filter: str | None = None,
    strategies_filter: list[str] | None = None,
    K_filter: list[int] | None = None,
    coordination_filter: list[str] | None = None,
    N_filter: list[int] | None = None,
) -> pd.DataFrame:
    """Запускає multi-agent M-серію за переданим config."""
    series_id = str(cfg["series"])
    has_scaling_schedule = "M_schedule" in cfg and "N" in cfg.get("varying", {})
    cell_index_offset = int(cfg.get("cell_index_offset", 0))
    if n_runs is None and not has_scaling_schedule:
        n_runs = int(cfg.get("n_runs", cfg["constants"].get("M", 200)))
    topologies, strategies, K_values, coordination_modes = _series_values(cfg)
    N_values = [int(value) for value in cfg.get("varying", {}).get("N", [cfg.get("network", {}).get("N", 50)])]
    if topology_filter:
        topologies = [topology for topology in topologies if topology == topology_filter]
    if N_filter:
        wanted_N = {int(value) for value in N_filter}
        N_values = [N for N in N_values if N in wanted_N]
    if strategies_filter:
        strategies = _filter_values(strategies, strategies_filter)
    if K_filter:
        K_values = [K for K in K_values if K in {int(value) for value in K_filter}]
    if coordination_filter:
        wanted_modes = {str(value) for value in coordination_filter}
        coordination_modes = [
            mode for mode in coordination_modes
            if mode is not None and str(mode) in wanted_modes
        ]
    if not topologies or not strategies or not K_values or not coordination_modes or not N_values:
        raise ValueError(f"{series_id} фільтри не залишили жодної конфігурації.")

    rl_seed_indices = list(range(int(cfg.get("constants", {}).get("n_train_seeds", 1)))) if series_id == "M5ceiling" else [None]

    schedule = {int(key): int(value) for key, value in cfg.get("M_schedule", {}).items()}
    if has_scaling_schedule:
        missing = [N for N in N_values if N not in schedule]
        if missing:
            raise ValueError(f"{series_id} M_schedule не містить N={missing}.")
        total = 0
        for _topology in topologies:
            for N in N_values:
                runs_for_cell = int(n_runs) if n_runs is not None else schedule[N]
                strategy_multiplier = sum(len(rl_seed_indices) if strategy == "rl_policy" else 1 for strategy in strategies)
                total += strategy_multiplier * len(K_values) * len(coordination_modes) * runs_for_cell
        print(f"\nСерія {series_id}: scaling N={N_values}, total прогонів={total}")
    else:
        strategy_multiplier = sum(len(rl_seed_indices) if strategy == "rl_policy" else 1 for strategy in strategies)
        n_configs = len(topologies) * strategy_multiplier * len(K_values) * len(coordination_modes)
        total = n_configs * int(n_runs)
        print(f"\nСерія {series_id}: {n_configs} конфігурацій × {n_runs} прогонів")
    start = time.perf_counter()
    raw_rows: list[dict] = []
    q_rows: list[dict] = []
    done = 0
    for cell_index, (topology, N_value) in enumerate(
        (topology, N_value) for topology in topologies for N_value in N_values
    ):
        effective_cell_index = cell_index + cell_index_offset
        cfg_cell = _cfg_with_network_N(cfg, N_value) if has_scaling_schedule else cfg
        runs_for_cell = int(n_runs) if n_runs is not None else schedule.get(N_value, int(n_runs or 0))
        for K_agents in K_values:
            for strategy in strategies:
                for coordination_mode in coordination_modes:
                    model_seed_values = rl_seed_indices if strategy == "rl_policy" else [None]
                    for rl_model_seed_idx in model_seed_values:
                        for run_idx in range(runs_for_cell):
                            row, q = _run_single(
                                cfg_cell,
                                topology=topology,
                                strategy=strategy,
                                K_agents=K_agents,
                                run_idx=run_idx,
                                coordination_mode=coordination_mode,
                                cell_index=effective_cell_index if has_scaling_schedule else None,
                                rl_model_seed_idx=rl_model_seed_idx,
                            )
                            raw_rows.append(row)
                            q_rows.extend(q)
                            done += 1
                            if done % max(1, total // 10) == 0 or done == total:
                                print(f"  progress: {done}/{total}")
    raw_df = pd.DataFrame(raw_rows)
    agg_df = _aggregate(raw_df, alpha=float(cfg["metrics"].get("alpha_cvar", 0.10)))
    elapsed = time.perf_counter() - start
    _save_outputs(cfg, raw_df, agg_df, q_rows, output_dir=output_dir, elapsed=elapsed)
    print(f"{series_id} complete in {elapsed:.1f}s; rows={len(raw_df)}, valid={int(raw_df['valid'].sum())}")
    return raw_df


def run_paired_series(
    cfg: dict,
    n_runs: int | None = None,
    output_dir: str | Path = "results",
    topology_filter: str | None = None,
) -> pd.DataFrame:
    """
    Запускає I1 у парному режимі: одна мережа+збій → r=2 і r=∞.

    Ключова гарантія Протоколу I1: для кожного paired_run_id створюється
    один net0 і один disrupted template; кожен r отримує власний net.copy(),
    тому recovery у першій гілці не може змінити другу.
    """
    series_id = str(cfg["series"])
    if not bool(cfg.get("paired_design", False)):
        raise ValueError("run_paired_series очікує config з paired_design: true.")
    if n_runs is None:
        n_runs = int(cfg.get("n_runs", cfg["constants"].get("M", 200)))

    varying = cfg.get("varying", {})
    topologies = list(varying.get("topology", [cfg.get("network", {}).get("topology", "hierarchical")]))
    if topology_filter:
        topologies = [topology for topology in topologies if topology == topology_filter]
    r_values = list(varying.get("r_visibility", [2, None]))
    if not topologies or not r_values:
        raise ValueError(f"{series_id} paired config не має topology або r_visibility.")

    strategy = str(cfg["constants"].get("agent_strategy", "coord_greedy"))
    mode = str(cfg["constants"].get("coordination_mode", "token"))
    K_agents = int(cfg["constants"].get("K_agents", 3))
    T_max = int(cfg["simulation"].get("T_max", cfg["constants"].get("T", 100)))
    p_v = float(cfg["disruption"].get("p_v", 0.20))
    seed_base = int(cfg.get("random_seed_base", cfg["constants"].get("random_seed_base", 5000)))

    n_configs = len(topologies) * len(r_values)
    print(f"\nСерія {series_id}: {len(topologies)} topology × {len(r_values)} r × {n_runs} пар")
    start = time.perf_counter()
    raw_rows: list[dict] = []
    q_rows: list[dict] = []
    total_rows = n_configs * n_runs
    done_rows = 0

    for topology in topologies:
        for run_idx in range(n_runs):
            run_number = run_idx + 1
            seed = _stable_seed(series_id, seed_base, topology, run_number)
            network_seed = _stable_seed(seed, "network")
            disruption_seed = _stable_seed(seed, "disruption")
            paired_id = f"{series_id}_pair_{topology}_{run_number:05d}"
            net0 = _create_network(cfg, topology=topology, seed=network_seed)
            network_hash = _network_hash(net0)
            F0 = compute_max_flow(net0)
            if F0 > 0.0:
                disrupted_template = NodeFailureDisruption(p_v=p_v).apply(net0, seed=disruption_seed)
                disruption_hash = _disruption_hash(disrupted_template)
                n_failed_initial = int(len(disrupted_template.failed_nodes))
                active_ok = _has_active_supply_and_demand(disrupted_template)
            else:
                disrupted_template = net0.copy()
                disruption_hash = ""
                n_failed_initial = 0
                active_ok = False

            for r in r_values:
                r_label = _r_label(r)
                run_id = f"{series_id}_{topology}_{strategy}_{mode}_K{K_agents}_{r_label}_{run_number:05d}"
                row = {
                    "series": series_id,
                    "article": int(cfg.get("article", 4)),
                    "run_id": run_id,
                    "paired_id": paired_id,
                    "paired_run_id": paired_id,
                    "seed": seed,
                    "network_seed": network_seed,
                    "disruption_seed": disruption_seed,
                    "network_hash": network_hash,
                    "disruption_hash": disruption_hash,
                    "topology": topology,
                    "agent_strategy": strategy,
                    "recovery_strategy": strategy,
                    "coordination_mode": mode,
                    "K_agents": K_agents,
                    "r_visibility": r,
                    "r_label": r_label,
                    "N": int(cfg["network"].get("N", 50)),
                    "H": int(cfg["network"].get("H", 4)),
                    "b": int(cfg["network"].get("b", 3)),
                    "K_res": int(cfg["network"].get("K_res", 2)),
                    "p_v": p_v,
                    "T_max": T_max,
                    "mu": int(cfg["recovery"].get("mu", cfg["constants"].get("mu", 1))),
                    "F0": float(F0),
                    "N_failed_initial": n_failed_initial,
                }
                if F0 <= 0.0:
                    row.update(_invalid_metrics("F0_zero"))
                    raw_rows.append(row)
                    done_rows += 1
                    continue
                if not active_ok:
                    row.update(_invalid_metrics("supply_or_demand_destroyed"))
                    raw_rows.append(row)
                    done_rows += 1
                    continue

                net = disrupted_template.copy()
                agents = _make_agents_with_r(
                    cfg=cfg,
                    strategy=strategy,
                    network=net0,
                    seed=seed,
                    K_agents=K_agents,
                    r=r,
                )
                coordinator = create_coordinator(mode)
                Q_curve, n_repaired_total = _simulate_recovery(
                    network=net,
                    F0=F0,
                    agents=agents,
                    coordinator=coordinator,
                    T_max=T_max,
                )
                q_target = float(cfg["simulation"].get("Q_target_frac", 0.95)) * Q_curve[0]
                metrics = compute_integral_metrics(Q_curve, t_e=1, Q_target=q_target)
                row.update(metrics)
                row["valid"] = True
                row["exclusion_reason"] = ""
                row["non_recovered"] = bool(Q_curve[-1] < q_target)
                row["Q_final"] = float(Q_curve[-1])
                row["N_repaired_total"] = int(n_repaired_total)
                raw_rows.append(row)
                q_rows.extend(
                    {
                        "run_id": run_id,
                        "paired_id": paired_id,
                        "paired_run_id": paired_id,
                        "topology": topology,
                        "agent_strategy": strategy,
                        "coordination_mode": mode,
                        "K_agents": K_agents,
                        "r_visibility": r,
                        "r_label": r_label,
                        "t": t,
                        "Q": float(Q),
                    }
                    for t, Q in enumerate(Q_curve)
                )
                done_rows += 1
                if done_rows % max(1, total_rows // 10) == 0 or done_rows == total_rows:
                    print(f"  progress: {done_rows}/{total_rows}")

    raw_df = pd.DataFrame(raw_rows)
    agg_df = _aggregate_paired(raw_df, alpha=float(cfg["metrics"].get("alpha_cvar", 0.10)))
    elapsed = time.perf_counter() - start
    _save_outputs(cfg, raw_df, agg_df, q_rows, output_dir=output_dir, elapsed=elapsed)
    print(f"{series_id} complete in {elapsed:.1f}s; rows={len(raw_df)}, valid={int(raw_df['valid'].sum())}")
    return raw_df


def run_m1_series(
    n_runs: int | None = None,
    output_dir: str | Path = "results",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    topology_filter: str | None = None,
    strategies_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper: запускає M1 через generic run_series()."""
    cfg = load_config("M1", config_dir=config_dir)
    return run_series(
        cfg,
        n_runs=n_runs,
        output_dir=output_dir,
        topology_filter=topology_filter,
        strategies_filter=strategies_filter,
        coordination_filter=None,
    )

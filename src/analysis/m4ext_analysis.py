"""
Аналіз M4ext: розширення масштабування hierarchical до N=160,200.

Реалізує: додаткову перевірку M4 без зміни базових результатів M4
(див. config/series_M4ext.yaml).

Приклад використання:
    >>> from src.analysis.m4ext_analysis import run_analysis
    >>> run_analysis()
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.disruptions.node_failure import NodeFailureDisruption
from src.experiments.multi_agent_runner import (
    _cfg_with_network_N,
    _create_network,
    _has_active_supply_and_demand,
    _stable_seed,
)
from src.metrics.integral import compute_integral_metrics
from src.metrics.stochastic import compute_rho_CVaR
from src.network.flow import compute_max_flow
from src.recovery.strategies import get_recovery_strategy


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "series_M4ext.yaml"
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M4ext"
M4_RAW_DIR = ROOT / "results" / "raw" / "article4" / "M4"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"

N_EXT = [160, 200]
N_FULL = [30, 50, 80, 120, 160, 200]
ALPHA = 0.10


def _load_config() -> dict:
    """Завантажує M4ext config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _simulate_single_agent(
    net,
    F0: float,
    strategy_name: str,
    T_max: int,
    rng: np.random.Generator,
) -> list[float]:
    """Симулює single-agent recovery за канонічною логікою статті 3."""
    strategy = get_recovery_strategy(strategy_name)
    Q_curve: list[float] = [1.0]
    for t in range(1, T_max + 1):
        F_t = compute_max_flow(net)
        Q_curve.append(float(np.clip(F_t / F0 if F0 > 0.0 else 0.0, 0.0, 1.0)))
        element = strategy.select_element_to_repair(net, t, rng=rng)
        if element is None:
            continue
        strategy.repair(net, element)
    return Q_curve


def _benchmark_seed(cfg: dict, N: int, run_idx: int) -> tuple[int, int]:
    """
    Повертає (seed, effective_cell_index) у тому самому namespace, що M4ext.

    M4 мав hierarchical N=30,50,80,120 як cell_index 0..3. M4ext продовжує
    ряд cell_index 4..5, щоб не перетинатися зі старими seed ranges.
    """
    seed_base = int(cfg["constants"]["random_seed_base"])
    cell_index_offset = int(cfg.get("cell_index_offset", 0))
    ext_index = N_EXT.index(int(N))
    effective_cell_index = cell_index_offset + ext_index
    run_number = run_idx + 1
    return seed_base + effective_cell_index * 1000 + run_number, effective_cell_index


def _run_benchmarks_if_needed(cfg: dict) -> pd.DataFrame:
    """
    Генерує flow_greedy/random benchmark для N=160,200, якщо його ще немає.

    R5 закінчується на N=120, тому ці дві точки рахуються наживо, але
    зберігаються тільки у namespaced M4ext artifacts.
    """
    path = RAW_DIR / "benchmark_raw_results.csv"
    if path.exists():
        return pd.read_csv(path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    q_rows: list[dict] = []
    T_max = int(cfg["simulation"].get("T_max", 100))
    q_target_frac = float(cfg["simulation"].get("Q_target_frac", 0.95))
    p_v = float(cfg["disruption"].get("p_v", 0.20))
    n_runs_by_N = {int(k): int(v) for k, v in cfg["M_schedule"].items()}
    started = time.perf_counter()

    total = sum(n_runs_by_N[N] for N in N_EXT) * 2
    done = 0
    for N in N_EXT:
        cfg_cell = _cfg_with_network_N(cfg, N)
        for run_idx in range(n_runs_by_N[N]):
            seed, effective_cell_index = _benchmark_seed(cfg, N, run_idx)
            network_seed = _stable_seed(seed, "network")
            disruption_seed = _stable_seed(seed, "disruption")

            net0 = _create_network(cfg_cell, topology="hierarchical", seed=network_seed)
            F0 = compute_max_flow(net0)
            disrupted_template = NodeFailureDisruption(p_v=p_v).apply(net0, seed=disruption_seed)
            is_valid = F0 > 0.0 and _has_active_supply_and_demand(disrupted_template)
            paired_id = f"M4ext_hierarchical_N{N}_{run_idx + 1:05d}"

            for strategy_name in ["flow_greedy", "random"]:
                run_id = f"M4ext_benchmark_hierarchical_N{N}_{strategy_name}_{run_idx + 1:05d}"
                base = {
                    "series": "M4ext_benchmark",
                    "article": 4,
                    "run_id": run_id,
                    "paired_id": paired_id,
                    "seed": seed,
                    "effective_cell_index": effective_cell_index,
                    "topology": "hierarchical",
                    "recovery_strategy": strategy_name,
                    "N": N,
                    "H": int(cfg["network"].get("H", 4)),
                    "b": int(cfg["network"].get("b", 3)),
                    "K_res": int(cfg["network"].get("K_res", 2)),
                    "p_v": p_v,
                    "T_max": T_max,
                    "mu": int(cfg["recovery"].get("mu", 1)),
                    "F0": float(F0),
                    "N_failed_initial": int(len(disrupted_template.failed_nodes)),
                }
                if not is_valid:
                    rows.append({
                        **base,
                        "rho": float("nan"),
                        "R_rb": float("nan"),
                        "R_rp": float("nan"),
                        "R_inf": float("nan"),
                        "valid": False,
                        "exclusion_reason": "F0_zero" if F0 <= 0.0 else "supply_or_demand_destroyed",
                        "non_recovered": False,
                        "Q_final": float("nan"),
                    })
                    done += 1
                    continue

                net = disrupted_template.copy()
                rng = np.random.default_rng(_stable_seed(seed, strategy_name, "recovery"))
                Q_curve = _simulate_single_agent(net, F0, strategy_name, T_max, rng)
                q_target = q_target_frac * Q_curve[0]
                metrics = compute_integral_metrics(Q_curve, t_e=1, Q_target=q_target)
                rows.append({
                    **base,
                    **metrics,
                    "valid": True,
                    "exclusion_reason": "",
                    "non_recovered": bool(Q_curve[-1] < q_target),
                    "Q_final": float(Q_curve[-1]),
                })
                q_rows.extend(
                    {
                        "run_id": run_id,
                        "paired_id": paired_id,
                        "topology": "hierarchical",
                        "N": N,
                        "recovery_strategy": strategy_name,
                        "t": t,
                        "Q": float(Q),
                    }
                    for t, Q in enumerate(Q_curve)
                )
                done += 1
                if done % max(1, total // 10) == 0 or done == total:
                    print(f"  M4ext benchmark progress: {done}/{total}")

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(path, index=False)
    pd.DataFrame(q_rows).to_parquet(RAW_DIR / "benchmark_Q_curves.parquet", index=False)
    metadata = {
        "series": "M4ext_benchmark",
        "description": "Live flow_greedy/random benchmark for hierarchical N=160,200",
        "seed_namespace": "same as M4ext, cell_index_offset=4",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    with open(RAW_DIR / "benchmark_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return benchmark


def _cvar_by_group(df: pd.DataFrame, strategy_col: str) -> pd.DataFrame:
    """Обчислює rho_CVaR по N × strategy."""
    rows: list[dict] = []
    for (N, strategy), group_all in df.groupby(["N", strategy_col]):
        group = group_all[group_all["valid"].astype(bool)]
        rows.append({
            "N": int(N),
            "strategy": strategy,
            "n_total": int(len(group_all)),
            "n_valid": int(len(group)),
            "rho_CVaR_0.10": compute_rho_CVaR(group["rho"].astype(float), alpha=ALPHA),
            "F0_median": float(group["F0"].median()) if len(group) else float("nan"),
        })
    return pd.DataFrame(rows)


def _build_scaling_table(m4ext_raw: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Створює M4ext_scaling.csv для N=160,200."""
    coord = _cvar_by_group(m4ext_raw, "agent_strategy")
    coord = coord[coord["strategy"] == "coord_greedy"].rename(
        columns={"rho_CVaR_0.10": "rho_cvar_coord", "n_valid": "n_valid_coord"}
    )
    bench = _cvar_by_group(benchmark, "recovery_strategy")
    pivot = bench.pivot_table(index="N", columns="strategy", values="rho_CVaR_0.10", aggfunc="first").reset_index()
    counts = bench.pivot_table(index="N", columns="strategy", values="n_valid", aggfunc="first").reset_index()
    out = coord[["N", "n_valid_coord", "rho_cvar_coord", "F0_median"]].merge(pivot, on="N", how="left")
    out = out.merge(counts, on="N", how="left", suffixes=("", "_n_valid"))
    out = out.rename(columns={
        "flow_greedy": "rho_cvar_flowgreedy",
        "random": "rho_cvar_random",
        "flow_greedy_n_valid": "n_valid_flowgreedy",
        "random_n_valid": "n_valid_random",
        "F0_median": "F0_median_coord",
    })
    out["topology"] = "hierarchical"
    # M4 convention: negative Δρ_dec means multi-agent coord_greedy exceeds flow_greedy.
    out["delta_rho_dec"] = out["rho_cvar_flowgreedy"] - out["rho_cvar_coord"]
    out["delta_rho_fg"] = out["rho_cvar_flowgreedy"] - out["rho_cvar_random"]
    return out[
        [
            "N", "topology", "rho_cvar_coord", "rho_cvar_flowgreedy", "rho_cvar_random",
            "delta_rho_dec", "delta_rho_fg", "F0_median_coord",
            "n_valid_coord", "n_valid_flowgreedy", "n_valid_random",
        ]
    ].sort_values("N").reset_index(drop=True)


def _full_hierarchical_scaling(ext: pd.DataFrame) -> pd.DataFrame:
    """Об'єднує старі точки M4 з M4ext."""
    old = pd.read_csv(AGG_DIR / "M4_scaling.csv")
    old_h = old[old["topology"] == "hierarchical"].copy()
    old_part = pd.DataFrame({
        "N": old_h["N"].astype(int),
        "topology": "hierarchical",
        "rho_cvar_coord": old_h["rho_best"].astype(float),
        "rho_cvar_flowgreedy": old_h["rho_fg_R5"].astype(float),
        "rho_cvar_random": old_h["rho_random_R5"].astype(float),
        "delta_rho_dec": old_h["d_rho_dec"].astype(float),
        "delta_rho_fg": old_h["d_rho_fg"].astype(float),
        "source": "M4",
    })
    ext_part = ext.copy()
    ext_part["source"] = "M4ext"
    return pd.concat([old_part, ext_part[old_part.columns]], ignore_index=True).sort_values("N").reset_index(drop=True)


def _sanity_checks(full: pd.DataFrame, m4ext_raw: pd.DataFrame) -> pd.DataFrame:
    """Виконує sanity checks для M4ext."""
    old_raw = pd.read_csv(M4_RAW_DIR / "raw_results.csv")
    old_h = old_raw[old_raw["topology"] == "hierarchical"]
    new_h = m4ext_raw[m4ext_raw["topology"] == "hierarchical"]
    seed_overlap = set(old_h["seed"].astype(int)) & set(new_h["seed"].astype(int))

    f0_old = old_h[old_h["valid"].astype(bool)].groupby("N")["F0"].median()
    f0_new = new_h[new_h["valid"].astype(bool)].groupby("N")["F0"].median()
    f0_full = pd.concat([f0_old, f0_new]).sort_index()
    f0_growth = bool(np.all(np.diff(f0_full.loc[N_FULL].to_numpy(dtype=float)) > 0))

    delta_dec_negative = bool((full["delta_rho_dec"] < 0.0).all())
    delta_fg = full["delta_rho_fg"].to_numpy(dtype=float)
    delta_fg_positive = bool((delta_fg > 0.0).all())
    delta_fg_non_decreasing = bool(np.all(np.diff(delta_fg) >= -1e-12))

    return pd.DataFrame([
        {
            "check": "seed namespace no overlap with base M4 hierarchical",
            "passed": len(seed_overlap) == 0,
            "details": f"overlap_count={len(seed_overlap)}",
        },
        {
            "check": "F0 median grows with N",
            "passed": f0_growth,
            "details": "; ".join(f"N={int(N)} F0={float(v):.1f}" for N, v in f0_full.loc[N_FULL].items()),
        },
        {
            "check": "delta_rho_dec remains negative",
            "passed": delta_dec_negative,
            "details": ", ".join(f"{v:.6f}" for v in full["delta_rho_dec"]),
        },
        {
            "check": "delta_rho_fg positive",
            "passed": delta_fg_positive,
            "details": ", ".join(f"{v:.6f}" for v in full["delta_rho_fg"]),
        },
        {
            "check": "delta_rho_fg non-decreasing",
            "passed": delta_fg_non_decreasing,
            "details": ", ".join(f"{v:.6f}" for v in full["delta_rho_fg"]),
        },
    ])


def _write_figure(full: pd.DataFrame) -> Path:
    """Створює окрему Fig.8b для hierarchical M4 extension."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "M4ext_hierarchical_delta_rho_dec.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    old = full[full["source"] == "M4"]
    new = full[full["source"] == "M4ext"]
    ax.plot(full["N"], full["delta_rho_dec"], color="#1f77b4", linewidth=2.2, label="Δρ_dec")
    ax.scatter(old["N"], old["delta_rho_dec"], color="#1f77b4", s=55, label="M4 base")
    ax.scatter(new["N"], new["delta_rho_dec"], color="#d62728", s=72, marker="D", label="M4ext")
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Δρ_dec = flow_greedy − coord_greedy")
    ax.set_title("M4 extension: hierarchical decentralization price")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def _write_report(full: pd.DataFrame, ext: pd.DataFrame, sanity: pd.DataFrame, spearman_r: float) -> Path:
    """Пише короткий TXT-звіт для M4ext."""
    report_path = AGG_DIR / "M4ext_report.txt"
    verdict = "continues_deepening"
    if ext["delta_rho_dec"].iloc[-1] > ext["delta_rho_dec"].iloc[0]:
        verdict = "saturation_or_reversal"
    lines = [
        "M4ext: hierarchical N=160,200",
        "=" * 60,
        "",
        "Scaling table:",
        ext.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "",
        "Full hierarchical delta_rho_dec:",
        full[["N", "delta_rho_dec", "delta_rho_fg", "source"]].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        ),
        "",
        f"Spearman r(N, delta_rho_dec) full range: {spearman_r:.6f}",
        "",
        "Sanity checks:",
        sanity.to_string(index=False),
        "",
        f"Verdict: {verdict}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_analysis() -> dict[str, object]:
    """Запускає повний M4ext analysis pipeline."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    raw_path = RAW_DIR / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Спершу запустіть python main.py --series M4ext: {raw_path}")

    m4ext_raw = pd.read_csv(raw_path)
    benchmark = _run_benchmarks_if_needed(cfg)
    ext_scaling = _build_scaling_table(m4ext_raw, benchmark)
    ext_scaling.to_csv(AGG_DIR / "M4ext_scaling.csv", index=False)

    full = _full_hierarchical_scaling(ext_scaling)
    full.to_csv(AGG_DIR / "M4ext_full_hierarchical_scaling.csv", index=False)
    spearman_r = float(spearmanr(full["N"].astype(int), full["delta_rho_dec"].astype(float)).statistic)
    pd.DataFrame([{
        "topology": "hierarchical",
        "N_values": ",".join(str(N) for N in N_FULL),
        "spearman_r_delta_rho_dec": spearman_r,
    }]).to_csv(AGG_DIR / "M4ext_spearman.csv", index=False)

    sanity = _sanity_checks(full, m4ext_raw)
    sanity.to_csv(AGG_DIR / "M4ext_sanity_checks.csv", index=False)
    figure_path = _write_figure(full)
    report_path = _write_report(full, ext_scaling, sanity, spearman_r)
    print(f"M4ext analysis saved: {AGG_DIR / 'M4ext_scaling.csv'}")
    print(f"Figure saved: {figure_path}")
    print(f"Report saved: {report_path}")
    return {
        "ext_scaling": ext_scaling,
        "full_scaling": full,
        "sanity": sanity,
        "spearman_r": spearman_r,
        "figure_path": figure_path,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_analysis()

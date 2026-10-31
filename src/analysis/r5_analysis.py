"""
Аналіз серії R5 для статті 3.

Реалізує: Protocol R5 + Math Spec Article 3 Patch v3.4
(див. docs/Протокол_R5.docx та docs/Math_Spec_Article_3_Patch_v3_4.docx).

Приклад використання:
    python -m src.analysis.r5_analysis
"""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "R5"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

TOPOLOGIES = ["hierarchical", "dense"]
STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]
N_VALUES = [30, 50, 80, 120]
ALPHA = 0.10
BOOTSTRAP_B = 1000
PERMUTATION_B = 10_000
PAIRWISE_ALPHA = 0.05 / 48
R3_REFERENCE_CVAR_GAIN = {"hierarchical": 0.0349, "dense": 0.0188}
R3_CONTINUITY_TOL = 0.02


def _clean(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
    """Повертає float-масив без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap/permutation."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_ci(
    values: pd.Series | list[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    B: int = BOOTSTRAP_B,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI для довільної статистики."""
    arr = _clean(values)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = statistic(sample)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r через ранги; використовується у permutation test."""
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_spearman_p(
    x_values: pd.Series | np.ndarray,
    y_values: pd.Series | np.ndarray,
    B: int = PERMUTATION_B,
    seed: int = 42,
) -> tuple[float, float]:
    """Двосторонній permutation p-value для Spearman r."""
    x = _clean(x_values)
    y = _clean(y_values)
    if len(x) != len(y):
        raise ValueError("x та y мають мати однакову довжину.")
    if len(x) < 3:
        return (float("nan"), float("nan"))
    observed = _spearman_fast(x, y)
    if math.isnan(observed):
        return (observed, float("nan"))
    rng = np.random.default_rng(seed)
    rx = rankdata(x)
    ry = rankdata(y)
    exceed = 0
    for _ in range(B):
        permuted = rng.permutation(ry)
        r_perm = float(np.corrcoef(rx, permuted)[0, 1])
        if abs(r_perm) >= abs(observed) - 1e-15:
            exceed += 1
    return (observed, float((exceed + 1) / (B + 1)))


def load_raw() -> pd.DataFrame:
    """Завантажує raw_results.csv R5."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"R5 raw_results.csv not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("R5 raw_results.csv is empty.")
    raw["valid"] = raw["valid"].astype(bool)
    raw["N"] = raw["N"].astype(int)
    if "runtime_seconds" not in raw.columns:
        raw["runtime_seconds"] = np.nan
    return raw


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для topology × N × strategy."""
    rows: list[dict] = []
    for (topology, N, strategy), group_all in raw.groupby(
        ["topology", "N", "recovery_strategy"]
    ):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)
        rho_med_ci = bootstrap_ci(
            rho, np.median, seed=stable_seed("R5", topology, N, strategy, "rho_med")
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("R5", topology, N, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho, compute_rho_IQR, seed=stable_seed("R5", topology, N, strategy, "rho_iqr")
        )
        rows.append(
            {
                "topology": topology,
                "N": int(N),
                "recovery_strategy": strategy,
                "n_total": int(len(group_all)),
                "n_valid": int(len(group)),
                "n_invalid": int(len(group_all) - len(group)),
                "excl_pct": float((len(group_all) - len(group)) / len(group_all) * 100.0)
                if len(group_all) else float("nan"),
                "rho_med": compute_rho_med(rho),
                "rho_med_ci_low": rho_med_ci[0],
                "rho_med_ci_high": rho_med_ci[1],
                "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=ALPHA),
                "rho_CVaR_ci_low": rho_cvar_ci[0],
                "rho_CVaR_ci_high": rho_cvar_ci[1],
                "rho_IQR": compute_rho_IQR(rho),
                "rho_IQR_ci_low": rho_iqr_ci[0],
                "rho_IQR_ci_high": rho_iqr_ci[1],
                "R_rb_med": float(np.median(r_rb)) if len(r_rb) else float("nan"),
                "F0_median": float(group["F0"].median()) if len(group) else float("nan"),
                "F0_mean": float(group["F0"].mean()) if len(group) else float("nan"),
                "runtime_seconds_mean": float(group_all["runtime_seconds"].mean()),
                "runtime_seconds_max": float(group_all["runtime_seconds"].max()),
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0)
                if len(group) else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["topology", "N", "recovery_strategy"])
        .reset_index(drop=True)
    )


def _tail_rows_for_cvar(raw_group: pd.DataFrame) -> pd.DataFrame:
    """Вибирає нижній α-хвіст ρ окремо для кожного N."""
    parts: list[pd.DataFrame] = []
    for _, group in raw_group.groupby("N"):
        valid = group[group["valid"]].sort_values("rho", ascending=True)
        n_tail = max(1, int(np.floor(ALPHA * len(valid))))
        parts.append(valid.head(n_tail))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def raw_spearman_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(N, metric) на сирих даних для кожної topology × strategy."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        for strategy in STRATEGIES:
            group = raw[
                (raw["valid"])
                & (raw["topology"] == topology)
                & (raw["recovery_strategy"] == strategy)
            ].copy()
            for metric, source in [
                ("rho_med", group),
                ("rho_CVaR_0.10", _tail_rows_for_cvar(group)),
            ]:
                if source.empty:
                    r_value, p_perm, scipy_p, n_raw = np.nan, np.nan, np.nan, 0
                else:
                    x = source["N"].astype(float).to_numpy()
                    y = source["rho"].astype(float).to_numpy()
                    r_value, p_perm = permutation_spearman_p(
                        x, y, seed=stable_seed("R5", topology, strategy, metric)
                    )
                    scipy_p = float(spearmanr(x, y).pvalue)
                    n_raw = int(len(source))
                rows.append(
                    {
                        "topology": topology,
                        "recovery_strategy": strategy,
                        "metric": metric,
                        "n_raw": n_raw,
                        "spearman_r": r_value,
                        "p_perm": p_perm,
                        "p_scipy": scipy_p,
                    }
                )
    return pd.DataFrame(rows)


def strategy_gain(desc: pd.DataFrame) -> pd.DataFrame:
    """Δ metrics between intelligent strategies and random by topology × N."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        for N in N_VALUES:
            sub = desc[(desc["topology"] == topology) & (desc["N"] == N)]
            by_strategy = {row["recovery_strategy"]: row for _, row in sub.iterrows()}
            if "random" not in by_strategy:
                continue
            random_row = by_strategy["random"]
            out = {"topology": topology, "N": int(N)}
            for strategy in ["centrality_betweenness", "flow_greedy"]:
                if strategy not in by_strategy:
                    continue
                current = by_strategy[strategy]
                prefix = "centrality" if strategy == "centrality_betweenness" else "flow"
                out[f"delta_rho_med_{prefix}_minus_random"] = (
                    float(current["rho_med"]) - float(random_row["rho_med"])
                )
                out[f"delta_rho_CVaR_{prefix}_minus_random"] = (
                    float(current["rho_CVaR_0.10"]) - float(random_row["rho_CVaR_0.10"])
                )
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["topology", "N"]).reset_index(drop=True)


def gain_spearman_analysis(gain: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(N, Δmetric) для strategy gain кривих."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        sub = gain[gain["topology"] == topology]
        for strategy_key in ["centrality", "flow"]:
            for metric_key, col in [
                ("delta_rho_med", f"delta_rho_med_{strategy_key}_minus_random"),
                ("delta_rho_CVaR", f"delta_rho_CVaR_{strategy_key}_minus_random"),
            ]:
                if col not in sub:
                    continue
                r_value, p_perm = permutation_spearman_p(
                    sub["N"].astype(float),
                    sub[col].astype(float),
                    seed=stable_seed("R5", topology, strategy_key, metric_key, "gain"),
                )
                rows.append(
                    {
                        "topology": topology,
                        "strategy_gain": f"{strategy_key}_minus_random",
                        "metric": metric_key,
                        "spearman_r": r_value,
                        "p_perm": p_perm,
                    }
                )
    return pd.DataFrame(rows)


def paired_wilcoxon_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon within topology × N (48 tests)."""
    rows: list[dict] = []
    strategy_pairs = [
        ("random", "centrality_betweenness"),
        ("random", "flow_greedy"),
        ("centrality_betweenness", "flow_greedy"),
    ]
    for topology in TOPOLOGIES:
        for N in N_VALUES:
            sub = raw[(raw["valid"]) & (raw["topology"] == topology) & (raw["N"] == N)]
            for left, right in strategy_pairs:
                left_df = sub[sub["recovery_strategy"] == left][["paired_id", "rho", "R_rb"]]
                right_df = sub[sub["recovery_strategy"] == right][["paired_id", "rho", "R_rb"]]
                merged = left_df.merge(
                    right_df, on="paired_id", suffixes=("_left", "_right")
                )
                for metric in ["rho_med", "rho_CVaR_0.10"]:
                    source = merged.copy()
                    if metric == "rho_CVaR_0.10" and not source.empty:
                        n_tail = max(1, int(np.floor(ALPHA * len(source))))
                        left_tail = source.nsmallest(n_tail, "rho_left")["paired_id"]
                        right_tail = source.nsmallest(n_tail, "rho_right")["paired_id"]
                        source = source[
                            source["paired_id"].isin(set(left_tail) | set(right_tail))
                        ]
                    if len(source) < 2:
                        stat, p_value, delta = np.nan, np.nan, np.nan
                    else:
                        try:
                            stat, p_value = wilcoxon(
                                source["rho_right"], source["rho_left"], alternative="greater"
                            )
                        except ValueError:
                            stat, p_value = 0.0, 1.0
                        delta = float(source["rho_right"].mean() - source["rho_left"].mean())
                    rows.append(
                        {
                            "topology": topology,
                            "N": int(N),
                            "left_strategy": left,
                            "right_strategy": right,
                            "metric": metric,
                            "n_pairs": int(len(source)),
                            "mean_delta_right_minus_left": delta,
                            "wilcoxon_stat": float(stat),
                            "p_value": float(p_value),
                        }
                    )
    out = pd.DataFrame(rows)
    out["p_adj_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    out["significant"] = out["p_adj_bonferroni"] < 0.05
    return out


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """R_rb має бути однаковим по strategies у межах topology × N × paired_id."""
    rows: list[dict] = []
    for (topology, N, paired_id), group in raw[raw["valid"]].groupby(
        ["topology", "N", "paired_id"]
    ):
        vals = group["R_rb"].astype(float)
        rows.append(
            {
                "topology": topology,
                "N": int(N),
                "paired_id": paired_id,
                "n_strategies": int(group["recovery_strategy"].nunique()),
                "R_rb_min": float(vals.min()),
                "R_rb_max": float(vals.max()),
                "spread": float(vals.max() - vals.min()),
                "passed": bool((vals.max() - vals.min()) <= 1e-9),
            }
        )
    return pd.DataFrame(rows)


def f0_scaling(raw: pd.DataFrame) -> pd.DataFrame:
    """F0 scaling sanity check by topology and N."""
    dedup = raw[raw["valid"]].drop_duplicates(["topology", "N", "paired_id"])
    rows: list[dict] = []
    for (topology, N), group in dedup.groupby(["topology", "N"]):
        rows.append(
            {
                "topology": topology,
                "N": int(N),
                "n_pairs": int(len(group)),
                "F0_mean": float(group["F0"].mean()),
                "F0_median": float(group["F0"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["topology", "N"]).reset_index(drop=True)
    out["monotonic_passed"] = False
    for topology, group in out.groupby("topology"):
        passed = bool(np.all(np.diff(group.sort_values("N")["F0_mean"].to_numpy()) > 0.0))
        out.loc[out["topology"] == topology, "monotonic_passed"] = passed
    return out


def performance_by_N(raw: pd.DataFrame) -> pd.DataFrame:
    """Computational performance log by N."""
    return (
        raw.groupby("N")["runtime_seconds"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": "runtime_mean_s",
                "median": "runtime_median_s",
                "max": "runtime_max_s",
            }
        )
    )


def r3_continuity_check(gain: pd.DataFrame) -> pd.DataFrame:
    """Compares R5 N=50 flow_greedy gain with R3 reference values."""
    rows: list[dict] = []
    for topology, reference in R3_REFERENCE_CVAR_GAIN.items():
        sub = gain[(gain["topology"] == topology) & (gain["N"] == 50)]
        observed = (
            float(sub["delta_rho_CVaR_flow_minus_random"].iloc[0])
            if not sub.empty else float("nan")
        )
        diff = observed - reference
        rows.append(
            {
                "topology": topology,
                "R3_reference_delta_CVaR": reference,
                "R5_N50_delta_CVaR": observed,
                "difference": diff,
                "tolerance": R3_CONTINUITY_TOL,
                "passed": bool(abs(diff) <= R3_CONTINUITY_TOL),
            }
        )
    return pd.DataFrame(rows)


def hr1_robustness_status(gain: pd.DataFrame, gain_spearman: pd.DataFrame, wilcoxon_df: pd.DataFrame) -> str:
    """Formal HR1 robustness decision according to Protocol R5."""
    delta_col = "delta_rho_CVaR_flow_minus_random"
    all_positive = bool((gain[delta_col] > 0.0).all())
    flow_tests = wilcoxon_df[
        (wilcoxon_df["left_strategy"] == "random")
        & (wilcoxon_df["right_strategy"] == "flow_greedy")
        & (wilcoxon_df["metric"] == "rho_CVaR_0.10")
    ]
    all_significant = bool((flow_tests["p_adj_bonferroni"] < 0.05).all())
    flow_gain_spearman = gain_spearman[
        (gain_spearman["strategy_gain"] == "flow_minus_random")
        & (gain_spearman["metric"] == "delta_rho_CVaR")
    ]
    stable_scale = bool(
        flow_gain_spearman["spearman_r"].between(-0.5, 0.5, inclusive="both").all()
    )
    if all_positive and all_significant and stable_scale:
        return "ROBUST"
    if all_positive:
        return "PARTIALLY_ROBUST"
    return "NOT_ROBUST"


def _strategy_label(strategy: str, lang: str) -> str:
    """Human-readable strategy label."""
    labels = {
        "random": {"uk": "random", "en": "random"},
        "centrality_betweenness": {"uk": "centrality", "en": "centrality"},
        "flow_greedy": {"uk": "flow_greedy", "en": "flow_greedy"},
    }
    return labels.get(strategy, {"uk": strategy, "en": strategy})[lang]


def _topology_label(topology: str, lang: str) -> str:
    """Human-readable topology label."""
    labels = {
        "hierarchical": {"uk": "Ієрархічна", "en": "Hierarchical"},
        "dense": {"uk": "Щільна", "en": "Dense"},
    }
    return labels.get(topology, {"uk": topology, "en": topology})[lang]


def plot_metric_vs_N(desc: pd.DataFrame, metric: str, lang: str = "uk") -> plt.Figure:
    """Line plot of rho_med or rho_CVaR vs N."""
    key = "R5.rho_vs_N" if metric == "rho_med" else "R5.cvar_vs_N"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, topology in zip(axes, TOPOLOGIES):
        sub_top = desc[desc["topology"] == topology]
        for strategy in STRATEGIES:
            sub = sub_top[sub_top["recovery_strategy"] == strategy].sort_values("N")
            if sub.empty:
                continue
            if metric == "rho_med":
                low_col, high_col = "rho_med_ci_low", "rho_med_ci_high"
            else:
                low_col, high_col = "rho_CVaR_ci_low", "rho_CVaR_ci_high"
            ax.plot(sub["N"], sub[metric], marker="o", label=_strategy_label(strategy, lang))
            ax.fill_between(sub["N"], sub[low_col], sub[high_col], alpha=0.15)
        ax.set_title(_topology_label(topology, lang))
        ax.set_xlabel(label(f"{key}.x_label", lang))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(label(f"{key}.y_label", lang))
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(label(f"{key}.title", lang))
    fig.tight_layout()
    return fig


def plot_strategy_gain_vs_N(gain: pd.DataFrame, lang: str = "uk") -> plt.Figure:
    """Flow-greedy CVaR gain vs N."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for topology in TOPOLOGIES:
        sub = gain[gain["topology"] == topology].sort_values("N")
        ax.plot(
            sub["N"],
            sub["delta_rho_CVaR_flow_minus_random"],
            marker="o",
            label=_topology_label(topology, lang),
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_title(label("R5.strategy_gain_vs_N.title", lang))
    ax.set_xlabel(label("R5.strategy_gain_vs_N.x_label", lang))
    ax.set_ylabel(label("R5.strategy_gain_vs_N.y_label", lang))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_F0_scaling(f0_df: pd.DataFrame, lang: str = "uk") -> plt.Figure:
    """F0 scaling sanity check."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for topology in TOPOLOGIES:
        sub = f0_df[f0_df["topology"] == topology].sort_values("N")
        ax.plot(sub["N"], sub["F0_mean"], marker="o", label=_topology_label(topology, lang))
    ax.set_title(label("R5.F0_scaling.title", lang))
    ax.set_xlabel(label("R5.F0_scaling.x_label", lang))
    ax.set_ylabel(label("R5.F0_scaling.y_label", lang))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figures(desc: pd.DataFrame, gain: pd.DataFrame, f0_df: pd.DataFrame) -> list[Path]:
    """Зберігає 4 типи рисунків × 2 мови."""
    outputs: list[Path] = []
    outputs += save_dual_lang(
        lambda lang: plot_metric_vs_N(desc, metric="rho_med", lang=lang),
        str(FIG_DIR / "R5_rho_vs_N_{lang}.png"),
    )
    outputs += save_dual_lang(
        lambda lang: plot_metric_vs_N(desc, metric="rho_CVaR_0.10", lang=lang),
        str(FIG_DIR / "R5_cvar_vs_N_{lang}.png"),
    )
    outputs += save_dual_lang(
        lambda lang: plot_strategy_gain_vs_N(gain, lang=lang),
        str(FIG_DIR / "R5_strategy_gain_vs_N_{lang}.png"),
    )
    outputs += save_dual_lang(
        lambda lang: plot_F0_scaling(f0_df, lang=lang),
        str(FIG_DIR / "R5_F0_scaling_{lang}.png"),
    )
    return outputs


def make_report(
    desc: pd.DataFrame,
    raw_spearman: pd.DataFrame,
    gain: pd.DataFrame,
    gain_spearman: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    r_rb_check: pd.DataFrame,
    f0_df: pd.DataFrame,
    performance: pd.DataFrame,
    continuity: pd.DataFrame,
    status: str,
) -> str:
    """Builds R5 text report."""
    lines: list[str] = []
    lines.append("R5 REPORT — SCALE INTERACTION (HR1 ROBUSTNESS ACROSS N)")
    lines.append("=" * 78)
    lines.append("Design: topology ∈ {hierarchical, dense}; N ∈ {30,50,80,120}; "
                 "strategy ∈ {random, centrality_betweenness, flow_greedy}; M=100.")
    lines.append("Disruption: node_failure, p_v=0.20; recovery μ=1; T_max=100.")
    lines.append("")

    lines.append("DESCRIPTIVE STATISTICS")
    lines.append("-" * 78)
    lines.append(
        desc[[
            "topology", "N", "recovery_strategy", "n_valid", "n_invalid",
            "rho_med", "rho_CVaR_0.10", "R_rb_med", "F0_median",
            "runtime_seconds_mean",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    lines.append("")

    lines.append("RAW SPEARMAN r(N, metric)")
    lines.append("-" * 78)
    lines.append(
        raw_spearman.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    lines.append("")

    lines.append("STRATEGY GAIN: flow_greedy − random")
    lines.append("-" * 78)
    lines.append(
        gain[[
            "topology", "N", "delta_rho_med_flow_minus_random",
            "delta_rho_CVaR_flow_minus_random",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    lines.append("")

    lines.append("SPEARMAN r(N, strategy gain)")
    lines.append("-" * 78)
    lines.append(gain_spearman.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    lines.append("PAIRWISE WILCOXON TESTS")
    lines.append("-" * 78)
    lines.append(f"Bonferroni alpha: {PAIRWISE_ALPHA:.6f}; p_adj < 0.05 used for HR1 robustness.")
    lines.append(
        wilcoxon_df[[
            "topology", "N", "left_strategy", "right_strategy", "metric",
            "n_pairs", "mean_delta_right_minus_left", "p_value",
            "p_adj_bonferroni", "significant",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    lines.append("")

    lines.append("F0 SCALING SANITY CHECK")
    lines.append("-" * 78)
    lines.append(f0_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    lines.append("R3 CONTINUITY CHECK")
    lines.append("-" * 78)
    lines.append(continuity.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    lines.append("R_rb SANITY CHECK")
    lines.append("-" * 78)
    lines.append(
        f"passed={bool(r_rb_check['passed'].all())}; "
        f"max spread={r_rb_check['spread'].max():.12f}; "
        f"checked pairs={len(r_rb_check)}"
    )
    lines.append("")

    lines.append("COMPUTATIONAL PERFORMANCE")
    lines.append("-" * 78)
    lines.append(performance.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    lines.append("FORMAL HR1 ROBUSTNESS DECISION")
    lines.append("-" * 78)
    lines.append(f"Status: {status}")
    if status == "ROBUST":
        lines.append("Interpretation: HR1 gain is positive, statistically supported, and not dominated by scale.")
    elif status == "PARTIALLY_ROBUST":
        lines.append("Interpretation: HR1 gain remains positive, but at least one strict robustness criterion is not met.")
    else:
        lines.append("Interpretation: HR1 gain disappears or reverses for at least one topology×N condition.")

    return "\n".join(lines) + "\n"


def main() -> None:
    """Runs the full R5 analysis and writes artifacts."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    desc = descriptive_stats(raw)
    raw_spearman = raw_spearman_analysis(raw)
    gain = strategy_gain(desc)
    gain_spearman = gain_spearman_analysis(gain)
    wilcoxon_df = paired_wilcoxon_tests(raw)
    r_rb_check = r_rb_sanity_check(raw)
    f0_df = f0_scaling(raw)
    performance = performance_by_N(raw)
    continuity = r3_continuity_check(gain)
    status = hr1_robustness_status(gain, gain_spearman, wilcoxon_df)

    desc.to_csv(AGG_DIR / "R5_descriptive_statistics.csv", index=False)
    raw_spearman.to_csv(AGG_DIR / "R5_raw_spearman.csv", index=False)
    gain.to_csv(AGG_DIR / "R5_strategy_gain.csv", index=False)
    gain_spearman.to_csv(AGG_DIR / "R5_gain_spearman.csv", index=False)
    wilcoxon_df.to_csv(AGG_DIR / "R5_pairwise_wilcoxon.csv", index=False)
    r_rb_check.to_csv(AGG_DIR / "R5_R_rb_sanity_check.csv", index=False)
    f0_df.to_csv(AGG_DIR / "R5_F0_scaling.csv", index=False)
    performance.to_csv(AGG_DIR / "R5_performance.csv", index=False)
    continuity.to_csv(AGG_DIR / "R5_R3_continuity_check.csv", index=False)

    save_figures(desc, gain, f0_df)

    report = make_report(
        desc=desc,
        raw_spearman=raw_spearman,
        gain=gain,
        gain_spearman=gain_spearman,
        wilcoxon_df=wilcoxon_df,
        r_rb_check=r_rb_check,
        f0_df=f0_df,
        performance=performance,
        continuity=continuity,
        status=status,
    )
    report_path = AGG_DIR / "R5_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

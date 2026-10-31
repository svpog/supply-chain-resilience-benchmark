"""
Аналіз серії R4 для статті 3.

Реалізує: Protocol R4 + Math Spec Article 3 Patch v3.3
(див. docs/Протокол_R4.docx та docs/Math_Spec_Article_3_Patch_v3_3.docx).

Приклад використання:
    python -m src.analysis.r4_analysis
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
from scipy.stats import wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "R4"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

TOPOLOGIES = ["hierarchical", "dense", "reserved"]
DISRUPTION_TYPES = ["node_failure", "edge_degradation", "cluster"]
STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]
ALPHA = 0.10
BOOTSTRAP_B = 1000
PAIRWISE_ALPHA = 0.05 / 54


def _clean(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
    """Повертає float-масив без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap за текстовими частинами."""
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


def cliffs_delta(x_values: pd.Series | np.ndarray, y_values: pd.Series | np.ndarray) -> float:
    """Cliff's δ: positive якщо y загалом більший за x."""
    x = _clean(x_values)
    y = _clean(y_values)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = y[:, None] - x[None, :]
    return float(np.sum(np.sign(diff)) / (x.size * y.size))


def effect_size_label(delta: float) -> str:
    """Інтерпретація |Cliff's δ| за Romano et al. (2006)."""
    if math.isnan(delta):
        return "NA"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def topo_label(value: str, lang: str) -> str:
    """Локалізована назва topology."""
    return label(f"common.{value}", lang)


def strategy_label(value: str, lang: str) -> str:
    """Локалізована назва recovery strategy."""
    return label(f"common.{value}", lang)


def disruption_label(value: str, lang: str) -> str:
    """Локалізована назва disruption type."""
    return label(f"R4.disruption_types.{value}", lang)


def load_raw() -> pd.DataFrame:
    """Завантажує raw_results.csv R4."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"R4 raw_results.csv not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("R4 raw_results.csv is empty.")
    raw["valid"] = raw["valid"].astype(bool)
    return raw


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для 27 комбінацій topology × disruption × strategy."""
    rows: list[dict] = []
    for (topology, disruption, strategy), group_all in raw.groupby(
        ["topology", "disruption_type", "recovery_strategy"]
    ):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)

        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("R4", topology, disruption, strategy, "rho_med"),
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("R4", topology, disruption, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho,
            compute_rho_IQR,
            seed=stable_seed("R4", topology, disruption, strategy, "rho_iqr"),
        )
        r_rb_ci = bootstrap_ci(
            r_rb,
            np.median,
            seed=stable_seed("R4", topology, disruption, strategy, "r_rb"),
        )

        rows.append(
            {
                "topology": topology,
                "disruption_type": disruption,
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
                "R_rb_ci_low": r_rb_ci[0],
                "R_rb_ci_high": r_rb_ci[1],
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0)
                if len(group) else float("nan"),
            }
        )
    order = {"hierarchical": 0, "dense": 1, "reserved": 2}
    out = pd.DataFrame(rows)
    out["_topology_order"] = out["topology"].map(order)
    return (
        out.sort_values(["_topology_order", "disruption_type", "recovery_strategy"])
        .drop(columns=["_topology_order"])
        .reset_index(drop=True)
    )


def _paired_values(
    raw: pd.DataFrame,
    topology: str,
    disruption: str,
    left_strategy: str,
    right_strategy: str,
) -> pd.DataFrame:
    """Паровані значення rho за paired_id для двох стратегій."""
    subset = raw[
        raw["valid"].astype(bool)
        & (raw["topology"] == topology)
        & (raw["disruption_type"] == disruption)
    ]
    left = subset[subset["recovery_strategy"] == left_strategy][["paired_id", "rho"]]
    right = subset[subset["recovery_strategy"] == right_strategy][["paired_id", "rho"]]
    return left.merge(right, on="paired_id", suffixes=("_left", "_right"))


def _empty_test_row(
    topology: str,
    disruption: str,
    left: str,
    right: str,
    metric: str,
    n_pairs: int,
) -> dict:
    """Порожній рядок pairwise-тесту, якщо пар недостатньо."""
    return {
        "topology": topology,
        "disruption_type": disruption,
        "left_strategy": left,
        "right_strategy": right,
        "metric": metric,
        "n_pairs": n_pairs,
        "median_delta": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "p_adj_bonferroni": float("nan"),
        "significant": False,
        "cliffs_delta": float("nan"),
        "effect_size": "NA",
        "protocol_alpha_pairwise": PAIRWISE_ALPHA,
    }


def paired_wilcoxon_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """54 paired Wilcoxon tests: 9 topo-disruption pairs × 3 pairs × 2 metrics."""
    rows: list[dict] = []
    strategy_pairs = [
        ("random", "centrality_betweenness"),
        ("random", "flow_greedy"),
        ("centrality_betweenness", "flow_greedy"),
    ]

    for topology in TOPOLOGIES:
        for disruption in DISRUPTION_TYPES:
            for left, right in strategy_pairs:
                pairs = _paired_values(raw, topology, disruption, left, right)
                for metric, mode in [("rho_med", "all"), ("rho_CVaR_0.10", "tail")]:
                    if len(pairs) < 2:
                        rows.append(_empty_test_row(topology, disruption, left, right, metric, len(pairs)))
                        continue

                    x = pairs["rho_left"].astype(float)
                    y = pairs["rho_right"].astype(float)
                    if mode == "tail":
                        x_thr = np.quantile(x, ALPHA)
                        y_thr = np.quantile(y, ALPHA)
                        mask = (x <= x_thr) | (y <= y_thr)
                        x_test = x[mask]
                        y_test = y[mask]
                        delta_value = compute_rho_CVaR(y, ALPHA) - compute_rho_CVaR(x, ALPHA)
                    else:
                        x_test = x
                        y_test = y
                        delta_value = float(np.median(y) - np.median(x))

                    if len(x_test) < 2 or np.allclose(y_test - x_test, 0.0):
                        statistic, p_value = 0.0, 1.0
                    else:
                        statistic, p_value = wilcoxon(y_test, x_test, alternative="greater")

                    delta = cliffs_delta(x_test, y_test)
                    rows.append(
                        {
                            "topology": topology,
                            "disruption_type": disruption,
                            "left_strategy": left,
                            "right_strategy": right,
                            "metric": metric,
                            "n_pairs": int(len(x_test)),
                            "median_delta": float(delta_value),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "cliffs_delta": delta,
                            "effect_size": effect_size_label(delta),
                        }
                    )

    tests = pd.DataFrame(rows)
    tests["p_adj_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    tests["significant"] = tests["p_value"] < PAIRWISE_ALPHA
    tests["protocol_alpha_pairwise"] = PAIRWISE_ALPHA
    return tests


def _paired_strategy_gain_ci(raw: pd.DataFrame, topology: str, disruption: str, metric: str) -> tuple[float, float]:
    """Bootstrap CI для paired gain flow_greedy − random."""
    pairs = _paired_values(raw, topology, disruption, "random", "flow_greedy")
    if pairs.empty:
        return (float("nan"), float("nan"))
    x = pairs["rho_left"].astype(float).to_numpy()
    y = pairs["rho_right"].astype(float).to_numpy()
    rng = np.random.default_rng(stable_seed("R4", topology, disruption, metric, "gain_ci"))
    stats = np.empty(BOOTSTRAP_B, dtype=float)
    for i in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(pairs), size=len(pairs))
        xb = x[idx]
        yb = y[idx]
        if metric == "rho_CVaR_0.10":
            stats[i] = compute_rho_CVaR(yb, ALPHA) - compute_rho_CVaR(xb, ALPHA)
        else:
            stats[i] = float(np.median(yb) - np.median(xb))
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def strategy_gain(desc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Δρ_med і Δρ_CVaR: flow_greedy − random для 9 topology × disruption пар."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        for disruption in DISRUPTION_TYPES:
            sub = desc[
                (desc["topology"] == topology)
                & (desc["disruption_type"] == disruption)
            ].set_index("recovery_strategy")
            if not {"random", "flow_greedy"}.issubset(sub.index):
                continue
            med_ci = _paired_strategy_gain_ci(raw, topology, disruption, "rho_med")
            cvar_ci = _paired_strategy_gain_ci(raw, topology, disruption, "rho_CVaR_0.10")
            rows.append(
                {
                    "topology": topology,
                    "disruption_type": disruption,
                    "delta_rho_med_flow_minus_random": float(
                        sub.loc["flow_greedy", "rho_med"] - sub.loc["random", "rho_med"]
                    ),
                    "delta_rho_med_ci_low": med_ci[0],
                    "delta_rho_med_ci_high": med_ci[1],
                    "delta_rho_CVaR_flow_minus_random": float(
                        sub.loc["flow_greedy", "rho_CVaR_0.10"] - sub.loc["random", "rho_CVaR_0.10"]
                    ),
                    "delta_rho_CVaR_ci_low": cvar_ci[0],
                    "delta_rho_CVaR_ci_high": cvar_ci[1],
                }
            )
    return pd.DataFrame(rows)


def backup_activation_test(desc: pd.DataFrame) -> pd.DataFrame:
    """HR3'' test: Δρ_CVaR(reserved − hierarchical) for random recovery."""
    rows: list[dict] = []
    random_desc = desc[desc["recovery_strategy"] == "random"]
    for disruption in DISRUPTION_TYPES:
        h = random_desc[
            (random_desc["topology"] == "hierarchical")
            & (random_desc["disruption_type"] == disruption)
        ]
        r = random_desc[
            (random_desc["topology"] == "reserved")
            & (random_desc["disruption_type"] == disruption)
        ]
        delta = float("nan")
        if not h.empty and not r.empty:
            delta = float(r["rho_CVaR_0.10"].iloc[0] - h["rho_CVaR_0.10"].iloc[0])
        rows.append(
            {
                "disruption_type": disruption,
                "delta_rho_CVaR_reserved_minus_hierarchical": delta,
            }
        )
    out = pd.DataFrame(rows)
    node_delta = float(
        out.loc[out["disruption_type"] == "node_failure",
                "delta_rho_CVaR_reserved_minus_hierarchical"].iloc[0]
    )
    edge_delta = float(
        out.loc[out["disruption_type"] == "edge_degradation",
                "delta_rho_CVaR_reserved_minus_hierarchical"].iloc[0]
    )
    ratio = edge_delta / node_delta if node_delta > 0 else float("nan")
    out["edge_node_ratio"] = ratio
    # Protocol R4 формулює HR3'' через ratio edge/node, але цей ratio
    # має сенс лише якщо node baseline додатний. Якщо reserved не краща за
    # hierarchical при node_failure, а edge effect залишається малим, це не
    # підтвердження активації backup, а відхилення HR3''.
    if node_delta > 0.0 and ratio > 2.0 and edge_delta > node_delta:
        status = "CONFIRMED"
    elif node_delta > 0.0 and ratio > 1.25 and edge_delta > node_delta:
        status = "PARTIALLY"
    else:
        status = "REJECTED"
    out["HR3pp_status"] = status
    return out


def disruption_severity(desc: pd.DataFrame) -> pd.DataFrame:
    """Cross-comparison: нижчий ρ_CVaR означає жорсткіший тип збою."""
    rows: list[dict] = []
    for disruption in DISRUPTION_TYPES:
        group = desc[desc["disruption_type"] == disruption]
        rows.append(
            {
                "disruption_type": disruption,
                "mean_rho_CVaR": float(group["rho_CVaR_0.10"].mean()),
                "median_rho_CVaR": float(group["rho_CVaR_0.10"].median()),
                "mean_R_rb": float(group["R_rb_med"].mean()),
                "severity_rank_key": float(group["rho_CVaR_0.10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("severity_rank_key").reset_index(drop=True)


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє ідентичність R_rb по стратегіях у межах paired_id."""
    rows: list[dict] = []
    valid = raw[raw["valid"]].copy()
    for (topology, disruption, paired_id), group in valid.groupby(
        ["topology", "disruption_type", "paired_id"]
    ):
        if group["recovery_strategy"].nunique() < len(STRATEGIES):
            continue
        spread = float(group["R_rb"].max() - group["R_rb"].min())
        rows.append(
            {
                "topology": topology,
                "disruption_type": disruption,
                "paired_id": paired_id,
                "n_strategies": int(group["recovery_strategy"].nunique()),
                "R_rb_min": float(group["R_rb"].min()),
                "R_rb_max": float(group["R_rb"].max()),
                "R_rb_spread": spread,
                "passed": bool(spread <= 1e-12),
            }
        )
    return pd.DataFrame(rows)


def hr1_extended_summary(gain: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Формальна перевірка HR1 на 9 topology × disruption комбінаціях."""
    rows: list[dict] = []
    fg_tests = pairwise[
        (pairwise["left_strategy"] == "random")
        & (pairwise["right_strategy"] == "flow_greedy")
    ]
    for _, grow in gain.iterrows():
        topology = grow["topology"]
        disruption = grow["disruption_type"]
        rho_test = fg_tests[
            (fg_tests["topology"] == topology)
            & (fg_tests["disruption_type"] == disruption)
            & (fg_tests["metric"] == "rho_med")
        ]
        cvar_test = fg_tests[
            (fg_tests["topology"] == topology)
            & (fg_tests["disruption_type"] == disruption)
            & (fg_tests["metric"] == "rho_CVaR_0.10")
        ]
        rho_hit = (
            not rho_test.empty
            and bool(rho_test["significant"].iloc[0])
            and float(rho_test["cliffs_delta"].iloc[0]) >= 0.30
            and float(grow["delta_rho_med_flow_minus_random"]) > 0.0
        )
        cvar_hit = (
            not cvar_test.empty
            and bool(cvar_test["significant"].iloc[0])
            and float(cvar_test["cliffs_delta"].iloc[0]) >= 0.474
            and float(grow["delta_rho_CVaR_flow_minus_random"]) > 0.0
        )
        rows.append(
            {
                "topology": topology,
                "disruption_type": disruption,
                "rho_med_HR1_hit": bool(rho_hit),
                "rho_CVaR_HR1_hit": bool(cvar_hit),
                "delta_rho_med": float(grow["delta_rho_med_flow_minus_random"]),
                "delta_rho_CVaR": float(grow["delta_rho_CVaR_flow_minus_random"]),
            }
        )
    out = pd.DataFrame(rows)
    out["HR1_extended_hit"] = out["rho_med_HR1_hit"] & out["rho_CVaR_HR1_hit"]
    return out


def plot_cvar_by_topology_disruption(desc: pd.DataFrame, lang: str) -> plt.Figure:
    """Grouped bar chart ρ_CVaR by topology, disruption, strategy."""
    data = desc.copy()
    data["group_label"] = [
        f"{topo_label(t, lang)}\n{disruption_label(d, lang)}"
        for t, d in zip(data["topology"], data["disruption_type"])
    ]
    groups = [
        f"{topo_label(t, lang)}\n{disruption_label(d, lang)}"
        for t in TOPOLOGIES for d in DISRUPTION_TYPES
    ]
    x = np.arange(len(groups))
    width = 0.24
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = {"random": "#7c8a99", "centrality_betweenness": "#4c78a8", "flow_greedy": "#f58518"}
    for i, strategy in enumerate(STRATEGIES):
        ys = []
        lows = []
        highs = []
        for t in TOPOLOGIES:
            for d in DISRUPTION_TYPES:
                row = data[
                    (data["topology"] == t)
                    & (data["disruption_type"] == d)
                    & (data["recovery_strategy"] == strategy)
                ]
                if row.empty:
                    ys.append(np.nan)
                    lows.append(0.0)
                    highs.append(0.0)
                else:
                    y = float(row["rho_CVaR_0.10"].iloc[0])
                    ys.append(y)
                    lows.append(y - float(row["rho_CVaR_ci_low"].iloc[0]))
                    highs.append(float(row["rho_CVaR_ci_high"].iloc[0]) - y)
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            ys,
            width=width,
            yerr=[lows, highs],
            capsize=3,
            color=colors[strategy],
            label=strategy_label(strategy, lang),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylabel(label("R4.cvar_by_topology.y_label", lang))
    ax.set_title(label("R4.cvar_by_topology.title", lang))
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    return fig


def plot_strategy_gain_by_disruption(gain: pd.DataFrame, lang: str) -> plt.Figure:
    """Bar chart Δρ_CVaR(flow_greedy − random) for 9 pairs."""
    data = gain.copy()
    labels = [
        f"{topo_label(t, lang)}\n{disruption_label(d, lang)}"
        for t, d in zip(data["topology"], data["disruption_type"])
    ]
    x = np.arange(len(data))
    y = data["delta_rho_CVaR_flow_minus_random"].astype(float).to_numpy()
    low = y - data["delta_rho_CVaR_ci_low"].astype(float).to_numpy()
    high = data["delta_rho_CVaR_ci_high"].astype(float).to_numpy() - y
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x, y, yerr=[low, high], capsize=4, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(label("R4.strategy_gain_by_disruption.y_label", lang))
    ax.set_title(label("R4.strategy_gain_by_disruption.title", lang))
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_backup_activation(activation: pd.DataFrame, lang: str) -> plt.Figure:
    """Central HR3'' figure: reserved − hierarchical Δρ_CVaR by disruption."""
    data = activation.copy()
    x = np.arange(len(data))
    y = data["delta_rho_CVaR_reserved_minus_hierarchical"].astype(float).to_numpy()
    colors = ["#7c8a99", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, color=colors[: len(x)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([disruption_label(d, lang) for d in data["disruption_type"]])
    ax.set_ylabel(label("R4.backup_activation.y_label", lang))
    ax.set_title(label("R4.backup_activation.title", lang))
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(y):
        ax.text(i, value + 0.003 if value >= 0 else value - 0.006, f"{value:.3f}", ha="center")
    return fig


def plot_disruption_severity(raw: pd.DataFrame, lang: str) -> plt.Figure:
    """Boxplot raw ρ by disruption type; нижче = жорсткіший збій."""
    valid = raw[raw["valid"]].copy()
    data = [valid[valid["disruption_type"] == d]["rho"].astype(float).to_numpy() for d in DISRUPTION_TYPES]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        data,
        tick_labels=[disruption_label(d, lang) for d in DISRUPTION_TYPES],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d7e5f0", "edgecolor": "#2f4b66"},
        medianprops={"color": "#c43c39", "linewidth": 1.5},
    )
    ax.set_ylabel(label("R4.disruption_severity.y_label", lang))
    ax.set_title(label("R4.disruption_severity.title", lang))
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_figures(desc: pd.DataFrame, gain: pd.DataFrame, activation: pd.DataFrame, raw: pd.DataFrame) -> list[Path]:
    """Створює 4 типи × 2 мови PNG-файлів."""
    outputs: list[Path] = []
    outputs.extend(save_dual_lang(
        plot_cvar_by_topology_disruption,
        str(FIG_DIR / "R4_cvar_by_topology_disruption_{lang}.png"),
        desc=desc,
    ))
    outputs.extend(save_dual_lang(
        plot_strategy_gain_by_disruption,
        str(FIG_DIR / "R4_strategy_gain_by_disruption_{lang}.png"),
        gain=gain,
    ))
    outputs.extend(save_dual_lang(
        plot_backup_activation,
        str(FIG_DIR / "R4_backup_activation_{lang}.png"),
        activation=activation,
    ))
    outputs.extend(save_dual_lang(
        plot_disruption_severity,
        str(FIG_DIR / "R4_disruption_severity_{lang}.png"),
        raw=raw,
    ))
    return outputs


def _format_float_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Форматує таблицю для TXT-звіту."""
    return df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}")


def write_report(
    raw: pd.DataFrame,
    desc: pd.DataFrame,
    gain: pd.DataFrame,
    activation: pd.DataFrame,
    severity: pd.DataFrame,
    pairwise: pd.DataFrame,
    sanity: pd.DataFrame,
    hr1: pd.DataFrame,
    figures: list[Path],
) -> Path:
    """Пише R4_report.txt з усіма секціями протоколу."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = AGG_DIR / "R4_report.txt"

    ratio = float(activation["edge_node_ratio"].iloc[0]) if not activation.empty else float("nan")
    status = str(activation["HR3pp_status"].iloc[0]) if not activation.empty else "UNKNOWN"
    sanity_passed = bool(not sanity.empty and sanity["passed"].all())
    hardest = str(severity.iloc[0]["disruption_type"]) if not severity.empty else "unknown"
    hr1_hits = int(hr1["HR1_extended_hit"].sum()) if not hr1.empty else 0

    desc_cols = [
        "topology", "disruption_type", "recovery_strategy", "n_valid", "n_invalid",
        "excl_pct", "rho_med", "rho_med_ci_low", "rho_med_ci_high",
        "rho_CVaR_0.10", "rho_CVaR_ci_low", "rho_CVaR_ci_high",
        "rho_IQR", "R_rb_med", "non_recovered_pct",
    ]
    gain_cols = [
        "topology", "disruption_type",
        "delta_rho_med_flow_minus_random",
        "delta_rho_CVaR_flow_minus_random",
        "delta_rho_CVaR_ci_low",
        "delta_rho_CVaR_ci_high",
    ]
    activation_cols = [
        "disruption_type",
        "delta_rho_CVaR_reserved_minus_hierarchical",
        "edge_node_ratio",
        "HR3pp_status",
    ]
    pairwise_cols = [
        "topology", "disruption_type", "left_strategy", "right_strategy",
        "metric", "n_pairs", "median_delta", "p_value",
        "p_adj_bonferroni", "significant", "cliffs_delta", "effect_size",
    ]

    lines = [
        "============================================================",
        "R4: STRATEGY × DISRUPTION TYPE INTERACTION (HR3'' test)",
        "============================================================",
        "",
        "Parameters: 3 topologies × 3 disruption types × 3 strategies, "
        "N=50, T=100, M=100, μ=1, paired within disruption_type.",
        f"Total runs: {len(raw)}",
        f"Valid runs: {int(raw['valid'].sum())}",
        f"Invalid runs: {int((~raw['valid']).sum())}",
        "",
        "--- DESCRIPTIVE STATISTICS (27 rows) ---",
        _format_float_table(desc, desc_cols),
        "",
        "--- STRATEGY GAIN BY DISRUPTION TYPE ---",
        _format_float_table(gain, gain_cols),
        "",
        "--- HR3'' TEST: BACKUP ACTIVATION AT edge_degradation ---",
        "Δρ_CVaR (reserved − hierarchical) at random recovery:",
        _format_float_table(activation, activation_cols),
        f"",
        f"Ratio edge/node: {ratio:.4f}",
        f"HR3'' status: {status}",
        "",
        "--- EXTENDED HR1 CHECK ---",
        f"HR1 extended hits (rho_med and rho_CVaR criteria): {hr1_hits}/9",
        _format_float_table(
            hr1,
            [
                "topology", "disruption_type", "delta_rho_med", "delta_rho_CVaR",
                "rho_med_HR1_hit", "rho_CVaR_HR1_hit", "HR1_extended_hit",
            ],
        ),
        "",
        "--- DISRUPTION SEVERITY ---",
        "Lower rho_CVaR means harsher disruption.",
        _format_float_table(
            severity,
            ["disruption_type", "mean_rho_CVaR", "median_rho_CVaR", "mean_R_rb"],
        ),
        f"Hardest disruption type by mean rho_CVaR: {hardest}",
        "",
        f"--- PAIRED WILCOXON (54 tests, Bonferroni α={PAIRWISE_ALPHA:.6f}) ---",
        _format_float_table(pairwise, pairwise_cols),
        "",
        "--- R_RB SANITY CHECK ---",
        f"R_rb identical across strategies within paired_id: {'YES' if sanity_passed else 'NO'}",
        f"Checked paired groups: {len(sanity)}",
        f"Failed groups: {0 if sanity.empty else int((~sanity['passed']).sum())}",
        "",
        "--- FIGURES ---",
        *[str(path.relative_to(ROOT)) for path in figures],
        "",
        "--- INTERPRETATION ---",
        (
            "HR3'' is confirmed if edge/node > 2.0. "
            "This means the reserved topology advantage over hierarchical is much stronger "
            "under edge_degradation than under node_failure, consistent with backup edge "
            "activation when endpoints remain alive."
        ),
        (
            "If HR3'' is rejected, the topology-function dissonance is deeper than the "
            "edge/node mechanism proposed in Patch v3.3 and needs discussion before further series."
        ),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_tables(
    desc: pd.DataFrame,
    gain: pd.DataFrame,
    activation: pd.DataFrame,
    severity: pd.DataFrame,
    pairwise: pd.DataFrame,
    sanity: pd.DataFrame,
    hr1: pd.DataFrame,
) -> None:
    """Зберігає CSV-артефакти R4 analysis."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(AGG_DIR / "R4_descriptive_statistics.csv", index=False)
    gain.to_csv(AGG_DIR / "R4_strategy_gain.csv", index=False)
    activation.to_csv(AGG_DIR / "R4_backup_activation.csv", index=False)
    severity.to_csv(AGG_DIR / "R4_disruption_severity.csv", index=False)
    pairwise.to_csv(AGG_DIR / "R4_pairwise_wilcoxon.csv", index=False)
    sanity.to_csv(AGG_DIR / "R4_R_rb_sanity_check.csv", index=False)
    hr1.to_csv(AGG_DIR / "R4_HR1_extended_check.csv", index=False)


def run_analysis() -> dict[str, object]:
    """Повний pipeline аналізу R4."""
    raw = load_raw()
    desc = descriptive_stats(raw)
    pairwise = paired_wilcoxon_tests(raw)
    gain = strategy_gain(desc, raw)
    activation = backup_activation_test(desc)
    severity = disruption_severity(desc)
    sanity = r_rb_sanity_check(raw)
    hr1 = hr1_extended_summary(gain, pairwise)
    figures = make_figures(desc, gain, activation, raw)
    save_tables(desc, gain, activation, severity, pairwise, sanity, hr1)
    report_path = write_report(raw, desc, gain, activation, severity, pairwise, sanity, hr1, figures)
    return {
        "raw": raw,
        "desc": desc,
        "pairwise": pairwise,
        "gain": gain,
        "activation": activation,
        "severity": severity,
        "sanity": sanity,
        "hr1": hr1,
        "figures": figures,
        "report_path": report_path,
    }


def main() -> None:
    """CLI entrypoint для python -m src.analysis.r4_analysis."""
    result = run_analysis()
    report_path = result["report_path"]
    print(f"R4 analysis complete: {report_path}")


if __name__ == "__main__":
    main()

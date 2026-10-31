"""
Аналіз серії R3 для статті 3.

Реалізує: Protocol R3 + Math Spec Article 3 Patch v3.3
(див. docs/Протокол_R3.docx та docs/Math_Spec_Article_3_Patch_v3_3.docx).

Приклад використання:
    python -m src.analysis.r3_analysis
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
RAW_DIR = ROOT / "results" / "raw" / "R3"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

TOPOLOGIES = ["hierarchical", "dense"]
STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]
MU_VALUES = [1, 2, 5, 10]
ALPHA = 0.10
BOOTSTRAP_B = 1000
PERMUTATION_B = 10_000
PAIRWISE_ALPHA = 0.05 / 48


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
    """Завантажує raw_results.csv R3."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"R3 raw_results.csv not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("R3 raw_results.csv is empty.")
    raw["valid"] = raw["valid"].astype(bool)
    raw["mu"] = raw["mu"].astype(int)
    return raw


def _find_t_r(Q_curve: list[float], t_e: int = 1, q_target: float | None = None) -> int:
    """Перший t ≥ t_e, де Q(t) ≥ Q_target; інакше T_max."""
    if not Q_curve:
        return 0
    target = 0.95 * Q_curve[0] if q_target is None else q_target
    T_max = len(Q_curve) - 1
    start = min(max(t_e, 0), T_max)
    for t in range(start, T_max + 1):
        if Q_curve[t] >= target:
            return int(t)
    return int(T_max)


def compute_t_r_by_run() -> pd.DataFrame:
    """Обчислює t_r з Q_curves.parquet для кожного run_id."""
    path = RAW_DIR / "Q_curves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"R3 Q_curves.parquet not found: {path}")
    q_df = pd.read_parquet(path)
    rows: list[dict] = []
    for run_id, group in q_df.groupby("run_id"):
        q_curve = group.sort_values("t")["Q"].astype(float).tolist()
        rows.append({"run_id": run_id, "t_r": _find_t_r(q_curve, t_e=1)})
    return pd.DataFrame(rows)


def attach_t_r(raw: pd.DataFrame) -> pd.DataFrame:
    """Додає колонку t_r до raw_results."""
    t_r = compute_t_r_by_run()
    out = raw.merge(t_r, on="run_id", how="left")
    out.loc[~out["valid"], "t_r"] = np.nan
    return out


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для 24 комбінацій topology × μ × strategy."""
    rows: list[dict] = []
    for (topology, mu, strategy), group_all in raw.groupby(
        ["topology", "mu", "recovery_strategy"]
    ):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)
        t_r = group["t_r"].astype(float)

        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("R3", topology, mu, strategy, "rho_med"),
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("R3", topology, mu, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho,
            compute_rho_IQR,
            seed=stable_seed("R3", topology, mu, strategy, "rho_iqr"),
        )
        r_rb_ci = bootstrap_ci(
            r_rb,
            np.median,
            seed=stable_seed("R3", topology, mu, strategy, "r_rb"),
        )

        rows.append(
            {
                "topology": topology,
                "mu": int(mu),
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
                "t_r_med": float(np.median(t_r)) if len(t_r) else float("nan"),
                "t_r_mean": float(np.mean(t_r)) if len(t_r) else float("nan"),
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0)
                if len(group) else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["topology", "mu", "recovery_strategy"])
        .reset_index(drop=True)
    )


def _tail_rows_for_cvar(raw_group: pd.DataFrame) -> pd.DataFrame:
    """Вибирає нижній α-хвіст ρ окремо для кожного μ."""
    parts: list[pd.DataFrame] = []
    for _, group in raw_group.groupby("mu"):
        valid = group[group["valid"]].sort_values("rho", ascending=True)
        n_tail = max(1, int(np.floor(ALPHA * len(valid))))
        parts.append(valid.head(n_tail))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def raw_spearman_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(μ, metric) на сирих даних для кожної topology × strategy."""
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
                    x = source["mu"].astype(float).to_numpy()
                    y = source["rho"].astype(float).to_numpy()
                    r_value, p_perm = permutation_spearman_p(
                        x,
                        y,
                        B=PERMUTATION_B,
                        seed=stable_seed("R3", topology, strategy, metric, "perm"),
                    )
                    scipy_r, scipy_p = spearmanr(x, y)
                    if not math.isnan(scipy_r):
                        r_value = float(scipy_r)
                    n_raw = len(source)
                rows.append(
                    {
                        "topology": topology,
                        "recovery_strategy": strategy,
                        "metric": metric,
                        "spearman_r_raw": float(r_value),
                        "p_value_permutation": float(p_perm),
                        "p_value_scipy_asymptotic": float(scipy_p),
                        "n_raw_points": int(n_raw),
                        "permutation_B": PERMUTATION_B,
                    }
                )
    return pd.DataFrame(rows)


def strategy_gain(desc: pd.DataFrame) -> pd.DataFrame:
    """Δρ_med і Δρ_CVaR для flow_greedy/centrality_betweenness проти random."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        topo_desc = desc[desc["topology"] == topology]
        random = topo_desc[topo_desc["recovery_strategy"] == "random"].set_index("mu")
        flow = topo_desc[topo_desc["recovery_strategy"] == "flow_greedy"].set_index("mu")
        cent = topo_desc[
            topo_desc["recovery_strategy"] == "centrality_betweenness"
        ].set_index("mu")
        for mu in MU_VALUES:
            if mu not in random.index or mu not in flow.index or mu not in cent.index:
                continue
            rows.append(
                {
                    "topology": topology,
                    "mu": int(mu),
                    "delta_rho_med_flow_minus_random": float(
                        flow.loc[mu, "rho_med"] - random.loc[mu, "rho_med"]
                    ),
                    "delta_rho_CVaR_flow_minus_random": float(
                        flow.loc[mu, "rho_CVaR_0.10"] - random.loc[mu, "rho_CVaR_0.10"]
                    ),
                    "delta_rho_med_centrality_minus_random": float(
                        cent.loc[mu, "rho_med"] - random.loc[mu, "rho_med"]
                    ),
                    "delta_rho_CVaR_centrality_minus_random": float(
                        cent.loc[mu, "rho_CVaR_0.10"] - random.loc[mu, "rho_CVaR_0.10"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["topology", "mu"]).reset_index(drop=True)


def gain_spearman_analysis(gain: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(μ, Δρ) для strategy gain; центральний аналіз HR5."""
    rows: list[dict] = []
    comparisons = {
        "flow_greedy_minus_random": {
            "rho_med": "delta_rho_med_flow_minus_random",
            "rho_CVaR_0.10": "delta_rho_CVaR_flow_minus_random",
        },
        "centrality_betweenness_minus_random": {
            "rho_med": "delta_rho_med_centrality_minus_random",
            "rho_CVaR_0.10": "delta_rho_CVaR_centrality_minus_random",
        },
    }
    for topology in TOPOLOGIES:
        group = gain[gain["topology"] == topology].sort_values("mu")
        for comparison, metrics in comparisons.items():
            for metric, col in metrics.items():
                r_value, p_perm = permutation_spearman_p(
                    group["mu"],
                    group[col],
                    B=PERMUTATION_B,
                    seed=stable_seed("R3", topology, comparison, metric, "gain"),
                )
                rows.append(
                    {
                        "topology": topology,
                        "comparison": comparison,
                        "metric": metric,
                        "spearman_r": float(r_value),
                        "p_value_permutation": float(p_perm),
                        "n_points": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _paired_values(
    raw: pd.DataFrame,
    topology: str,
    mu: int,
    left_strategy: str,
    right_strategy: str,
    column: str,
) -> pd.DataFrame:
    subset = raw[
        (raw["valid"])
        & (raw["topology"] == topology)
        & (raw["mu"].astype(int) == int(mu))
    ]
    left = subset[subset["recovery_strategy"] == left_strategy][["paired_id", column]]
    right = subset[subset["recovery_strategy"] == right_strategy][["paired_id", column]]
    return left.merge(right, on="paired_id", suffixes=("_left", "_right"))


def paired_wilcoxon_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """48 paired Wilcoxon tests: 2 topology × 4 μ × 3 pairs × 2 metrics."""
    rows: list[dict] = []
    strategy_pairs = [
        ("random", "centrality_betweenness"),
        ("random", "flow_greedy"),
        ("centrality_betweenness", "flow_greedy"),
    ]
    for topology in TOPOLOGIES:
        for mu in MU_VALUES:
            for left, right in strategy_pairs:
                pairs = _paired_values(raw, topology, mu, left, right, "rho")
                for metric, mode in [("rho_med", "all"), ("rho_CVaR_0.10", "tail")]:
                    if len(pairs) < 2:
                        rows.append(_empty_test_row(topology, mu, left, right, metric, len(pairs)))
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

                    rows.append(
                        {
                            "topology": topology,
                            "mu": int(mu),
                            "left_strategy": left,
                            "right_strategy": right,
                            "metric": metric,
                            "n_pairs": int(len(x_test)),
                            "median_delta": float(delta_value),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                        }
                    )
    tests = pd.DataFrame(rows)
    tests["p_adj_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    tests["significant"] = tests["p_value"] < PAIRWISE_ALPHA
    tests["protocol_alpha_pairwise"] = PAIRWISE_ALPHA
    return tests


def _empty_test_row(
    topology: str,
    mu: int,
    left: str,
    right: str,
    metric: str,
    n_pairs: int,
) -> dict:
    return {
        "topology": topology,
        "mu": int(mu),
        "left_strategy": left,
        "right_strategy": right,
        "metric": metric,
        "n_pairs": n_pairs,
        "median_delta": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє ідентичність R_rb по μ×strategy для кожного topology/paired_id."""
    valid = raw[raw["valid"]].copy()
    rows: list[dict] = []
    for (topology, paired_id), group in valid.groupby(["topology", "paired_id"]):
        values = group["R_rb"].astype(float).to_numpy()
        spread = float(np.nanmax(values) - np.nanmin(values)) if len(values) else float("nan")
        rows.append(
            {
                "topology": topology,
                "paired_id": paired_id,
                "n_rows": int(len(group)),
                "n_mu": int(group["mu"].nunique()),
                "n_strategies": int(group["recovery_strategy"].nunique()),
                "R_rb_min": float(np.nanmin(values)) if len(values) else float("nan"),
                "R_rb_max": float(np.nanmax(values)) if len(values) else float("nan"),
                "spread": spread,
                "passed": bool(spread <= 1e-12),
            }
        )
    sanity = pd.DataFrame(rows)
    if not sanity.empty and not sanity["passed"].all():
        bad = sanity[~sanity["passed"]].head()
        raise AssertionError(
            "R_rb varies across μ/strategies for at least one (topology, paired_id):\n"
            + bad.to_string(index=False)
        )
    return sanity


def t_r_sanity_check(raw: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє, що t_r зменшується при зростанні μ."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        for strategy in STRATEGIES:
            group = desc[
                (desc["topology"] == topology)
                & (desc["recovery_strategy"] == strategy)
            ].sort_values("mu")
            med_values = group["t_r_med"].astype(float).to_numpy()
            median_monotonic = bool(np.all(np.diff(med_values) <= 1e-12))

            valid = raw[
                (raw["valid"])
                & (raw["topology"] == topology)
                & (raw["recovery_strategy"] == strategy)
            ].copy()
            paired_total = 0
            paired_violations = 0
            for _, paired in valid.groupby("paired_id"):
                if paired["mu"].nunique() != len(MU_VALUES):
                    continue
                t_values = paired.sort_values("mu")["t_r"].astype(float).to_numpy()
                paired_total += 1
                paired_violations += int(not np.all(np.diff(t_values) <= 1e-12))

            rows.append(
                {
                    "topology": topology,
                    "recovery_strategy": strategy,
                    "median_t_r_by_mu": ",".join(f"{v:.1f}" for v in med_values),
                    "median_monotonic_nonincreasing": median_monotonic,
                    "paired_groups_checked": int(paired_total),
                    "paired_monotonic_violations": int(paired_violations),
                    "passed": bool(median_monotonic and paired_violations == 0),
                }
            )
    return pd.DataFrame(rows)


def hr5_status(gain_spearman: pd.DataFrame, gain: pd.DataFrame) -> tuple[str, str]:
    """Формальна перевірка HR5."""
    flow_cvar = gain_spearman[
        (gain_spearman["comparison"] == "flow_greedy_minus_random")
        & (gain_spearman["metric"] == "rho_CVaR_0.10")
    ]
    checks: list[dict] = []
    for topology in TOPOLOGIES:
        r_row = flow_cvar[flow_cvar["topology"] == topology].iloc[0]
        topo_gain = gain[gain["topology"] == topology].set_index("mu")
        gain_mu1 = float(topo_gain.loc[1, "delta_rho_CVaR_flow_minus_random"])
        gain_mu10 = float(topo_gain.loc[10, "delta_rho_CVaR_flow_minus_random"])
        checks.append(
            {
                "topology": topology,
                "r": float(r_row["spearman_r"]),
                "p": float(r_row["p_value_permutation"]),
                "gain_mu1": gain_mu1,
                "gain_mu10": gain_mu10,
                "half_or_less": bool(gain_mu10 <= 0.5 * gain_mu1),
                "positive_mu10": bool(gain_mu10 > 0.0),
            }
        )

    confirmed = all(
        item["r"] <= -0.5 and item["half_or_less"] and item["positive_mu10"]
        for item in checks
    )
    rejected = all(item["r"] >= 0.0 for item in checks)

    if confirmed:
        return (
            "CONFIRMED",
            "Для обох топологій Spearman r(μ, Δρ_CVaR_flow-random) ≤ -0.5, "
            "виграш при μ=10 не перевищує 50% від μ=1 і лишається позитивним.",
        )
    if rejected:
        return (
            "REJECTED",
            "Для обох топологій Spearman r(μ, Δρ_CVaR_flow-random) ≥ 0; "
            "очікуваного спадання виграшу зі зростанням μ не видно.",
        )
    return (
        "PARTIALLY_CONFIRMED",
        "Тренд HR5 має правильний напрям або проходить лише частину порогів; "
        "результат потребує обережної інтерпретації.",
    )


def localized_name(value: str, lang: str) -> str:
    """Локалізована назва з fallback."""
    try:
        return label(f"common.{value}", lang)
    except KeyError:
        return value


def _format_mu_axis(ax: plt.Axes) -> None:
    ax.set_xticks(MU_VALUES)
    ax.grid(alpha=0.25)


def plot_metric_vs_mu(desc: pd.DataFrame, lang: str, metric: str) -> plt.Figure:
    """Дві панелі topology; лінії strategy; metric ∈ {rho_med, rho_CVaR}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    if metric == "rho_med":
        y_col, low_col, high_col = "rho_med", "rho_med_ci_low", "rho_med_ci_high"
        title_key = "R3.rho_vs_mu.title"
        y_label = label("common.rho_med", lang)
    else:
        y_col, low_col, high_col = "rho_CVaR_0.10", "rho_CVaR_ci_low", "rho_CVaR_ci_high"
        title_key = "R3.cvar_vs_mu.title"
        y_label = label("common.rho_cvar", lang)

    for ax, topology in zip(axes, TOPOLOGIES):
        for strategy in STRATEGIES:
            group = desc[
                (desc["topology"] == topology)
                & (desc["recovery_strategy"] == strategy)
            ].sort_values("mu")
            x = group["mu"].astype(float)
            y = group[y_col].astype(float)
            low = group[low_col].astype(float)
            high = group[high_col].astype(float)
            ax.plot(x, y, marker="o", linewidth=2, label=localized_name(strategy, lang))
            ax.fill_between(x, low, high, alpha=0.16)
        ax.set_title(localized_name(topology, lang))
        ax.set_xlabel(label("R3.rho_vs_mu.x_label", lang))
        _format_mu_axis(ax)
    axes[0].set_ylabel(y_label)
    axes[1].legend(title=label("common.strategy", lang), loc="best")
    fig.suptitle(label(title_key, lang))
    fig.tight_layout()
    return fig


def plot_rho_vs_mu(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_mu(desc, lang=lang, metric="rho_med")


def plot_cvar_vs_mu(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_mu(desc, lang=lang, metric="rho_CVaR_0.10")


def plot_strategy_gain_vs_mu(gain: pd.DataFrame, lang: str) -> plt.Figure:
    """Δρ_CVaR(flow_greedy − random) за μ для двох топологій."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for topology in TOPOLOGIES:
        group = gain[gain["topology"] == topology].sort_values("mu")
        ax.plot(
            group["mu"],
            group["delta_rho_CVaR_flow_minus_random"],
            marker="o",
            linewidth=2.2,
            label=localized_name(topology, lang),
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(label("R3.strategy_gain_vs_mu.title", lang))
    ax.set_xlabel(label("R3.rho_vs_mu.x_label", lang))
    ax.set_ylabel(label("R3.strategy_gain_vs_mu.y_label", lang))
    _format_mu_axis(ax)
    ax.legend(title=label("common.topology", lang), loc="best")
    return fig


def plot_tr_vs_mu(desc: pd.DataFrame, lang: str) -> plt.Figure:
    """t_r median vs μ, дві панелі topology."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, topology in zip(axes, TOPOLOGIES):
        for strategy in STRATEGIES:
            group = desc[
                (desc["topology"] == topology)
                & (desc["recovery_strategy"] == strategy)
            ].sort_values("mu")
            ax.plot(
                group["mu"],
                group["t_r_med"],
                marker="o",
                linewidth=2,
                label=localized_name(strategy, lang),
            )
        ax.set_title(localized_name(topology, lang))
        ax.set_xlabel(label("R3.rho_vs_mu.x_label", lang))
        _format_mu_axis(ax)
    axes[0].set_ylabel(label("R3.tr_vs_mu.y_label", lang))
    axes[1].legend(title=label("common.strategy", lang), loc="best")
    fig.suptitle(label("R3.tr_vs_mu.title", lang))
    fig.tight_layout()
    return fig


def make_figures(desc: pd.DataFrame, gain: pd.DataFrame) -> list[Path]:
    """Зберігає 4 типи рисунків у двох мовах."""
    outputs: list[Path] = []
    outputs += save_dual_lang(
        plot_rho_vs_mu,
        str(FIG_DIR / "R3_rho_vs_mu_{lang}.png"),
        desc=desc,
    )
    outputs += save_dual_lang(
        plot_cvar_vs_mu,
        str(FIG_DIR / "R3_cvar_vs_mu_{lang}.png"),
        desc=desc,
    )
    outputs += save_dual_lang(
        plot_strategy_gain_vs_mu,
        str(FIG_DIR / "R3_strategy_gain_vs_mu_{lang}.png"),
        gain=gain,
    )
    outputs += save_dual_lang(
        plot_tr_vs_mu,
        str(FIG_DIR / "R3_tr_vs_mu_{lang}.png"),
        desc=desc,
    )
    return outputs


def make_report(
    raw: pd.DataFrame,
    desc: pd.DataFrame,
    spearman: pd.DataFrame,
    gain: pd.DataFrame,
    gain_spearman: pd.DataFrame,
    pairwise: pd.DataFrame,
    r_rb_sanity: pd.DataFrame,
    t_r_sanity: pd.DataFrame,
    figures: list[Path],
) -> str:
    """Формує TXT-звіт R3."""
    status, status_reason = hr5_status(gain_spearman, gain)
    r_rb_passed = bool(r_rb_sanity["passed"].all()) if not r_rb_sanity.empty else False
    t_r_passed = bool(t_r_sanity["passed"].all()) if not t_r_sanity.empty else False
    invalid = raw[~raw["valid"]]
    paired_counts = raw.groupby(["topology", "paired_id"]).size()
    expected_rows_per_pair = len(MU_VALUES) * len(STRATEGIES)

    desc_cols = [
        "topology",
        "mu",
        "recovery_strategy",
        "n_valid",
        "excl_pct",
        "rho_med",
        "rho_med_ci_low",
        "rho_med_ci_high",
        "rho_CVaR_0.10",
        "rho_CVaR_ci_low",
        "rho_CVaR_ci_high",
        "rho_IQR",
        "R_rb_med",
        "t_r_med",
        "non_recovered_pct",
    ]
    spearman_cols = [
        "topology",
        "recovery_strategy",
        "metric",
        "spearman_r_raw",
        "p_value_permutation",
        "n_raw_points",
    ]
    gain_cols = [
        "topology",
        "mu",
        "delta_rho_med_flow_minus_random",
        "delta_rho_CVaR_flow_minus_random",
        "delta_rho_med_centrality_minus_random",
        "delta_rho_CVaR_centrality_minus_random",
    ]
    gain_spearman_cols = [
        "topology",
        "comparison",
        "metric",
        "spearman_r",
        "p_value_permutation",
        "n_points",
    ]
    pairwise_cols = [
        "topology",
        "mu",
        "left_strategy",
        "right_strategy",
        "metric",
        "n_pairs",
        "median_delta",
        "p_value",
        "p_adj_bonferroni",
        "significant",
    ]

    lines = [
        "============================================================",
        "R3: RECOVERY SPEED MU SENSITIVITY (HR5 verification)",
        "============================================================",
        "",
        "Parameters: 2 topologies × 3 strategies × 4 mu values,",
        "            N=50, p_v=0.20, T=100, M=100, paired design",
        "",
        "--- DESCRIPTIVE STATISTICS (24 rows) ---",
        desc[desc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- SPEARMAN ON RAW DATA (per topology × strategy) ---",
        "rho_med uses all valid raw runs; rho_CVaR_0.10 uses the bottom 10% tail within each μ group.",
        spearman[spearman_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- STRATEGY GAIN AS FUNCTION OF MU (CENTRAL HR5 ANALYSIS) ---",
        gain[gain_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Spearman r(mu, strategy gain):",
        gain_spearman[gain_spearman_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- HYPOTHESIS HR5 STATUS ---",
        status,
        status_reason,
        "",
        "--- PAIRED WILCOXON TESTS (per topology × μ, Bonferroni alpha=0.001) ---",
        pairwise[pairwise_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        "",
        "--- T_R SANITY CHECK ---",
        f"t_r monotonically decreases with mu for all strategies: {'YES' if t_r_passed else 'NO'}",
        t_r_sanity.to_string(index=False),
        "",
        "--- R_RB SANITY CHECK ---",
        f"R_rb identical across (mu, strategy) within paired_id: {'YES' if r_rb_passed else 'NO'}",
        f"Checked paired groups: {len(r_rb_sanity)}",
        "",
        "--- ACCEPTANCE NOTES ---",
        f"Raw paired design: {len(paired_counts)} (topology, paired_id) groups; expected rows per group = {expected_rows_per_pair}.",
        f"Raw paired rows complete: {'YES' if (paired_counts == expected_rows_per_pair).all() else 'NO'}",
        f"Invalid runs: {len(invalid)}",
        (
            "Invalid reasons: "
            + (
                invalid["exclusion_reason"].value_counts(dropna=False).to_string()
                if not invalid.empty else "none"
            )
        ),
        "",
        "--- INTERPRETATION ---",
        "R3 tests whether the advantage of intelligent recovery is an early-recovery effect. If μ grows, more failed nodes are repaired per step, so the ordering decision has fewer opportunities to matter. The central evidence is the sign and magnitude of Spearman r(μ, Δρ_CVaR_flow-random).",
        "",
        "--- GENERATED FIGURES ---",
        *[str(path.relative_to(ROOT)) for path in figures],
        "",
    ]
    return "\n".join(lines)


def save_artifacts(
    desc: pd.DataFrame,
    spearman: pd.DataFrame,
    gain: pd.DataFrame,
    gain_spearman: pd.DataFrame,
    pairwise: pd.DataFrame,
    r_rb_sanity: pd.DataFrame,
    t_r_sanity: pd.DataFrame,
    report: str,
) -> None:
    """Зберігає CSV і TXT артефакти аналізу."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(AGG_DIR / "R3_descriptive_statistics.csv", index=False)
    spearman.to_csv(AGG_DIR / "R3_raw_spearman.csv", index=False)
    gain.to_csv(AGG_DIR / "R3_strategy_gain.csv", index=False)
    gain_spearman.to_csv(AGG_DIR / "R3_gain_spearman.csv", index=False)
    pairwise.to_csv(AGG_DIR / "R3_pairwise_wilcoxon.csv", index=False)
    r_rb_sanity.to_csv(AGG_DIR / "R3_R_rb_sanity_check.csv", index=False)
    t_r_sanity.to_csv(AGG_DIR / "R3_t_r_sanity_check.csv", index=False)
    (AGG_DIR / "R3_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    """CLI entrypoint для аналізу R3."""
    raw = attach_t_r(load_raw())
    desc = descriptive_stats(raw)
    spearman = raw_spearman_analysis(raw)
    gain = strategy_gain(desc)
    gain_spearman = gain_spearman_analysis(gain)
    pairwise = paired_wilcoxon_tests(raw)
    r_rb_sanity = r_rb_sanity_check(raw)
    t_r_sanity = t_r_sanity_check(raw, desc)
    figures = make_figures(desc, gain)
    report = make_report(
        raw,
        desc,
        spearman,
        gain,
        gain_spearman,
        pairwise,
        r_rb_sanity,
        t_r_sanity,
        figures,
    )
    save_artifacts(
        desc,
        spearman,
        gain,
        gain_spearman,
        pairwise,
        r_rb_sanity,
        t_r_sanity,
        report,
    )
    print(report)
    print(f"\nSaved report: {AGG_DIR / 'R3_report.txt'}")


if __name__ == "__main__":
    main()

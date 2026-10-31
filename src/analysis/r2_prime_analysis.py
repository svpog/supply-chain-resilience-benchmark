"""
Аналіз серії R2' для статті 3.

Реалізує: Protocol R2' + Math Spec Article 3 Patch v3.2
(див. docs/Протокол_R2_prime.docx та docs/Math_Spec_Article_3_Patch_v3_2.docx).

Приклад використання:
    python -m src.analysis.r2_prime_analysis
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "R2_prime"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]
C_BACKUP_VALUES = [100, 300, 500, 1000]
ALPHA = 0.10
BOOTSTRAP_B = 1000
PERMUTATION_B = 10_000


def _clean(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
    """Повертає float-масив без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap/permutation за текстовими частинами."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_ci(
    values: pd.Series | list[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    B: int = BOOTSTRAP_B,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI для заданої статистики."""
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
    """Spearman r через ранги; швидше для permutation test."""
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
    """
    Двосторонній permutation p-value для Spearman r.

    У R2' це основний спосіб оцінити p-value на сирих paired-даних, а не
    на чотирьох агрегованих точках c_backup.
    """
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
    p_value = (exceed + 1) / (B + 1)
    return (observed, float(p_value))


def load_raw() -> pd.DataFrame:
    """Завантажує raw_results.csv R2'."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"R2' raw_results.csv not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("R2' raw_results.csv is empty.")
    raw["valid"] = raw["valid"].astype(bool)
    raw["c_backup"] = raw["c_backup"].astype(float)
    return raw


def load_actions() -> pd.DataFrame:
    """Завантажує recovery_actions_log.parquet R2'."""
    path = RAW_DIR / "recovery_actions_log.parquet"
    if not path.exists():
        raise FileNotFoundError(f"R2' recovery log not found: {path}")
    actions = pd.read_parquet(path)
    if actions.empty:
        raise ValueError("R2' recovery log is empty.")
    actions["c_backup"] = actions["c_backup"].astype(float)
    actions["has_backup_neighbor"] = actions["has_backup_neighbor"].astype(bool)
    return actions


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для 12 комбінацій c_backup × strategy."""
    rows: list[dict] = []
    for (c_backup, strategy), group_all in raw.groupby(["c_backup", "recovery_strategy"]):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)

        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("R2_prime", c_backup, strategy, "rho_med"),
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("R2_prime", c_backup, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho,
            compute_rho_IQR,
            seed=stable_seed("R2_prime", c_backup, strategy, "rho_iqr"),
        )
        r_rb_ci = bootstrap_ci(
            r_rb,
            np.median,
            seed=stable_seed("R2_prime", c_backup, strategy, "r_rb"),
        )

        rows.append(
            {
                "c_backup": float(c_backup),
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
    return pd.DataFrame(rows).sort_values(["c_backup", "recovery_strategy"]).reset_index(drop=True)


def _tail_rows_for_cvar(raw_strategy: pd.DataFrame) -> pd.DataFrame:
    """
    Вибирає нижній α-хвіст ρ окремо для кожного c_backup.

    Так Spearman для rho_CVaR оцінюється на сирих хвостових прогонах, не на
    чотирьох агрегованих CVaR-точках.
    """
    parts: list[pd.DataFrame] = []
    for _, group in raw_strategy.groupby("c_backup"):
        valid = group[group["valid"]].sort_values("rho", ascending=True)
        n_tail = max(1, int(np.floor(ALPHA * len(valid))))
        parts.append(valid.head(n_tail))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def raw_spearman_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(c_backup, metric) на сирих даних для rho і tail-CVaR."""
    rows: list[dict] = []
    for strategy in STRATEGIES:
        group = raw[(raw["valid"]) & (raw["recovery_strategy"] == strategy)].copy()
        for metric, source in [
            ("rho_med", group),
            ("rho_CVaR_0.10", _tail_rows_for_cvar(group)),
        ]:
            if source.empty:
                r_value, p_perm = float("nan"), float("nan")
                scipy_p = float("nan")
                n_raw = 0
            else:
                x = source["c_backup"].astype(float).to_numpy()
                y = source["rho"].astype(float).to_numpy()
                r_value, p_perm = permutation_spearman_p(
                    x,
                    y,
                    B=PERMUTATION_B,
                    seed=stable_seed("R2_prime", strategy, metric, "perm"),
                )
                scipy_r, scipy_p = spearmanr(x, y)
                if not math.isnan(r_value) and not math.isnan(scipy_r):
                    r_value = float(scipy_r)
                n_raw = len(source)
            rows.append(
                {
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
    """Δρ_CVaR і Δρ_med: flow_greedy − random для кожного c_backup."""
    random = desc[desc["recovery_strategy"] == "random"].set_index("c_backup")
    flow = desc[desc["recovery_strategy"] == "flow_greedy"].set_index("c_backup")
    rows: list[dict] = []
    for c_backup in C_BACKUP_VALUES:
        c = float(c_backup)
        if c not in random.index or c not in flow.index:
            continue
        rows.append(
            {
                "c_backup": c,
                "delta_rho_med_flow_minus_random": float(
                    flow.loc[c, "rho_med"] - random.loc[c, "rho_med"]
                ),
                "delta_rho_CVaR_flow_minus_random": float(
                    flow.loc[c, "rho_CVaR_0.10"] - random.loc[c, "rho_CVaR_0.10"]
                ),
            }
        )
    gain = pd.DataFrame(rows)
    if len(gain) >= 3:
        r_value, p_perm = permutation_spearman_p(
            gain["c_backup"],
            gain["delta_rho_CVaR_flow_minus_random"],
            B=PERMUTATION_B,
            seed=stable_seed("R2_prime", "strategy_gain", "perm"),
        )
    else:
        r_value, p_perm = float("nan"), float("nan")
    gain["spearman_gain_r"] = float(r_value)
    gain["spearman_gain_p_perm"] = float(p_perm)
    return gain


def recovery_log_summary(actions: pd.DataFrame) -> pd.DataFrame:
    """Агрегує node-level recovery log за c_backup × strategy."""
    rows: list[dict] = []
    for (c_backup, strategy), group in actions.groupby(["c_backup", "recovery_strategy"]):
        first3 = group[group["t"] <= 3]
        trunk_total = float(group["newly_unlocked_trunk_capacity"].sum())
        backup_total = float(group["newly_unlocked_backup_capacity"].sum())
        ratio = trunk_total / backup_total if backup_total > 0.0 else float("inf")
        rows.append(
            {
                "c_backup": float(c_backup),
                "recovery_strategy": strategy,
                "has_backup_share": float(group["has_backup_neighbor"].mean()),
                "has_backup_share_first3": (
                    float(first3["has_backup_neighbor"].mean()) if len(first3) else np.nan
                ),
                "mean_trunk_unlocked_per_repair": float(
                    group["newly_unlocked_trunk_capacity"].mean()
                ),
                "mean_backup_unlocked_per_repair": float(
                    group["newly_unlocked_backup_capacity"].mean()
                ),
                "trunk_to_backup_capacity_ratio": float(ratio),
                "n_actions": int(len(group)),
                "n_runs_logged": int(group["run_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["c_backup", "recovery_strategy"]).reset_index(drop=True)


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє ідентичність R_rb по стратегіях у межах (c_backup, paired_id)."""
    valid = raw[raw["valid"]].copy()
    rows: list[dict] = []
    for (c_backup, paired_id), group in valid.groupby(["c_backup", "paired_id"]):
        values = group["R_rb"].astype(float).to_numpy()
        spread = float(np.nanmax(values) - np.nanmin(values)) if len(values) else float("nan")
        rows.append(
            {
                "c_backup": float(c_backup),
                "paired_id": paired_id,
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
            "R_rb varies across strategies for at least one (c_backup, paired_id):\n"
            + bad.to_string(index=False)
        )
    return sanity


def hr3_prime_status(spearman: pd.DataFrame, gain: pd.DataFrame) -> tuple[str, str]:
    """Формальна перевірка HR3' за Patch v3.2."""
    flow = spearman[
        (spearman["recovery_strategy"] == "flow_greedy")
        & (spearman["metric"] == "rho_CVaR_0.10")
    ].iloc[0]
    flow_r = float(flow["spearman_r_raw"])
    flow_p = float(flow["p_value_permutation"])

    gain_index = gain.set_index("c_backup")
    if 100.0 in gain_index.index and 1000.0 in gain_index.index:
        gain_increase = float(
            gain_index.loc[1000.0, "delta_rho_CVaR_flow_minus_random"]
            - gain_index.loc[100.0, "delta_rho_CVaR_flow_minus_random"]
        )
    else:
        gain_increase = float("nan")

    gain_values = gain.sort_values("c_backup")["delta_rho_CVaR_flow_minus_random"].to_numpy()
    gain_grows = bool(len(gain_values) >= 2 and gain_values[-1] > gain_values[0])
    monotone_gain = bool(np.all(np.diff(gain_values) >= -1e-12)) if len(gain_values) else False

    if flow_r >= 0.5 and flow_p < 0.05 and gain_increase >= 0.05 and gain_grows:
        return (
            "CONFIRMED",
            "flow_greedy має r_raw(c_backup, tail-rho) ≥ 0.5, p_perm < 0.05, "
            "Δρ_CVaR gain зростає щонайменше на 0.05 між c_backup=100 і 1000.",
        )
    if 0.3 <= flow_r < 0.5:
        return (
            "PARTIALLY_CONFIRMED",
            "flow_greedy має помірну raw Spearman тенденцію (0.3 ≤ r < 0.5), "
            "але не досягає порогу повного підтвердження HR3'.",
        )
    if flow_r >= 0.5 and (gain_increase < 0.05 or not gain_grows):
        return (
            "PARTIALLY_CONFIRMED",
            "resilience flow_greedy зростає з c_backup, але strategy gain над random "
            "не проходить поріг Patch v3.2.",
        )
    return (
        "REJECTED",
        "flow_greedy не демонструє достатньо сильного raw-data зв'язку "
        "між c_backup і tail resilience за порогами Patch v3.2.",
    )


def localized_name(value: str, lang: str) -> str:
    """Локалізована назва стратегії з fallback."""
    try:
        return label(f"common.{value}", lang)
    except KeyError:
        return value


def _format_cbackup_axis(ax: plt.Axes) -> None:
    ax.set_xticks(C_BACKUP_VALUES)
    ax.set_xscale("linear")
    ax.grid(alpha=0.25)


def plot_metric_vs_cbackup(desc: pd.DataFrame, lang: str, metric: str) -> plt.Figure:
    """Лінійний графік metric vs c_backup з bootstrap CI."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if metric == "rho_med":
        y_col, low_col, high_col = "rho_med", "rho_med_ci_low", "rho_med_ci_high"
        title_key = "R2_prime.rho_vs_cbackup.title"
        y_label_key = "R2_prime.rho_vs_cbackup.y_label"
    else:
        y_col, low_col, high_col = "rho_CVaR_0.10", "rho_CVaR_ci_low", "rho_CVaR_ci_high"
        title_key = "R2_prime.cvar_vs_cbackup.title"
        y_label_key = "common.rho_cvar"

    for strategy in STRATEGIES:
        group = desc[desc["recovery_strategy"] == strategy].sort_values("c_backup")
        x = group["c_backup"].astype(float)
        y = group[y_col].astype(float)
        low = group[low_col].astype(float)
        high = group[high_col].astype(float)
        ax.plot(x, y, marker="o", linewidth=2, label=localized_name(strategy, lang))
        ax.fill_between(x, low, high, alpha=0.16)

    ax.set_title(label(title_key, lang))
    ax.set_xlabel(label("R2_prime.rho_vs_cbackup.x_label", lang))
    ax.set_ylabel(label(y_label_key, lang))
    _format_cbackup_axis(ax)
    ax.legend(title=label("common.strategy", lang), loc="best")
    return fig


def plot_rho_vs_cbackup(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_cbackup(desc, lang=lang, metric="rho_med")


def plot_cvar_vs_cbackup(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_cbackup(desc, lang=lang, metric="rho_CVaR_0.10")


def plot_strategy_gain(gain: pd.DataFrame, lang: str) -> plt.Figure:
    """Δρ_CVaR(flow_greedy − random) за c_backup."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = gain.sort_values("c_backup")
    ax.plot(
        data["c_backup"],
        data["delta_rho_CVaR_flow_minus_random"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(label("R2_prime.strategy_gain.title", lang))
    ax.set_xlabel(label("R2_prime.rho_vs_cbackup.x_label", lang))
    ax.set_ylabel(label("R2_prime.strategy_gain.y_label", lang))
    _format_cbackup_axis(ax)
    return fig


def plot_capacity_ratio(log_summary: pd.DataFrame, lang: str) -> plt.Figure:
    """Trunk/backup unlocked capacity ratio за c_backup і стратегією."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strategy in STRATEGIES:
        group = log_summary[log_summary["recovery_strategy"] == strategy].sort_values("c_backup")
        ax.plot(
            group["c_backup"],
            group["trunk_to_backup_capacity_ratio"].replace(np.inf, np.nan),
            marker="o",
            linewidth=2,
            label=localized_name(strategy, lang),
        )
    ax.set_title(label("R2_prime.capacity_ratio.title", lang))
    ax.set_xlabel(label("R2_prime.rho_vs_cbackup.x_label", lang))
    ax.set_ylabel("trunk / backup")
    _format_cbackup_axis(ax)
    ax.legend(title=label("common.strategy", lang), loc="best")
    return fig


def make_figures(
    desc: pd.DataFrame,
    gain: pd.DataFrame,
    log_summary: pd.DataFrame,
) -> list[Path]:
    """Зберігає 4 типи рисунків у двох мовах."""
    outputs: list[Path] = []
    outputs += save_dual_lang(
        plot_rho_vs_cbackup,
        str(FIG_DIR / "R2_prime_rho_vs_cbackup_{lang}.png"),
        desc=desc,
    )
    outputs += save_dual_lang(
        plot_cvar_vs_cbackup,
        str(FIG_DIR / "R2_prime_cvar_vs_cbackup_{lang}.png"),
        desc=desc,
    )
    outputs += save_dual_lang(
        plot_strategy_gain,
        str(FIG_DIR / "R2_prime_strategy_gain_{lang}.png"),
        gain=gain,
    )
    outputs += save_dual_lang(
        plot_capacity_ratio,
        str(FIG_DIR / "R2_prime_capacity_ratio_{lang}.png"),
        log_summary=log_summary,
    )
    return outputs


def make_report(
    raw: pd.DataFrame,
    desc: pd.DataFrame,
    spearman: pd.DataFrame,
    gain: pd.DataFrame,
    log_summary: pd.DataFrame,
    sanity: pd.DataFrame,
    figures: list[Path],
) -> str:
    """Формує TXT-звіт R2'."""
    status, status_reason = hr3_prime_status(spearman, gain)
    all_r_below_03 = bool((spearman["spearman_r_raw"].fillna(-1.0) < 0.3).all())
    sanity_passed = bool(sanity["passed"].all()) if not sanity.empty else False
    invalid = raw[~raw["valid"]]
    paired_counts = raw.groupby("paired_id").size()
    expected_rows_per_pair = len(C_BACKUP_VALUES) * len(STRATEGIES)

    desc_cols = [
        "c_backup",
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
        "non_recovered_pct",
    ]
    spearman_cols = [
        "recovery_strategy",
        "metric",
        "spearman_r_raw",
        "p_value_permutation",
        "n_raw_points",
    ]
    gain_cols = [
        "c_backup",
        "delta_rho_med_flow_minus_random",
        "delta_rho_CVaR_flow_minus_random",
        "spearman_gain_r",
        "spearman_gain_p_perm",
    ]
    log_cols = [
        "c_backup",
        "recovery_strategy",
        "has_backup_share",
        "has_backup_share_first3",
        "mean_trunk_unlocked_per_repair",
        "mean_backup_unlocked_per_repair",
        "trunk_to_backup_capacity_ratio",
        "n_actions",
    ]

    lines = [
        "============================================================",
        "R2': BACKUP CAPACITY SENSITIVITY (HR3' verification)",
        "============================================================",
        "",
        "Parameters: topology=reserved, K_res=3, c_backup ∈ {100,300,500,1000},",
        "            strategies={random, centrality_betweenness, flow_greedy},",
        "            N=50, H=4, b=3, p_v=0.20, T=100, M=100, paired seed design",
        "",
        "--- DESCRIPTIVE STATISTICS (12 rows) ---",
        desc[desc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- RAW-DATA SPEARMAN + PERMUTATION TESTS ---",
        "rho_med uses all valid raw runs; rho_CVaR_0.10 uses the bottom 10% tail within each c_backup group.",
        spearman[spearman_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- STRATEGY GAIN: flow_greedy − random ---",
        gain[gain_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- RECOVERY LOG SUMMARY ---",
        log_summary[log_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- R_RB SANITY CHECK ---",
        f"R_rb identical across strategies for each (c_backup, paired_id): {'YES' if sanity_passed else 'NO'}",
        f"Checked paired groups: {len(sanity)}",
        "",
        "--- HR3' STATUS (Patch v3.2 thresholds) ---",
        status,
        status_reason,
        f"All raw Spearman r < 0.3: {'YES' if all_r_below_03 else 'NO'}",
        "",
        "--- ACCEPTANCE NOTES ---",
        f"Raw paired design: {len(paired_counts)} paired_id values; expected rows per paired_id = {expected_rows_per_pair}.",
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
        "R2' tests whether increasing backup-edge capacity makes reserved redundancy functionally useful for recovery strategies, especially flow_greedy. The central criterion is raw-data Spearman r(c_backup, tail resilience) plus growth of Δρ_CVaR(flow_greedy − random).",
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
    log_summary: pd.DataFrame,
    sanity: pd.DataFrame,
    report: str,
) -> None:
    """Зберігає CSV і TXT артефакти аналізу."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(AGG_DIR / "R2_prime_descriptive_statistics.csv", index=False)
    spearman.to_csv(AGG_DIR / "R2_prime_raw_spearman.csv", index=False)
    gain.to_csv(AGG_DIR / "R2_prime_strategy_gain.csv", index=False)
    log_summary.to_csv(AGG_DIR / "R2_prime_recovery_log_summary.csv", index=False)
    sanity.to_csv(AGG_DIR / "R2_prime_R_rb_sanity_check.csv", index=False)
    (AGG_DIR / "R2_prime_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    """CLI entrypoint для аналізу R2'."""
    raw = load_raw()
    actions = load_actions()
    desc = descriptive_stats(raw)
    spearman = raw_spearman_analysis(raw)
    gain = strategy_gain(desc)
    log_summary = recovery_log_summary(actions)
    sanity = r_rb_sanity_check(raw)
    figures = make_figures(desc, gain, log_summary)
    report = make_report(raw, desc, spearman, gain, log_summary, sanity, figures)
    save_artifacts(desc, spearman, gain, log_summary, sanity, report)
    print(report)
    print(f"\nSaved report: {AGG_DIR / 'R2_prime_report.txt'}")


if __name__ == "__main__":
    main()

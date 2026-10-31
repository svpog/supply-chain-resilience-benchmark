"""
Аналіз діагностичної серії D2 для статті 3.

Реалізує: Protocol D2 + Math Spec Article 3 Patch v3.3
(див. docs/Протокол_D2.docx та docs/Math_Spec_Article_3_Patch_v3_3.docx).

Приклад використання:
    python -m src.analysis.d2_analysis
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
from scipy.stats import rankdata, wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

SERIES = ["D2a", "D2b"]
STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]
K_VALUES = [0, 1, 2, 3]
ALPHA = 0.10
BOOTSTRAP_B = 2000
PERMUTATION_B = 10_000


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


def _spearman_fast(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Spearman r через ранги; використовується у permutation test."""
    if len(x_values) < 3:
        return float("nan")
    rx = rankdata(x_values)
    ry = rankdata(y_values)
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
    observed = _spearman_fast(x, y)
    if math.isnan(observed):
        return (observed, float("nan"))

    rng = np.random.default_rng(seed)
    rx = rankdata(x)
    ry = rankdata(y)
    exceed = 0
    for _ in range(B):
        r_perm = float(np.corrcoef(rx, rng.permutation(ry))[0, 1])
        if abs(r_perm) >= abs(observed) - 1e-15:
            exceed += 1
    return (observed, float((exceed + 1) / (B + 1)))


def load_raw() -> pd.DataFrame:
    """Завантажує raw_results.csv для D2a і D2b."""
    frames: list[pd.DataFrame] = []
    for series in SERIES:
        path = RAW_ROOT / series / "raw_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"{series} raw_results.csv not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{series} raw_results.csv is empty.")
        df["subseries"] = series
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["valid"] = raw["valid"].astype(bool)
    raw["K_res"] = raw["K_res"].astype(int)
    raw["rho"] = raw["rho"].astype(float)
    raw["R_rb"] = raw["R_rb"].astype(float)
    return raw


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для 24 комбінацій subseries × K_res × strategy."""
    rows: list[dict] = []
    for (series, k_res, strategy), group_all in raw.groupby(
        ["subseries", "K_res", "recovery_strategy"]
    ):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)
        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed(series, k_res, strategy, "rho_med"),
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed(series, k_res, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho,
            compute_rho_IQR,
            seed=stable_seed(series, k_res, strategy, "rho_iqr"),
        )

        rows.append(
            {
                "subseries": series,
                "K_res": int(k_res),
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
                "R_rb_lt_0_95_pct": float((r_rb < 0.95).mean() * 100.0)
                if len(r_rb) else float("nan"),
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0)
                if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["subseries", "K_res", "recovery_strategy"]
    ).reset_index(drop=True)


def severity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Частка прогонах з R_rb < 0.95.

    R_rb однаковий по стратегіях у межах (subseries, K_res, paired_id), тому
    для чистої оцінки severity дублікати strategy прибираються.
    """
    valid = raw[raw["valid"]].drop_duplicates(["subseries", "K_res", "paired_id"])
    rows: list[dict] = []
    for series, group in valid.groupby("subseries"):
        rows.append(
            {
                "subseries": series,
                "n_unique_runs": int(len(group)),
                "R_rb_lt_0_95_pct": float((group["R_rb"] < 0.95).mean() * 100.0),
                "R_rb_median": float(group["R_rb"].median()),
                "R_rb_min": float(group["R_rb"].min()),
            }
        )
    for (series, k_res), group in valid.groupby(["subseries", "K_res"]):
        rows.append(
            {
                "subseries": f"{series}_K{k_res}",
                "n_unique_runs": int(len(group)),
                "R_rb_lt_0_95_pct": float((group["R_rb"] < 0.95).mean() * 100.0),
                "R_rb_median": float(group["R_rb"].median()),
                "R_rb_min": float(group["R_rb"].min()),
            }
        )
    return pd.DataFrame(rows)


def _tail_rows_for_cvar(group: pd.DataFrame) -> pd.DataFrame:
    """Вибирає нижній α-хвіст ρ окремо для кожного K_res."""
    parts: list[pd.DataFrame] = []
    for _, k_group in group.groupby("K_res"):
        valid = k_group[k_group["valid"]].sort_values("rho", ascending=True)
        n_tail = max(1, int(np.floor(ALPHA * len(valid))))
        parts.append(valid.head(n_tail))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def raw_spearman_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(K_res, metric) на сирих даних для rho і tail-CVaR."""
    rows: list[dict] = []
    for (series, strategy), group_all in raw.groupby(["subseries", "recovery_strategy"]):
        group = group_all[group_all["valid"]].copy()
        r_rho, p_rho = permutation_spearman_p(
            group["K_res"].astype(float),
            group["rho"].astype(float),
            seed=stable_seed(series, strategy, "rho"),
        )
        rows.append(
            {
                "subseries": series,
                "recovery_strategy": strategy,
                "metric": "rho_raw",
                "n": int(len(group)),
                "spearman_r": r_rho,
                "p_perm": p_rho,
            }
        )

        tail = _tail_rows_for_cvar(group_all)
        r_cvar, p_cvar = permutation_spearman_p(
            tail["K_res"].astype(float),
            tail["rho"].astype(float),
            seed=stable_seed(series, strategy, "rho_cvar"),
        )
        rows.append(
            {
                "subseries": series,
                "recovery_strategy": strategy,
                "metric": "rho_CVaR_tail_raw",
                "n": int(len(tail)),
                "spearman_r": r_cvar,
                "p_perm": p_cvar,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["subseries", "recovery_strategy", "metric"]
    ).reset_index(drop=True)


def delta_cvar_k3_minus_k0(raw: pd.DataFrame) -> pd.DataFrame:
    """Δρ_CVaR(K_res=3 − K_res=0) з bootstrap CI і Wilcoxon paired test."""
    rows: list[dict] = []
    for (series, strategy), group_all in raw.groupby(["subseries", "recovery_strategy"]):
        group = group_all[group_all["valid"]].copy()
        pivot = group.pivot_table(
            index="paired_id",
            columns="K_res",
            values="rho",
            aggfunc="first",
        )
        pivot = pivot.dropna(subset=[0, 3])
        k0 = pivot[0].to_numpy(dtype=float)
        k3 = pivot[3].to_numpy(dtype=float)
        observed = compute_rho_CVaR(k3, alpha=ALPHA) - compute_rho_CVaR(k0, alpha=ALPHA)

        rng = np.random.default_rng(stable_seed(series, strategy, "delta_cvar"))
        boot = np.empty(BOOTSTRAP_B, dtype=float)
        n = len(pivot)
        for i in range(BOOTSTRAP_B):
            idx = rng.integers(0, n, size=n)
            boot[i] = compute_rho_CVaR(k3[idx], alpha=ALPHA) - compute_rho_CVaR(
                k0[idx], alpha=ALPHA
            )
        try:
            w_stat, p_wilcoxon = wilcoxon(k3, k0, alternative="greater")
        except ValueError:
            w_stat, p_wilcoxon = float("nan"), float("nan")

        cvar_by_k = [
            compute_rho_CVaR(
                group[group["K_res"] == k]["rho"].astype(float),
                alpha=ALPHA,
            )
            for k in K_VALUES
        ]
        monotonic = all(cvar_by_k[i + 1] >= cvar_by_k[i] - 1e-12 for i in range(3))

        rows.append(
            {
                "subseries": series,
                "recovery_strategy": strategy,
                "n_pairs": int(n),
                "rho_CVaR_K0": cvar_by_k[0],
                "rho_CVaR_K3": cvar_by_k[3],
                "delta_CVaR_K3_minus_K0": observed,
                "delta_ci_low": float(np.quantile(boot, 0.025)),
                "delta_ci_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p_greater": float(p_wilcoxon),
                "monotonic_non_decreasing": bool(monotonic),
            }
        )
    return pd.DataFrame(rows).sort_values(["subseries", "recovery_strategy"]).reset_index(drop=True)


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє, що R_rb ідентичний по стратегіях у межах K_res×paired_id."""
    rows: list[dict] = []
    for (series, k_res, paired_id), group in raw[raw["valid"]].groupby(
        ["subseries", "K_res", "paired_id"]
    ):
        values = group["R_rb"].dropna().to_numpy(dtype=float)
        max_diff = float(np.max(values) - np.min(values)) if len(values) else float("nan")
        passed = bool(len(values) > 0 and max_diff <= 1e-9)
        rows.append(
            {
                "subseries": series,
                "K_res": int(k_res),
                "paired_id": paired_id,
                "n_strategies": int(group["recovery_strategy"].nunique()),
                "max_R_rb_diff": max_diff,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def hr3_status(
    severity: pd.DataFrame,
    spearman: pd.DataFrame,
    delta: pd.DataFrame,
) -> tuple[str, str]:
    """Формальна перевірка HR3'' за Protocol D2."""
    d2b_severity = float(
        severity.loc[severity["subseries"] == "D2b", "R_rb_lt_0_95_pct"].iloc[0]
    )
    if d2b_severity < 50.0:
        return (
            "SEVERITY_FAIL",
            "D2b не просаджує мережу достатньо: частка R_rb < 0.95 нижча за 50%.",
        )

    d2b_cvar = spearman[
        (spearman["subseries"] == "D2b")
        & (spearman["metric"] == "rho_CVaR_tail_raw")
    ]
    d2b_delta = delta[delta["subseries"] == "D2b"]
    random_delta = d2b_delta[d2b_delta["recovery_strategy"] == "random"].iloc[0]
    all_r_ge_05 = bool((d2b_cvar["spearman_r"] >= 0.5).all())
    all_monotonic = bool(d2b_delta["monotonic_non_decreasing"].all())
    all_r_lt_03 = bool((d2b_cvar["spearman_r"] < 0.3).all())

    if (
        all_r_ge_05
        and random_delta["delta_CVaR_K3_minus_K0"] >= 0.05
        and random_delta["delta_ci_low"] > 0.0
    ):
        return (
            "CONFIRMED",
            "D2b має сильний позитивний K_res тренд і Δρ_CVaR(random) ≥ 0.05.",
        )
    if all_r_lt_03:
        return (
            "REJECTED",
            "У D2b Spearman r < 0.3 для всіх стратегій; K_res ефект не активувався.",
        )
    if all_monotonic:
        return (
            "PARTIALLY",
            "Тренд монотонний, але не досягає повного порогу CONFIRMED.",
        )
    return (
        "PARTIALLY",
        "Є змішаний позитивний сигнал, але монотонність або пороги не виконані повністю.",
    )


def _strategy_label(strategy: str, lang: str) -> str:
    """Локалізований підпис strategy."""
    return label(f"common.{strategy}", lang)


def _series_label(series: str, lang: str) -> str:
    """Локалізований підпис D2a/D2b."""
    key = "D2.canonical" if series == "D2a" else "D2.enhanced"
    return label(key, lang)


def plot_severity(raw: pd.DataFrame, lang: str = "uk") -> plt.Figure:
    """Boxplot R_rb для D2a vs D2b."""
    plot_df = raw[raw["valid"]].drop_duplicates(["subseries", "K_res", "paired_id"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = [plot_df[plot_df["subseries"] == series]["R_rb"].astype(float) for series in SERIES]
    ax.boxplot(data, tick_labels=[_series_label(series, lang) for series in SERIES])
    ax.axhline(0.95, color="#b91c1c", linestyle="--", linewidth=1.2, label="R_rb = 0.95")
    ax.set_title(label("D2.severity_check.title", lang))
    ax.set_xlabel(label("D2.severity_check.x_label", lang))
    ax.set_ylabel(label("D2.severity_check.y_label", lang))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_cvar_vs_kres(desc: pd.DataFrame, lang: str = "uk") -> plt.Figure:
    """ρ_CVaR_0.10 vs K_res для D2a/D2b."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    colors = {
        "random": "#2563eb",
        "centrality_betweenness": "#16a34a",
        "flow_greedy": "#dc2626",
    }
    for ax, series in zip(axes, SERIES):
        sub = desc[desc["subseries"] == series]
        for strategy in STRATEGIES:
            g = sub[sub["recovery_strategy"] == strategy].sort_values("K_res")
            y = g["rho_CVaR_0.10"].to_numpy(dtype=float)
            lo = g["rho_CVaR_ci_low"].to_numpy(dtype=float)
            hi = g["rho_CVaR_ci_high"].to_numpy(dtype=float)
            x = g["K_res"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=colors[strategy], label=_strategy_label(strategy, lang))
            ax.fill_between(x, lo, hi, color=colors[strategy], alpha=0.14, linewidth=0)
        ax.set_title(_series_label(series, lang))
        ax.set_xlabel(label("D2.cvar_vs_Kres.x_label", lang))
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(label("D2.cvar_vs_Kres.y_label", lang))
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(label("D2.cvar_vs_Kres.title", lang), y=1.02)
    return fig


def plot_strategy_gain(delta: pd.DataFrame, lang: str = "uk") -> plt.Figure:
    """Bar chart Δρ_CVaR(K=3−K=0) для D2a/D2b."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.22
    x = np.arange(len(STRATEGIES))
    colors = {"D2a": "#64748b", "D2b": "#ea580c"}
    for offset, series in [(-width / 2, "D2a"), (width / 2, "D2b")]:
        g = delta[delta["subseries"] == series].set_index("recovery_strategy").loc[STRATEGIES]
        y = g["delta_CVaR_K3_minus_K0"].to_numpy(dtype=float)
        lo = g["delta_ci_low"].to_numpy(dtype=float)
        hi = g["delta_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.bar(
            x + offset,
            y,
            width=width,
            color=colors[series],
            label=_series_label(series, lang),
            yerr=yerr,
            capsize=4,
            alpha=0.88,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_strategy_label(strategy, lang) for strategy in STRATEGIES], rotation=15)
    ax.set_ylabel(label("D2.strategy_gain.y_label", lang))
    ax.set_title(label("D2.strategy_gain.title", lang))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return fig


def save_figures(raw: pd.DataFrame, desc: pd.DataFrame, delta: pd.DataFrame) -> list[Path]:
    """Зберігає 3 типи рисунків × 2 мови."""
    outputs: list[Path] = []
    outputs.extend(save_dual_lang(
        plot_severity,
        str(FIG_DIR / "D2_severity_check_{lang}.png"),
        raw=raw,
    ))
    outputs.extend(save_dual_lang(
        plot_cvar_vs_kres,
        str(FIG_DIR / "D2_cvar_vs_Kres_{lang}.png"),
        desc=desc,
    ))
    outputs.extend(save_dual_lang(
        plot_strategy_gain,
        str(FIG_DIR / "D2_strategy_gain_{lang}.png"),
        delta=delta,
    ))
    return outputs


def write_report(
    desc: pd.DataFrame,
    severity: pd.DataFrame,
    spearman: pd.DataFrame,
    delta: pd.DataFrame,
    sanity: pd.DataFrame,
    status: str,
    status_reason: str,
    figures: list[Path],
) -> Path:
    """Пише D2_diagnostic_report.txt."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / "D2_diagnostic_report.txt"
    sanity_passed = bool(sanity["passed"].all()) if not sanity.empty else False

    severity_main = severity[severity["subseries"].isin(SERIES)].copy()
    d2b_sev = float(
        severity_main.loc[severity_main["subseries"] == "D2b", "R_rb_lt_0_95_pct"].iloc[0]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("D2 DIAGNOSTIC REPORT — K_res sensitivity at edge_degradation\n")
        f.write("=" * 78 + "\n\n")
        f.write("Design\n")
        f.write("- Topology: reserved only\n")
        f.write("- K_res: 0, 1, 2, 3\n")
        f.write("- Strategies: random, centrality_betweenness, flow_greedy\n")
        f.write("- M: 50 paired seeds per configuration\n")
        f.write("- Disruption target: trunk_only\n")
        f.write("- D2a: p_e=0.20, gamma=0.50 → capacity = 50% of base\n")
        f.write(
            "- D2b використовує gamma=0.80 за формулою new_cap = "
            "(1 - gamma) * base_cap зі статті 1, що означає capacity → 20%.\n\n"
        )

        f.write("Disruption severity check\n")
        f.write(severity_main.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n")
        if d2b_sev < 50.0:
            f.write(
                "\nCRITICAL: D2b severity check FAILED. Fraction with R_rb < 0.95 "
                "is below 50%; HR3'' test is not methodologically meaningful.\n"
            )
        else:
            f.write("\nD2b severity check PASSED: at least 50% of runs have R_rb < 0.95.\n")

        f.write("\nDescriptive statistics\n")
        desc_cols = [
            "subseries", "K_res", "recovery_strategy", "n_valid", "n_invalid",
            "rho_med", "rho_CVaR_0.10", "R_rb_med", "R_rb_lt_0_95_pct",
        ]
        f.write(desc[desc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        f.write("\n\nSpearman on raw data\n")
        f.write(spearman.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        f.write("\n\nDelta rho_CVaR (K_res=3 minus K_res=0)\n")
        delta_cols = [
            "subseries", "recovery_strategy", "n_pairs", "rho_CVaR_K0",
            "rho_CVaR_K3", "delta_CVaR_K3_minus_K0", "delta_ci_low",
            "delta_ci_high", "wilcoxon_p_greater", "monotonic_non_decreasing",
        ]
        f.write(delta[delta_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        f.write("\n\nR_rb sanity check\n")
        f.write(f"Passed: {sanity_passed}\n")
        f.write(f"Checked groups: {len(sanity)}\n")
        if not sanity_passed:
            failed = sanity[~sanity["passed"]].head(10)
            f.write(failed.to_string(index=False))
            f.write("\n")

        f.write("\nFormal HR3'' status\n")
        f.write(f"Status: {status}\n")
        f.write(f"Reason: {status_reason}\n")

        f.write("\nFigures\n")
        for fig_path in figures:
            f.write(f"- {fig_path.relative_to(ROOT)}\n")

    return path


def save_tables(
    desc: pd.DataFrame,
    severity: pd.DataFrame,
    spearman: pd.DataFrame,
    delta: pd.DataFrame,
    sanity: pd.DataFrame,
) -> None:
    """Зберігає CSV-таблиці аналізу."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(AGG_DIR / "D2_descriptive_statistics.csv", index=False)
    severity.to_csv(AGG_DIR / "D2_severity_check.csv", index=False)
    spearman.to_csv(AGG_DIR / "D2_spearman.csv", index=False)
    delta.to_csv(AGG_DIR / "D2_delta_cvar.csv", index=False)
    sanity.to_csv(AGG_DIR / "D2_R_rb_sanity_check.csv", index=False)


def main() -> dict[str, object]:
    """Запускає повний аналіз D2 і повертає основні об'єкти."""
    raw = load_raw()
    desc = descriptive_stats(raw)
    severity = severity_check(raw)
    spearman = raw_spearman_analysis(raw)
    delta = delta_cvar_k3_minus_k0(raw)
    sanity = r_rb_sanity_check(raw)
    status, status_reason = hr3_status(severity, spearman, delta)
    figures = save_figures(raw, desc, delta)
    save_tables(desc, severity, spearman, delta, sanity)
    report_path = write_report(
        desc=desc,
        severity=severity,
        spearman=spearman,
        delta=delta,
        sanity=sanity,
        status=status,
        status_reason=status_reason,
        figures=figures,
    )

    print(f"Saved report: {report_path}")
    print(f"HR3'' status: {status}")
    return {
        "raw": raw,
        "descriptive": desc,
        "severity": severity,
        "spearman": spearman,
        "delta": delta,
        "sanity": sanity,
        "status": status,
        "report_path": report_path,
        "figures": figures,
    }


if __name__ == "__main__":
    main()

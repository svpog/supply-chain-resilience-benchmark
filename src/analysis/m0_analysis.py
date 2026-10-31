"""
Аналіз серії M0 для статті 4 після Patch v4.1.

Реалізує: Math Spec Article 4 Patch v4.1, §3 та Протокол M0
(див. docs/Math_Spec_Article_4_Patch_v4_1.docx).

Приклад використання:
    python -m src.analysis.m0_analysis
"""

from __future__ import annotations

import hashlib
import math
import shutil
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M0"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
R1_STATS_PATH = ROOT / "results" / "aggregated" / "article3" / "R1_descriptive_statistics.csv"

TOPOLOGIES = ["hierarchical", "dense", "hub", "reserved"]
R_ORDER = ["0", "1", "2", "3", "inf"]
FINITE_R_FOR_RSTAR = ["1", "2", "3"]
R_NUMERIC = {"0": 0, "1": 1, "2": 2, "3": 3, "inf": 4}
ALPHA = 0.10
BOOTSTRAP_B = 2000
PERMUTATION_B = 10_000
SANITY_TOL = 0.010
RIG_THRESHOLD = 80.0

LANG = {
    "uk": {
        "rig_title": "M0: відносний інформаційний виграш RIG(r)",
        "rig_y": "RIG, %",
        "rig_heatmap": "M0: теплова карта RIG(r), %",
        "rho_title": "M0: хвостова resilience ρ_CVaR за радіусом видимості",
        "rho_y": "ρ_CVaR_0.10",
        "rho_med_title": "M0: медіанна resilience ρ_med за радіусом видимості",
        "rho_med_y": "ρ_med",
        "x": "радіус видимості r",
        "topology": "топологія",
    },
    "en": {
        "rig_title": "M0: Relative Information Gain RIG(r)",
        "rig_y": "RIG, %",
        "rig_heatmap": "M0: RIG(r) heatmap, %",
        "rho_title": "M0: tail resilience ρ_CVaR by visibility radius",
        "rho_y": "ρ_CVaR_0.10",
        "rho_med_title": "M0: median resilience ρ_med by visibility radius",
        "rho_med_y": "ρ_med",
        "x": "visibility radius r",
        "topology": "topology",
    },
}


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
    """Percentile bootstrap 95% CI."""
    arr = _clean(values)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = statistic(sample)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def cliffs_delta(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Cliff's δ: positive якщо y загалом більший за x."""
    x_arr = _clean(x)
    y_arr = _clean(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return float("nan")
    diff = y_arr[:, None] - x_arr[None, :]
    return float(np.sum(np.sign(diff)) / (x_arr.size * y_arr.size))


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r через ранги."""
    if len(x) < 3:
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


def normalize_r(value: object) -> str:
    """Нормалізує r_visibility з CSV до {'0','1','2','3','inf'}."""
    text = str(value)
    if text in {"inf", "None", "nan"}:
        return "inf"
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_raw() -> pd.DataFrame:
    """Завантажує M0 raw_results.csv без зміни сирих даних."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"M0 raw_results.csv not found: {path}")
    raw = pd.read_csv(path, keep_default_na=False)
    raw["valid"] = raw["valid"].astype(str).str.lower().isin({"true", "1"})
    raw["r_label"] = raw["r_visibility"].apply(normalize_r)
    raw["r_numeric"] = raw["r_label"].map(R_NUMERIC)
    return raw


def load_r1_baselines() -> pd.DataFrame:
    """Завантажує R1 reference baselines для random та flow_greedy."""
    if not R1_STATS_PATH.exists():
        raise FileNotFoundError(f"R1 descriptive statistics not found: {R1_STATS_PATH}")
    r1 = pd.read_csv(R1_STATS_PATH)
    strategy_col = "recovery_strategy" if "recovery_strategy" in r1.columns else "strategy"
    required = {"topology", strategy_col, "rho_CVaR_0.10"}
    missing = required - set(r1.columns)
    if missing:
        raise ValueError(f"R1 descriptive statistics missing columns: {sorted(missing)}")

    baselines = r1.pivot_table(
        index="topology",
        columns=strategy_col,
        values="rho_CVaR_0.10",
        aggfunc="first",
    ).reset_index()
    if "random" not in baselines.columns or "flow_greedy" not in baselines.columns:
        raise ValueError("R1 statistics must include random and flow_greedy strategies.")
    return baselines.rename(
        columns={
            "random": "random_R1_rho_CVaR",
            "flow_greedy": "flow_greedy_R1_rho_CVaR",
        }
    )[["topology", "random_R1_rho_CVaR", "flow_greedy_R1_rho_CVaR"]]


def descriptive_stats(raw: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для topology × r з random_R1 reference."""
    rows: list[dict] = []
    baseline_map = baselines.set_index("topology")["random_R1_rho_CVaR"].to_dict()
    for (topology, r_label), group_all in raw.groupby(["topology", "r_label"]):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("M0", topology, r_label, "rho_cvar"),
        )
        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("M0", topology, r_label, "rho_med"),
        )
        rows.append(
            {
                "topology": topology,
                "r": r_label,
                "r_numeric": R_NUMERIC.get(r_label),
                "n_total": int(len(group_all)),
                "n_valid": int(len(group)),
                "n_invalid": int(len(group_all) - len(group)),
                "random_R1_rho_CVaR": float(baseline_map[topology]),
                "rho_med": compute_rho_med(rho),
                "rho_med_ci_low": rho_med_ci[0],
                "rho_med_ci_high": rho_med_ci[1],
                "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=ALPHA),
                "rho_CVaR_ci_low": rho_cvar_ci[0],
                "rho_CVaR_ci_high": rho_cvar_ci[1],
                "rho_IQR": compute_rho_IQR(rho),
                "R_rb_med": float(group["R_rb"].astype(float).median()) if len(group) else float("nan"),
                "non_recovered_pct": float(
                    group["non_recovered"].astype(str).str.lower().isin({"true", "1"}).mean() * 100.0
                )
                if len(group)
                else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["topology", "r_numeric"])
        .reset_index(drop=True)
    )


def rig_curve(desc: pd.DataFrame) -> pd.DataFrame:
    """Обчислює RIG(r)% за Patch v4.1, §3.1."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        sub = desc[desc["topology"] == topology].set_index("r")
        baseline = float(sub.loc["0", "random_R1_rho_CVaR"])
        cvar_inf = float(sub.loc["inf", "rho_CVaR_0.10"])
        denom = cvar_inf - baseline
        for r_label in R_ORDER:
            cvar_r = float(sub.loc[r_label, "rho_CVaR_0.10"])
            if abs(denom) <= 1e-12:
                rig_pct = float("nan")
            else:
                rig_pct = (cvar_r - baseline) / denom * 100.0
            rows.append(
                {
                    "topology": topology,
                    "r": r_label,
                    "r_numeric": R_NUMERIC[r_label],
                    "rho_CVaR_0.10": cvar_r,
                    "random_R1_rho_CVaR": baseline,
                    "rho_CVaR_inf": cvar_inf,
                    "denominator": denom,
                    "RIG_pct": rig_pct,
                }
            )
    return pd.DataFrame(rows).sort_values(["topology", "r_numeric"]).reset_index(drop=True)


def rstar_table(rig_df: pd.DataFrame) -> pd.DataFrame:
    """Визначає r* = min{r in {1,2,3}: RIG(r) >= 80%}; інакше inf."""
    rows: list[dict] = []
    for topology, group in rig_df.groupby("topology"):
        ordered = group.sort_values("r_numeric")
        denom = float(ordered["denominator"].iloc[0])
        finite = ordered[ordered["r"].isin(FINITE_R_FOR_RSTAR)]
        reached = finite[finite["RIG_pct"] >= RIG_THRESHOLD] if denom > 0.0 else pd.DataFrame()
        if reached.empty:
            r_star: str | int = "inf"
            rig_at_rstar = float(ordered.loc[ordered["r"] == "inf", "RIG_pct"].iloc[0])
            reached_finite = False
        else:
            r_star = int(reached.iloc[0]["r_numeric"])
            rig_at_rstar = float(reached.iloc[0]["RIG_pct"])
            reached_finite = True
        finite_values = finite["RIG_pct"].astype(float).to_numpy()
        rows.append(
            {
                "topology": topology,
                "denominator": denom,
                "valid_denominator": bool(denom > 0.0),
                "r_star": r_star,
                "RIG_at_r_star": rig_at_rstar,
                "reached_finite": bool(reached_finite),
                "visibility_trap": bool(not reached_finite),
                "partial_visibility_harmful": bool(np.all(finite_values < 0.0)) if finite_values.size else False,
                "r0_gt_random_R1": bool(
                    float(ordered.loc[ordered["r"] == "0", "RIG_pct"].iloc[0]) > 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("topology").reset_index(drop=True)


def sanity_vs_r1(desc: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Порівнює M0 r=inf із R1 flow_greedy. Перевірка Patch v4.1 лишається."""
    r1_fg = baselines[["topology", "flow_greedy_R1_rho_CVaR"]].rename(
        columns={"flow_greedy_R1_rho_CVaR": "rho_CVaR_R1_flow_greedy"}
    )
    m0_inf = desc[desc["r"] == "inf"][["topology", "rho_CVaR_0.10"]].rename(
        columns={"rho_CVaR_0.10": "rho_CVaR_M0_inf"}
    )
    out = m0_inf.merge(r1_fg, on="topology", how="left")
    out["delta_abs"] = (out["rho_CVaR_M0_inf"] - out["rho_CVaR_R1_flow_greedy"]).abs()
    out["passed"] = out["delta_abs"] <= SANITY_TOL
    return out.sort_values("topology").reset_index(drop=True)


def _tail_rows_for_cvar(raw_group: pd.DataFrame) -> pd.DataFrame:
    """Вибирає нижній α-хвіст ρ окремо для кожного r."""
    parts: list[pd.DataFrame] = []
    for _, group in raw_group.groupby("r_label"):
        valid = group[group["valid"]].sort_values("rho", ascending=True)
        n_tail = max(1, int(np.floor(ALPHA * len(valid))))
        parts.append(valid.head(n_tail))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def spearman_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(r_numeric, ρ/ρ_CVaR tail) на сирих даних."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        group = raw[(raw["valid"]) & (raw["topology"] == topology)].copy()
        for metric, source in [
            ("rho_med", group),
            ("rho_CVaR_0.10", _tail_rows_for_cvar(group)),
        ]:
            r_value, p_perm = permutation_spearman_p(
                source["r_numeric"].astype(float),
                source["rho"].astype(float),
                seed=stable_seed("M0", topology, metric, "spearman"),
            )
            rows.append(
                {
                    "topology": topology,
                    "metric": metric,
                    "n_raw": int(len(source)),
                    "spearman_r": r_value,
                    "p_perm": p_perm,
                }
            )
    return pd.DataFrame(rows)


def paired_wilcoxon(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon між усіма r у межах topology."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        sub = raw[(raw["valid"]) & (raw["topology"] == topology)]
        for left_r, right_r in combinations(R_ORDER, 2):
            left = sub[sub["r_label"] == left_r][["paired_id", "rho"]]
            right = sub[sub["r_label"] == right_r][["paired_id", "rho"]]
            merged = left.merge(right, on="paired_id", suffixes=("_left", "_right"))
            if len(merged) < 2:
                stat, p_value, delta, cliff = np.nan, np.nan, np.nan, np.nan
            else:
                left_values = merged["rho_left"].astype(float)
                right_values = merged["rho_right"].astype(float)
                try:
                    stat, p_value = wilcoxon(right_values, left_values, alternative="greater")
                except ValueError:
                    stat, p_value = 0.0, 1.0
                delta = float(right_values.median() - left_values.median())
                cliff = cliffs_delta(left_values, right_values)
            rows.append(
                {
                    "topology": topology,
                    "r_from": left_r,
                    "r_to": right_r,
                    "n_pairs": int(len(merged)),
                    "median_delta": delta,
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value),
                    "cliffs_delta": cliff,
                }
            )
    out = pd.DataFrame(rows)
    out["p_adj_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    out["significant"] = out["p_adj_bonferroni"] < 0.05
    return out


def determine_status(
    sanity: pd.DataFrame,
    rstar: pd.DataFrame,
) -> tuple[str, str, str, list[str]]:
    """Формальний статус HM0, HM0b, HM0c після Patch v4.1."""
    reasons: list[str] = []
    if not bool(sanity["passed"].all()):
        reasons.append("sanity_vs_R1_failed")
    if not bool(rstar["valid_denominator"].all()):
        bad = ", ".join(rstar.loc[~rstar["valid_denominator"], "topology"].tolist())
        reasons.append(f"invalid_RIG_denominator:{bad}")

    reached_count = int(rstar["reached_finite"].sum())
    if reached_count == len(rstar):
        hm0_status = "CONFIRMED"
    elif reached_count > 0:
        hm0_status = "PARTIALLY"
    else:
        hm0_status = "NOT_CONFIRMED"

    hier = rstar[rstar["topology"] == "hierarchical"].iloc[0]
    hm0b_status = "CONFIRMED" if bool(hier["partial_visibility_harmful"]) else "NOT_CONFIRMED"
    hm0c_status = "CONFIRMED" if bool(hier["r0_gt_random_R1"]) else "NOT_CONFIRMED"
    if reasons and hm0_status == "CONFIRMED":
        hm0_status = "PARTIALLY"
    return hm0_status, hm0b_status, hm0c_status, reasons


def _r_display(r_label: object) -> str:
    return "∞" if str(r_label) == "inf" else str(r_label)


def plot_rig_curve(rig_df: pd.DataFrame, rstar: pd.DataFrame, lang: str) -> None:
    """Будує RIG(r) криву."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for topology in TOPOLOGIES:
        sub = rig_df[rig_df["topology"] == topology].sort_values("r_numeric")
        ax.plot(sub["r_numeric"], sub["RIG_pct"], marker="o", linewidth=2, label=topology)
        star = rstar[rstar["topology"] == topology].iloc[0]["r_star"]
        if str(star) != "inf":
            star_x = R_NUMERIC[str(star)]
            star_y = float(sub.loc[sub["r_numeric"] == star_x, "RIG_pct"].iloc[0])
            ax.scatter([star_x], [star_y], s=90, edgecolor="black", zorder=5)
    ax.axhline(RIG_THRESHOLD, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_title(labels["rig_title"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["rig_y"])
    ax.set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "∞"])
    ax.legend(title=labels["topology"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M0_RIG_curve_{lang}.png", dpi=200)
    plt.close(fig)


def plot_rig_heatmap(rig_df: pd.DataFrame, lang: str) -> None:
    """Будує heatmap RIG(r)."""
    labels = LANG[lang]
    pivot = rig_df.pivot(index="topology", columns="r", values="RIG_pct").reindex(
        index=TOPOLOGIES, columns=R_ORDER
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=80.0,
        linewidths=0.5,
        cbar_kws={"label": labels["rig_y"]},
        ax=ax,
    )
    ax.set_title(labels["rig_heatmap"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["topology"])
    ax.set_xticklabels([_r_display(c.get_text()) for c in ax.get_xticklabels()])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M0_RIG_heatmap_{lang}.png", dpi=200)
    plt.close(fig)


def plot_rho_cvar(desc: pd.DataFrame, lang: str) -> None:
    """Будує абсолютну ρ_CVaR(r) з reference random_R1."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for topology in TOPOLOGIES:
        sub = desc[desc["topology"] == topology].sort_values("r_numeric")
        ax.plot(sub["r_numeric"], sub["rho_CVaR_0.10"], marker="o", linewidth=2, label=topology)
        ax.fill_between(
            sub["r_numeric"].to_numpy(dtype=float),
            sub["rho_CVaR_ci_low"].to_numpy(dtype=float),
            sub["rho_CVaR_ci_high"].to_numpy(dtype=float),
            alpha=0.10,
        )
        baseline = float(sub["random_R1_rho_CVaR"].iloc[0])
        ax.hlines(baseline, xmin=0, xmax=4, linestyles=":", linewidth=1, alpha=0.45)
    ax.set_title(labels["rho_title"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["rho_y"])
    ax.set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "∞"])
    ax.legend(title=labels["topology"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M0_rho_vs_r_{lang}.png", dpi=200)
    plt.close(fig)


def plot_rho_med(desc: pd.DataFrame, lang: str) -> None:
    """Будує абсолютну ρ_med(r)."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for topology in TOPOLOGIES:
        sub = desc[desc["topology"] == topology].sort_values("r_numeric")
        ax.plot(sub["r_numeric"], sub["rho_med"], marker="o", linewidth=2, label=topology)
        ax.fill_between(
            sub["r_numeric"].to_numpy(dtype=float),
            sub["rho_med_ci_low"].to_numpy(dtype=float),
            sub["rho_med_ci_high"].to_numpy(dtype=float),
            alpha=0.10,
        )
    ax.set_title(labels["rho_med_title"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["rho_med_y"])
    ax.set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "∞"])
    ax.legend(title=labels["topology"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M0_rho_med_vs_r_{lang}.png", dpi=200)
    plt.close(fig)


def make_figures(desc: pd.DataFrame, rig_df: pd.DataFrame, rstar: pd.DataFrame) -> None:
    """Створює 4 типи рисунків × 2 мови."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for lang in ("uk", "en"):
        plot_rig_curve(rig_df, rstar, lang)
        plot_rig_heatmap(rig_df, lang)
        plot_rho_cvar(desc, lang)
        plot_rho_med(desc, lang)


def make_report(
    desc: pd.DataFrame,
    rig_df: pd.DataFrame,
    rstar: pd.DataFrame,
    sanity: pd.DataFrame,
    spearman: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    hm0_status: str,
    hm0b_status: str,
    hm0c_status: str,
    reasons: list[str],
    raw: pd.DataFrame,
) -> str:
    """Формує текстовий M0 report після Patch v4.1."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("M0 RE-ANALYSIS: IS -> RIG (Patch v4.1)")
    lines.append("=" * 72)
    lines.append("Параметри: N=50, p_v=0.20, T=100, M=200, mu=1, alpha=0.10")
    lines.append("r-значення: {0, 1, 2, 3, inf}; K=1 (single-agent)")
    lines.append("Baseline RIG: rho_CVaR(random_R1), не r=0.")
    lines.append("")
    lines.append("--- SANITY CHECK: r=inf vs R1 Article 3 flow_greedy ---")
    lines.append(sanity.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("--- DESCRIPTIVE STATISTICS ---")
    lines.append(
        desc[
            [
                "topology",
                "r",
                "n_valid",
                "n_invalid",
                "random_R1_rho_CVaR",
                "rho_CVaR_0.10",
                "rho_CVaR_ci_low",
                "rho_CVaR_ci_high",
                "rho_med",
                "R_rb_med",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    lines.append("")
    lines.append("--- RIG(r) CURVE (baseline = random_R1) ---")
    pivot = rig_df.pivot(index="topology", columns="r", values="RIG_pct").reindex(
        index=TOPOLOGIES, columns=R_ORDER
    )
    lines.append(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    lines.append("")
    lines.append("--- r* DETERMINATION (search only r in {1,2,3}) ---")
    lines.append(rstar.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("--- HYPOTHESES AFTER PATCH v4.1 ---")
    lines.append(f"HM0 STATUS: {hm0_status}")
    lines.append(f"HM0b (visibility trap hierarchical): {hm0b_status}")
    lines.append(f"HM0c (r=0 > random hierarchical): {hm0c_status}")
    lines.append(f"Technical reasons: {', '.join(reasons) if reasons else 'none'}")
    lines.append("")
    lines.append("--- VISIBILITY TRAP CHECK ---")
    hier = rig_df[rig_df["topology"] == "hierarchical"].set_index("r")
    finite_hier = hier.loc[FINITE_R_FOR_RSTAR, "RIG_pct"]
    if bool((finite_hier < 0.0).all()):
        lines.append("hierarchical: RIG(r=1,2,3) < 0 -> partial visibility harmful.")
        lines.append("Interpretation: Information-Action Dissonance confirmed.")
    else:
        lines.append("hierarchical: RIG(r=1,2,3) is not negative.")
        lines.append("Interpretation: visibility trap is NOT confirmed on the existing M0 data.")
    lines.append("")
    lines.append("--- PARAMETER FOR M1 ---")
    for _, row in rstar.sort_values("topology").iterrows():
        lines.append(f"{row['topology']}: r* = {row['r_star']}")
    finite_modes = rstar[rstar["r_star"] != "inf"]["r_star"]
    if not finite_modes.empty:
        consensus = finite_modes.mode().iloc[0]
        lines.append(f"Finite-topology consensus r*: {consensus}")
    else:
        lines.append("Finite-topology consensus r*: none")
    lines.append("")
    lines.append("--- SPEARMAN r(r, rho) ---")
    lines.append(spearman.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("--- PAIRWISE WILCOXON (Bonferroni corrected) ---")
    lines.append(wilcoxon_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append("--- RUN SUMMARY ---")
    lines.append(
        f"runs_total={len(raw)}, runs_valid={int(raw['valid'].sum())}, "
        f"valid_pct={raw['valid'].mean() * 100.0:.2f}%"
    )
    return "\n".join(lines) + "\n"


def archive_old_report_once() -> None:
    """Зберігає старий IS report перед перегенерацією, якщо backup ще не існує."""
    old_report = AGG_DIR / "M0_report.txt"
    backup_report = AGG_DIR / "M0_report_v1_IS.txt"
    if old_report.exists() and not backup_report.exists():
        shutil.copy2(old_report, backup_report)


def main() -> None:
    """Виконує M0 re-analysis і зберігає CSV/TXT/PNG artifacts."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    archive_old_report_once()

    raw = load_raw()
    baselines = load_r1_baselines()
    desc = descriptive_stats(raw, baselines)
    rig_df = rig_curve(desc)
    rstar = rstar_table(rig_df)
    sanity = sanity_vs_r1(desc, baselines)
    spearman = spearman_analysis(raw)
    wilcoxon_df = paired_wilcoxon(raw)
    hm0_status, hm0b_status, hm0c_status, reasons = determine_status(sanity, rstar)

    desc.to_csv(AGG_DIR / "M0_descriptive_statistics.csv", index=False)
    rig_df.to_csv(AGG_DIR / "M0_RIG_curve.csv", index=False)
    rstar.to_csv(AGG_DIR / "M0_rstar.csv", index=False)
    sanity.to_csv(AGG_DIR / "M0_sanity_vs_R1.csv", index=False)
    spearman.to_csv(AGG_DIR / "M0_spearman.csv", index=False)
    wilcoxon_df.to_csv(AGG_DIR / "M0_pairwise_wilcoxon.csv", index=False)
    make_figures(desc, rig_df, rstar)

    report = make_report(
        desc=desc,
        rig_df=rig_df,
        rstar=rstar,
        sanity=sanity,
        spearman=spearman,
        wilcoxon_df=wilcoxon_df,
        hm0_status=hm0_status,
        hm0b_status=hm0b_status,
        hm0c_status=hm0c_status,
        reasons=reasons,
        raw=raw,
    )
    report_path = AGG_DIR / "M0_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

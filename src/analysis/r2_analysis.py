"""
Аналіз серії R2 для статті 3.

Реалізує: розділ 4 Протоколу R2
(див. docs/Протокол_R2.docx).

Приклад використання:
    python -m src.analysis.r2_analysis
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "R2"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

STRATEGIES = ["random", "centrality_degree", "centrality_betweenness", "flow_greedy"]
K_VALUES = [0, 1, 2, 3]
ALPHA = 0.10
BOOTSTRAP_B = 1000
PAIRWISE_ALPHA = 0.05 / 48
S2_REFERENCE_RHO_MED_R = 0.63
S2_REFERENCE_RHO_MED_P = 0.37


def _clean(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap CI."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_ci(
    values: pd.Series | list[float] | np.ndarray,
    statistic,
    B: int = BOOTSTRAP_B,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI для довільної статистики."""
    arr = _clean(values)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = statistic(sample)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для кожної K_res × strategy комбінації."""
    rows: list[dict] = []
    for (k_res, strategy), group_all in raw.groupby(["K_res", "recovery_strategy"]):
        group = group_all[group_all["valid"].astype(bool)].copy()
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)

        rho_med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("R2", k_res, strategy, "rho_med"),
        )
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("R2", k_res, strategy, "rho_cvar"),
        )
        rho_iqr_ci = bootstrap_ci(
            rho,
            compute_rho_IQR,
            seed=stable_seed("R2", k_res, strategy, "rho_iqr"),
        )
        r_rb_ci = bootstrap_ci(
            r_rb,
            np.median,
            seed=stable_seed("R2", k_res, strategy, "r_rb"),
        )

        rows.append(
            {
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
                "R_rb_ci_low": r_rb_ci[0],
                "R_rb_ci_high": r_rb_ci[1],
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0)
                if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["K_res", "recovery_strategy"]).reset_index(drop=True)


def spearman_analysis(desc: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(K_res, metric) для кожної recovery strategy."""
    rows: list[dict] = []
    for strategy in STRATEGIES:
        group = desc[desc["recovery_strategy"] == strategy].sort_values("K_res")
        x = group["K_res"].astype(float).to_numpy()
        for metric, col in [
            ("rho_med", "rho_med"),
            ("rho_CVaR_0.10", "rho_CVaR_0.10"),
        ]:
            y = group[col].astype(float).to_numpy()
            if len(x) < 2 or np.allclose(y, y[0], equal_nan=True):
                r_value, p_value = float("nan"), float("nan")
            else:
                r_value, p_value = spearmanr(x, y)
            rows.append(
                {
                    "recovery_strategy": strategy,
                    "metric": metric,
                    "spearman_r": float(r_value),
                    "p_value": float(p_value),
                    "n_points": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def _paired_values(
    raw: pd.DataFrame,
    k_res: int,
    left_strategy: str,
    right_strategy: str,
    column: str,
) -> pd.DataFrame:
    subset = raw[(raw["valid"].astype(bool)) & (raw["K_res"].astype(int) == int(k_res))]
    left = subset[subset["recovery_strategy"] == left_strategy][["paired_id", column]]
    right = subset[subset["recovery_strategy"] == right_strategy][["paired_id", column]]
    return left.merge(right, on="paired_id", suffixes=("_left", "_right"))


def paired_wilcoxon_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """
    48 paired Wilcoxon tests:
    4 K_res × 6 strategy pairs × 2 metrics (rho_med, rho_CVaR_0.10).
    """
    rows: list[dict] = []
    for k_res in K_VALUES:
        for left, right in combinations(STRATEGIES, 2):
            pairs = _paired_values(raw, k_res, left, right, "rho")
            for metric, mode in [("rho_med", "median"), ("rho_CVaR_0.10", "tail_cvar")]:
                if len(pairs) < 2:
                    rows.append(_empty_test_row(k_res, left, right, metric, len(pairs)))
                    continue

                x = pairs["rho_left"].astype(float)
                y = pairs["rho_right"].astype(float)
                if mode == "tail_cvar":
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
                        "K_res": int(k_res),
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


def _empty_test_row(k_res: int, left: str, right: str, metric: str, n_pairs: int) -> dict:
    return {
        "K_res": int(k_res),
        "left_strategy": left,
        "right_strategy": right,
        "metric": metric,
        "n_pairs": n_pairs,
        "median_delta": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }


def r_rb_sanity_check(raw: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє ідентичність R_rb по стратегіях для кожного (K_res, paired_id)."""
    valid = raw[raw["valid"].astype(bool)].copy()
    rows: list[dict] = []
    for (k_res, paired_id), group in valid.groupby(["K_res", "paired_id"]):
        values = group["R_rb"].astype(float).to_numpy()
        spread = float(np.nanmax(values) - np.nanmin(values)) if len(values) else float("nan")
        rows.append(
            {
                "K_res": int(k_res),
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
            "R_rb varies across strategies for at least one (K_res, paired_id):\n"
            + bad.to_string(index=False)
        )
    return sanity


def hr3_status(spearman: pd.DataFrame) -> tuple[str, str]:
    """Формальна перевірка HR3 за Patch v3.1."""
    flow = spearman[
        (spearman["recovery_strategy"] == "flow_greedy")
        & (spearman["metric"] == "rho_CVaR_0.10")
    ].iloc[0]
    r_value = float(flow["spearman_r"])
    p_value = float(flow["p_value"])
    if r_value >= 0.8 and p_value < 0.05:
        return "CONFIRMED", (
            "flow_greedy має Spearman r(K_res, rho_CVaR_0.10) ≥ 0.8 "
            "і p < 0.05."
        )
    if 0.6 <= r_value < 0.8:
        return "PARTIALLY_CONFIRMED", (
            "flow_greedy має помірну монотонну тенденцію, але не досягає "
            "порогу повного підтвердження."
        )
    return "REJECTED", (
        "flow_greedy не демонструє достатньо сильної залежності rho_CVaR_0.10 "
        "від K_res за порогами Patch v3.1."
    )


def cross_comparison_table(spearman: pd.DataFrame) -> pd.DataFrame:
    """Таблиця прямого порівняння з S2 статті 1."""
    rows = [
        {
            "source": "S2 random (Article 1)",
            "strategy": "random",
            "rho_med_spearman_r": S2_REFERENCE_RHO_MED_R,
            "rho_med_p_value": S2_REFERENCE_RHO_MED_P,
            "rho_CVaR_spearman_r": float("nan"),
            "rho_CVaR_p_value": float("nan"),
            "note": "reference from Article 1 S2",
        }
    ]
    for strategy in STRATEGIES:
        med = spearman[
            (spearman["recovery_strategy"] == strategy)
            & (spearman["metric"] == "rho_med")
        ].iloc[0]
        cvar = spearman[
            (spearman["recovery_strategy"] == strategy)
            & (spearman["metric"] == "rho_CVaR_0.10")
        ].iloc[0]
        rows.append(
            {
                "source": "R2",
                "strategy": strategy,
                "rho_med_spearman_r": float(med["spearman_r"]),
                "rho_med_p_value": float(med["p_value"]),
                "rho_CVaR_spearman_r": float(cvar["spearman_r"]),
                "rho_CVaR_p_value": float(cvar["p_value"]),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def localized_name(value: str, lang: str) -> str:
    try:
        return label(f"common.{value}", lang)
    except KeyError:
        return value


def plot_metric_vs_kres(desc: pd.DataFrame, lang: str, metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_col = "rho_med" if metric == "rho_med" else "rho_CVaR_0.10"
    low_col = "rho_med_ci_low" if metric == "rho_med" else "rho_CVaR_ci_low"
    high_col = "rho_med_ci_high" if metric == "rho_med" else "rho_CVaR_ci_high"
    title_key = "R2.rho_vs_Kres.title" if metric == "rho_med" else "R2.cvar_vs_Kres.title"
    y_label_key = "R2.rho_vs_Kres.y_label" if metric == "rho_med" else "R2.cvar_vs_Kres.y_label"

    for strategy in STRATEGIES:
        group = desc[desc["recovery_strategy"] == strategy].sort_values("K_res")
        x = group["K_res"].astype(float)
        y = group[y_col].astype(float)
        low = group[low_col].astype(float)
        high = group[high_col].astype(float)
        ax.plot(x, y, marker="o", linewidth=2, label=localized_name(strategy, lang))
        ax.fill_between(x, low, high, alpha=0.16)

    ax.set_title(label(title_key, lang))
    ax.set_xlabel(label("R2.rho_vs_Kres.x_label", lang))
    ax.set_ylabel(label(y_label_key, lang))
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.25)
    ax.legend(title=label("common.strategy", lang), loc="best")
    return fig


def plot_rho_vs_kres(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_kres(desc, lang, metric="rho_med")


def plot_cvar_vs_kres(desc: pd.DataFrame, lang: str) -> plt.Figure:
    return plot_metric_vs_kres(desc, lang, metric="rho_CVaR_0.10")


def plot_spearman_comparison(spearman: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = spearman.copy()
    data["strategy_label"] = data["recovery_strategy"].map(lambda v: localized_name(v, lang))
    data["metric_label"] = data["metric"].map(
        {
            "rho_med": label("common.rho_med", lang),
            "rho_CVaR_0.10": label("common.rho_cvar", lang),
        }
    )
    sns.barplot(data=data, x="strategy_label", y="spearman_r", hue="metric_label", ax=ax)
    ax.axhline(
        S2_REFERENCE_RHO_MED_R,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=label("R2.spearman_comparison.article1_reference", lang),
    )
    ax.set_title(label("R2.spearman_comparison.title", lang))
    ax.set_xlabel(label("common.strategy", lang))
    ax.set_ylabel(label("R2.spearman_comparison.y_label", lang))
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_delta_rho_per_kres(desc: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    random = desc[desc["recovery_strategy"] == "random"].set_index("K_res")
    flow = desc[desc["recovery_strategy"] == "flow_greedy"].set_index("K_res")
    delta = (flow["rho_CVaR_0.10"] - random["rho_CVaR_0.10"]).reindex(K_VALUES)
    heat = pd.DataFrame([delta.to_numpy()], index=["flow_greedy − random"], columns=K_VALUES)
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Δρ_CVaR"})
    ax.set_title(label("R2.delta_rho_per_Kres.title", lang))
    ax.set_xlabel(label("R2.rho_vs_Kres.x_label", lang))
    ax.set_ylabel("")
    return fig


def make_figures(desc: pd.DataFrame, spearman: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    outputs += save_dual_lang(plot_rho_vs_kres, str(FIG_DIR / "R2_rho_vs_Kres_{lang}.png"), desc=desc)
    outputs += save_dual_lang(plot_cvar_vs_kres, str(FIG_DIR / "R2_cvar_vs_Kres_{lang}.png"), desc=desc)
    outputs += save_dual_lang(
        plot_spearman_comparison,
        str(FIG_DIR / "R2_spearman_comparison_{lang}.png"),
        spearman=spearman,
    )
    outputs += save_dual_lang(
        plot_delta_rho_per_kres,
        str(FIG_DIR / "R2_delta_rho_per_Kres_{lang}.png"),
        desc=desc,
    )
    return outputs


def make_report(
    raw: pd.DataFrame,
    desc: pd.DataFrame,
    spearman: pd.DataFrame,
    cross: pd.DataFrame,
    tests: pd.DataFrame,
    sanity: pd.DataFrame,
    figures: list[Path],
) -> str:
    status, status_reason = hr3_status(spearman)
    desc_cols = [
        "K_res",
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
        "rho_IQR_ci_low",
        "rho_IQR_ci_high",
        "R_rb_med",
        "non_recovered_pct",
    ]
    spearman_cols = ["recovery_strategy", "metric", "spearman_r", "p_value", "n_points"]
    test_cols = [
        "K_res",
        "left_strategy",
        "right_strategy",
        "metric",
        "n_pairs",
        "median_delta",
        "p_value",
        "p_adj_bonferroni",
        "significant",
    ]
    sanity_passed = bool(sanity["passed"].all()) if not sanity.empty else False
    paired_counts = raw.groupby("paired_id").size()
    valid_paired_counts = raw[raw["valid"].astype(bool)].groupby("paired_id").size()
    invalid = raw[~raw["valid"].astype(bool)]
    invalid_paired_ids = sorted(invalid["paired_id"].dropna().unique().tolist())
    max_excl = float(desc["excl_pct"].max()) if not desc.empty else float("nan")
    configs_below_95 = int((desc["n_valid"] < 95).sum()) if "n_valid" in desc else 0
    lines = [
        "============================================================",
        "R2: STRATEGY x K_res INTERACTION (HR3 verification)",
        "============================================================",
        "",
        "Parameters: topology=reserved, K_res ∈ {0,1,2,3}, 4 strategies,",
        "            N=50, H=4, b=3, p_v=0.20, T=100, M=100, paired seed design",
        "",
        "--- DESCRIPTIVE STATISTICS ---",
        desc[desc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- SPEARMAN CORRELATIONS (CENTRAL ANALYSIS) ---",
        spearman[spearman_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Comparison with S2 of Article 1:",
        cross.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- HYPOTHESIS HR3 STATUS ---",
        status,
        status_reason,
        "",
        "--- PAIRED WILCOXON TESTS (per K_res, Bonferroni alpha=0.001) ---",
        tests[test_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        "",
        "--- R_RB SANITY CHECK ---",
        f"R_rb identical across strategies for each (K_res, paired_id): {'YES' if sanity_passed else 'NO'}",
        f"Checked paired groups: {len(sanity)}",
        "",
        "--- ACCEPTANCE NOTES ---",
        f"Raw paired design: {len(paired_counts)} paired_id values; expected rows per paired_id = 16.",
        f"Raw paired rows complete: {'YES' if (paired_counts == 16).all() else 'NO'}",
        f"Valid paired groups after exclusions: {len(valid_paired_counts)} paired_id values.",
        f"Max exclusion rate across configurations: {max_excl:.2f}%.",
        f"Configurations below n_valid >= 95 criterion: {configs_below_95} of {len(desc)}.",
        (
            "Invalid paired_ids: "
            + (", ".join(invalid_paired_ids) if invalid_paired_ids else "none")
        ),
        (
            "Invalid reasons: "
            + (
                invalid["exclusion_reason"].value_counts(dropna=False).to_string()
                if not invalid.empty else "none"
            )
        ),
        "",
        "--- INTERPRETATION ---",
        "R2 tests whether intelligent recovery turns K_res from a structurally useful but functionally muted reserve into a tail-resilience gain. The central criterion is Spearman r(K_res, rho_CVaR_0.10) for flow_greedy.",
        "",
        "--- GENERATED FIGURES ---",
        "\n".join(str(path.relative_to(ROOT)) for path in figures),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    raw_path = RAW_DIR / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"R2 raw results not found: {raw_path}")
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    desc = descriptive_stats(raw)
    spearman = spearman_analysis(desc)
    cross = cross_comparison_table(spearman)
    tests = paired_wilcoxon_tests(raw)
    sanity = r_rb_sanity_check(raw)
    figures = make_figures(desc, spearman)

    desc.to_csv(AGG_DIR / "R2_descriptive_statistics.csv", index=False)
    spearman.to_csv(AGG_DIR / "R2_spearman_correlations.csv", index=False)
    cross.to_csv(AGG_DIR / "R2_cross_comparison_S2.csv", index=False)
    tests.to_csv(AGG_DIR / "R2_pairwise_wilcoxon.csv", index=False)
    sanity.to_csv(AGG_DIR / "R2_R_rb_sanity_check.csv", index=False)

    report = make_report(raw, desc, spearman, cross, tests, sanity, figures)
    (AGG_DIR / "R2_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

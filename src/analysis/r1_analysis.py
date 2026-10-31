"""
Аналіз серії R1 для статті 3.

Реалізує: розділ 6 протоколу R1
(див. docs/Протокол_R1.docx).

Приклад використання:
    python -m src.analysis.r1_analysis
"""

from __future__ import annotations

from itertools import combinations
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.visualization.i18n.labels import label
from src.visualization.i18n.plot_helpers import save_dual_lang


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "R1"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"
FIG_DIR = ROOT / "results" / "figures" / "article3"

STRATEGIES = ["random", "centrality_degree", "centrality_betweenness", "flow_greedy"]
TOPOLOGIES = ["hierarchical", "hub", "reserved", "dense"]
ALPHA = 0.10
BONFERRONI_ALPHA = 0.05 / 24


def _clean(values: pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def bootstrap_ci(
    values: pd.Series | list[float],
    statistic,
    B: int = 1000,
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


def cliffs_delta(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Cliff's δ: positive якщо y загалом більший за x."""
    x_arr = _clean(x)
    y_arr = _clean(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return float("nan")
    diff = y_arr[:, None] - x_arr[None, :]
    return float(np.sum(np.sign(diff)) / (x_arr.size * y_arr.size))


def effect_size_label(delta: float) -> str:
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.330:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для кожної topology × strategy."""
    rows: list[dict] = []
    valid = raw[raw["valid"].astype(bool)].copy()
    for (topology, strategy), group in valid.groupby(["topology", "recovery_strategy"]):
        rho = group["rho"].astype(float)
        r_rb = group["R_rb"].astype(float)
        rho_med_ci = bootstrap_ci(rho, np.median, seed=stable_seed(topology, strategy, "rho_med"))
        rho_cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed(topology, strategy, "rho_cvar"),
        )
        r_rb_ci = bootstrap_ci(r_rb, np.median, seed=stable_seed(topology, strategy, "r_rb"))
        rows.append(
            {
                "topology": topology,
                "recovery_strategy": strategy,
                "n_valid": int(len(group)),
                "rho_med": compute_rho_med(rho),
                "rho_med_ci_low": rho_med_ci[0],
                "rho_med_ci_high": rho_med_ci[1],
                "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=ALPHA),
                "rho_CVaR_ci_low": rho_cvar_ci[0],
                "rho_CVaR_ci_high": rho_cvar_ci[1],
                "rho_IQR": compute_rho_IQR(rho),
                "R_rb_med": float(np.median(r_rb)),
                "R_rb_ci_low": r_rb_ci[0],
                "R_rb_ci_high": r_rb_ci[1],
                "non_recovered_pct": float(group["non_recovered"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["topology", "recovery_strategy"]).reset_index(drop=True)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _paired_values(
    raw: pd.DataFrame,
    topology: str,
    left_strategy: str,
    right_strategy: str,
    column: str,
) -> pd.DataFrame:
    subset = raw[(raw["valid"].astype(bool)) & (raw["topology"] == topology)]
    left = subset[subset["recovery_strategy"] == left_strategy][["paired_id", column]]
    right = subset[subset["recovery_strategy"] == right_strategy][["paired_id", column]]
    return left.merge(right, on="paired_id", suffixes=("_left", "_right"))


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise paired Wilcoxon tests між стратегіями для кожної topology."""
    rows: list[dict] = []
    strategy_pairs = [
        ("random", "centrality_degree"),
        ("random", "centrality_betweenness"),
        ("random", "flow_greedy"),
        ("centrality_degree", "centrality_betweenness"),
        ("centrality_degree", "flow_greedy"),
        ("centrality_betweenness", "flow_greedy"),
    ]
    for topology in TOPOLOGIES:
        for left, right in strategy_pairs:
            rho_pairs = _paired_values(raw, topology, left, right, "rho")
            rrb_pairs = _paired_values(raw, topology, left, right, "R_rb")

            metric_specs = [
                ("rho_med", rho_pairs, "rho", "median"),
                ("rho_CVaR_0.10", rho_pairs, "rho", "tail_cvar"),
                ("R_rb", rrb_pairs, "R_rb", "median"),
            ]
            for metric, pairs, column, mode in metric_specs:
                if len(pairs) < 2:
                    rows.append(_empty_test_row(topology, left, right, metric, len(pairs)))
                    continue
                x = pairs[f"{column}_left"].astype(float)
                y = pairs[f"{column}_right"].astype(float)
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
                delta = cliffs_delta(x_test, y_test)
                rows.append(
                    {
                        "topology": topology,
                        "left_strategy": left,
                        "right_strategy": right,
                        "metric": metric,
                        "n_pairs": int(len(x_test)),
                        "median_delta": float(delta_value),
                        "p_value": float(p_value),
                        "cliffs_delta": delta,
                        "effect_size": effect_size_label(delta),
                    }
                )

    tests = pd.DataFrame(rows)
    tests["p_adj_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    tests["significant"] = tests["p_adj_bonferroni"] < 0.05
    tests["protocol_alpha_pairwise"] = BONFERRONI_ALPHA
    return tests


def _empty_test_row(topology: str, left: str, right: str, metric: str, n_pairs: int) -> dict:
    return {
        "topology": topology,
        "left_strategy": left,
        "right_strategy": right,
        "metric": metric,
        "n_pairs": n_pairs,
        "median_delta": float("nan"),
        "p_value": float("nan"),
        "cliffs_delta": float("nan"),
        "effect_size": "NA",
    }


def hypothesis_summary(desc: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Формальна перевірка HR1/HR2/HR4 на основі R1."""
    rows: list[dict] = []
    fg = desc[desc["recovery_strategy"] == "flow_greedy"]
    rnd = desc[desc["recovery_strategy"] == "random"]
    merged = rnd.merge(fg, on="topology", suffixes=("_random", "_flow"))
    fg_tests = tests[
        (tests["left_strategy"] == "random")
        & (tests["right_strategy"] == "flow_greedy")
    ]

    hr1_med_hits = 0
    hr1_cvar_hits = 0
    hr4_hits = 0
    for _, row in merged.iterrows():
        topology = row["topology"]
        d_med = row["rho_med_flow"] - row["rho_med_random"]
        d_cvar = row["rho_CVaR_0.10_flow"] - row["rho_CVaR_0.10_random"]
        ratio = d_cvar / d_med if abs(d_med) > 1e-12 else np.nan
        rho_test = fg_tests[(fg_tests["topology"] == topology) & (fg_tests["metric"] == "rho_med")]
        cvar_test = fg_tests[(fg_tests["topology"] == topology) & (fg_tests["metric"] == "rho_CVaR_0.10")]
        rho_sig = bool(rho_test["significant"].iloc[0]) if not rho_test.empty else False
        cvar_sig = bool(cvar_test["significant"].iloc[0]) if not cvar_test.empty else False
        rho_delta = float(rho_test["cliffs_delta"].iloc[0]) if not rho_test.empty else np.nan
        cvar_delta = float(cvar_test["cliffs_delta"].iloc[0]) if not cvar_test.empty else np.nan
        med_hit = d_med >= 0.03 and rho_sig and rho_delta >= 0.30
        cvar_hit = d_cvar >= 0.05 and cvar_sig and cvar_delta >= 0.30
        hr1_med_hits += int(med_hit)
        hr1_cvar_hits += int(cvar_hit)
        hr4_hits += int(pd.notna(ratio) and ratio > 1.5)
        rows.append(
            {
                "hypothesis": "HR1_per_topology",
                "topology": topology,
                "delta_rho_med_flow_vs_random": d_med,
                "delta_rho_CVaR_flow_vs_random": d_cvar,
                "ratio_CVaR_to_med": ratio,
                "rho_med_hit": med_hit,
                "rho_CVaR_hit": cvar_hit,
                "HR4_ratio_hit": pd.notna(ratio) and ratio > 1.5,
            }
        )

    order_hits = 0
    for topology, group in desc.groupby("topology"):
        values = group.set_index("recovery_strategy")["rho_med"]
        order_ok = (
            values["flow_greedy"] >= values["centrality_betweenness"]
            >= values["centrality_degree"]
            >= values["random"]
        )
        order_hits += int(order_ok)
        rows.append({"hypothesis": "HR2_order", "topology": topology, "order_ok": order_ok})

    centrality_flow = tests[
        (tests["metric"] == "rho_med")
        & (
            (
                (tests["left_strategy"] == "centrality_degree")
                & (tests["right_strategy"] == "flow_greedy")
            )
            | (
                (tests["left_strategy"] == "centrality_betweenness")
                & (tests["right_strategy"] == "flow_greedy")
            )
        )
    ]
    centrality_flow_sig = int(centrality_flow["significant"].sum())

    rows.extend(
        [
            {
                "hypothesis": "HR1_status",
                "topology": "ALL",
                "status": (
                    "CONFIRMED"
                    if hr1_med_hits >= 3 and hr1_cvar_hits >= 3
                    else "PARTIALLY_CONFIRMED"
                    if hr1_med_hits >= 1 or hr1_cvar_hits >= 1
                    else "NOT_CONFIRMED"
                ),
                "rho_med_hits": hr1_med_hits,
                "rho_CVaR_hits": hr1_cvar_hits,
            },
            {
                "hypothesis": "HR2_preliminary_status",
                "topology": "ALL",
                "status": (
                    "CONFIRMED"
                    if order_hits >= 3 and centrality_flow_sig >= 4
                    else "NEEDS_R2"
                ),
                "order_hits": order_hits,
                "centrality_vs_flow_significant": centrality_flow_sig,
            },
            {
                "hypothesis": "HR4_preliminary_status",
                "topology": "ALL",
                "status": "CONFIRMED" if hr4_hits >= 3 else "NOT_CONFIRMED",
                "ratio_hits": hr4_hits,
            },
        ]
    )
    return pd.DataFrame(rows)


def localized_name(value: str, lang: str) -> str:
    try:
        return label(f"common.{value}", lang)
    except KeyError:
        return value


def plot_strategy_comparison(raw: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    data = raw[raw["valid"].astype(bool)].copy()
    data["strategy_label"] = data["recovery_strategy"].map(lambda v: localized_name(v, lang))
    data["topology_label"] = data["topology"].map(lambda v: localized_name(v, lang))
    sns.boxplot(data=data, x="topology_label", y="rho", hue="strategy_label", ax=ax)
    ax.set_title(label("R1.strategy_comparison.title", lang))
    ax.set_xlabel(label("common.topology", lang))
    ax.set_ylabel(label("R1.strategy_comparison.y_label", lang))
    ax.legend(title=label("common.strategy", lang), loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_cvar_dominance(desc: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    data = desc.copy()
    data["strategy_label"] = data["recovery_strategy"].map(lambda v: localized_name(v, lang))
    data["topology_label"] = data["topology"].map(lambda v: localized_name(v, lang))
    sns.barplot(data=data, x="topology_label", y="rho_CVaR_0.10", hue="strategy_label", ax=ax)
    ax.set_title(label("R1.cvar_dominance.title", lang))
    ax.set_xlabel(label("common.topology", lang))
    ax.set_ylabel(label("common.rho_cvar", lang))
    ax.legend(title=label("common.strategy", lang), loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_delta_heatmap(tests: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    data = tests[tests["metric"] == "rho_med"].copy()
    data["pair"] = data["left_strategy"] + " → " + data["right_strategy"]
    pivot = data.pivot_table(index="pair", columns="topology", values="median_delta")
    pivot = pivot.reindex(columns=TOPOLOGIES)
    pivot.columns = [localized_name(c, lang) for c in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="vlag", center=0.0, ax=ax)
    ax.set_title(label("R1.delta_heatmap.title", lang))
    ax.set_xlabel(label("common.topology", lang))
    ax.set_ylabel("")
    return fig


def plot_ratio_cvar_med(summary: pd.DataFrame, lang: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = summary[summary["hypothesis"] == "HR1_per_topology"].copy()
    data["topology_label"] = data["topology"].map(lambda v: localized_name(v, lang))
    sns.barplot(data=data, x="topology_label", y="ratio_CVaR_to_med", ax=ax)
    ax.axhline(1.5, color="red", linestyle="--", linewidth=1)
    ax.set_title(label("R1.ratio_cvar_med.title", lang))
    ax.set_xlabel(label("common.topology", lang))
    ax.set_ylabel(label("common.ratio", lang))
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_figures(raw: pd.DataFrame, desc: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    outputs += save_dual_lang(plot_strategy_comparison, str(FIG_DIR / "R1_strategy_comparison_{lang}.png"), raw=raw)
    outputs += save_dual_lang(plot_cvar_dominance, str(FIG_DIR / "R1_cvar_dominance_{lang}.png"), desc=desc)
    outputs += save_dual_lang(plot_delta_heatmap, str(FIG_DIR / "R1_delta_rho_heatmap_{lang}.png"), tests=tests)
    outputs += save_dual_lang(plot_ratio_cvar_med, str(FIG_DIR / "R1_ratio_cvar_med_{lang}.png"), summary=summary)
    return outputs


def make_report(desc: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame, figures: list[Path]) -> str:
    hr1 = summary[summary["hypothesis"] == "HR1_status"].iloc[0]
    hr2 = summary[summary["hypothesis"] == "HR2_preliminary_status"].iloc[0]
    hr4 = summary[summary["hypothesis"] == "HR4_preliminary_status"].iloc[0]
    desc_cols = [
        "topology",
        "recovery_strategy",
        "n_valid",
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
    test_cols = [
        "topology",
        "left_strategy",
        "right_strategy",
        "metric",
        "n_pairs",
        "median_delta",
        "p_value",
        "p_adj_bonferroni",
        "significant",
        "cliffs_delta",
        "effect_size",
    ]
    lines = [
        "============================================================",
        "R1: BASELINE STRATEGY COMPARISON",
        "============================================================",
        "",
        "Parameters: N=50, H=4, b=3, p_v=0.20, T=100, M=100, alpha=0.10",
        "Strategies: random, centrality_degree, centrality_betweenness, flow_greedy",
        "Topologies: hierarchical, hub, reserved, dense",
        "Paired seed design: yes",
        "",
        "--- DESCRIPTIVE STATISTICS ---",
        desc[desc_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- PAIRED WILCOXON TESTS (Bonferroni alpha=0.0021 protocol; p_adj across 72 tests) ---",
        tests[test_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        "",
        "--- HYPOTHESIS HR1 ---",
        summary[summary["hypothesis"] == "HR1_per_topology"].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        f"HR1 status: {hr1['status']} (rho_med hits={hr1.get('rho_med_hits')}, rho_CVaR hits={hr1.get('rho_CVaR_hits')})",
        "",
        "--- HYPOTHESIS HR2 (preliminary) ---",
        summary[summary["hypothesis"] == "HR2_order"].to_string(index=False),
        f"HR2 preliminary status: {hr2['status']} (order hits={hr2.get('order_hits')}, centrality-vs-flow significant={hr2.get('centrality_vs_flow_significant')})",
        "",
        "--- HYPOTHESIS HR4 (preliminary) ---",
        summary[summary["hypothesis"] == "HR1_per_topology"][["topology", "ratio_CVaR_to_med", "HR4_ratio_hit"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        f"HR4 preliminary status: {hr4['status']} (ratio hits={hr4.get('ratio_hits')})",
        "",
        "--- INTERPRETATION ---",
        "R1 tests whether intelligent recovery improves resilience under the same network and disruption seeds. Positive median_delta means the right-hand strategy outperforms the left-hand strategy in the listed pair.",
        "",
        "--- ANOMALIES ---",
        "If HR1 thresholds are not reached, this indicates that R1 neutral conditions are too easy or that node_failure p_v=0.20 recovers quickly enough for strategy choice to have limited integral impact.",
        "",
        "--- GENERATED FIGURES ---",
        "\n".join(str(path.relative_to(ROOT)) for path in figures),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    raw_path = RAW_DIR / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"R1 raw results not found: {raw_path}")
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    desc = descriptive_stats(raw)
    tests = paired_tests(raw)
    summary = hypothesis_summary(desc, tests)
    figures = make_figures(raw, desc, tests, summary)

    desc.to_csv(AGG_DIR / "R1_descriptive_statistics.csv", index=False)
    tests.to_csv(AGG_DIR / "R1_pairwise_tests.csv", index=False)
    summary.to_csv(AGG_DIR / "R1_hypothesis_summary.csv", index=False)
    desc.to_csv(AGG_DIR / "R1_aggregated_results.csv", index=False)
    report = make_report(desc, tests, summary, figures)
    (AGG_DIR / "R1_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

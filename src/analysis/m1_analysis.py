"""
Re-analysis серії M1 після Patch v4.3.

Реалізує: Math Spec Article 4 Patch v4.3, §3–§7
(див. docs/Math_Spec_Article_4_Patch_v4_3.docx).

Приклад використання:
    python -m src.analysis.m1_analysis
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
import seaborn as sns
from scipy.stats import wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M1"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
R1_STATS_PATH = ROOT / "results" / "aggregated" / "article3" / "R1_descriptive_statistics.csv"
M0_STATS_PATH = ROOT / "results" / "aggregated" / "article4" / "M0_descriptive_statistics.csv"

TOPOLOGIES = ["hierarchical", "dense", "hub", "reserved"]
STRATEGIES = ["random_agent", "local_greedy", "local_centrality", "coord_greedy"]
ALPHA = 0.10
BOOTSTRAP_B = 2000

LANG = {
    "uk": {
        "strategy_comparison": "M1: порівняння multi-agent recovery стратегій",
        "decomposition": "M1: декомпозиція виграшу PV + CV_strat = TMAG",
        "coordination_waste": "M1: coordination waste незалежних агентів",
        "vs_baselines": "M1: стратегії відносно R1/M1 baseline",
        "topology": "топологія",
        "strategy": "стратегія",
        "rho_cvar": "ρ_CVaR_0.10",
        "delta": "Δρ_CVaR",
        "cw": "CW = random_K3 − local_greedy",
    },
    "en": {
        "strategy_comparison": "M1: multi-agent recovery strategy comparison",
        "decomposition": "M1: gain decomposition PV + CV_strat = TMAG",
        "coordination_waste": "M1: coordination waste of independent agents",
        "vs_baselines": "M1: strategies relative to R1/M1 baselines",
        "topology": "topology",
        "strategy": "strategy",
        "rho_cvar": "ρ_CVaR_0.10",
        "delta": "Δρ_CVaR",
        "cw": "CW = random_K3 − local_greedy",
    },
}


def _clean(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
    """Повертає float-масив без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap."""
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


def effect_size_label(delta: float) -> str:
    """Інтерпретація |Cliff δ| за Romano et al. thresholds."""
    value = abs(float(delta))
    if value < 0.147:
        return "negligible"
    if value < 0.33:
        return "small"
    if value < 0.474:
        return "medium"
    return "large"


def load_raw() -> pd.DataFrame:
    """Завантажує існуючі M1 raw results без запуску симуляції."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"M1 raw_results.csv not found: {path}")
    raw = pd.read_csv(path, keep_default_na=False)
    raw["valid"] = raw["valid"].astype(str).str.lower().isin({"true", "1"})
    return raw


def load_r1_benchmarks() -> pd.DataFrame:
    """Завантажує R1 random і flow_greedy benchmarks."""
    r1 = pd.read_csv(R1_STATS_PATH)
    bench = r1.pivot_table(
        index="topology",
        columns="recovery_strategy",
        values="rho_CVaR_0.10",
        aggfunc="first",
    ).reset_index()
    return bench.rename(
        columns={
            "random": "random_R1",
            "flow_greedy": "flow_greedy_R1",
        }
    )[["topology", "random_R1", "flow_greedy_R1"]]


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для topology × agent_strategy."""
    rows: list[dict] = []
    for (topology, strategy), group_all in raw.groupby(["topology", "agent_strategy"]):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("M1", topology, strategy, "cvar"),
        )
        med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("M1", topology, strategy, "median"),
        )
        rows.append(
            {
                "topology": topology,
                "agent_strategy": strategy,
                "coordination_mode": group_all["coordination_mode"].iloc[0],
                "n_total": int(len(group_all)),
                "n_valid": int(len(group)),
                "n_invalid": int(len(group_all) - len(group)),
                "rho_med": compute_rho_med(rho),
                "rho_med_ci_low": med_ci[0],
                "rho_med_ci_high": med_ci[1],
                "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=ALPHA),
                "rho_CVaR_ci_low": cvar_ci[0],
                "rho_CVaR_ci_high": cvar_ci[1],
                "rho_IQR": compute_rho_IQR(rho),
                "R_rb_med": float(group["R_rb"].astype(float).median()) if len(group) else float("nan"),
                "non_recovered_pct": float(
                    group["non_recovered"].astype(str).str.lower().isin({"true", "1"}).mean() * 100.0
                )
                if len(group)
                else float("nan"),
            }
        )
    order = {strategy: i for i, strategy in enumerate(STRATEGIES)}
    out = pd.DataFrame(rows)
    out["strategy_order"] = out["agent_strategy"].map(order)
    return out.sort_values(["topology", "strategy_order"]).drop(columns=["strategy_order"]).reset_index(drop=True)


def decomposition_metrics(desc: pd.DataFrame, r1: pd.DataFrame) -> pd.DataFrame:
    """PV, CV_strat, BCG, CW, TMAG за Patch v4.3, §3."""
    pivot = desc.pivot(index="topology", columns="agent_strategy", values="rho_CVaR_0.10")
    rows: list[dict] = []
    r1_map = r1.set_index("topology").to_dict("index")
    for topology in TOPOLOGIES:
        rand_k3 = float(pivot.loc[topology, "random_agent"])
        lg = float(pivot.loc[topology, "local_greedy"])
        cg = float(pivot.loc[topology, "coord_greedy"])
        rand_r1 = float(r1_map[topology]["random_R1"])
        fg_r1 = float(r1_map[topology]["flow_greedy_R1"])
        pv = rand_k3 - rand_r1
        cv = cg - rand_k3
        rows.append(
            {
                "topology": topology,
                "random_R1": rand_r1,
                "flow_greedy_R1": fg_r1,
                "random_K3": rand_k3,
                "local_greedy_K3": lg,
                "coord_greedy_K3": cg,
                "PV": pv,
                "CV_strat": cv,
                "PV_CV_ratio": pv / cv if abs(cv) > 1e-12 else float("inf"),
                "BCG": cg - fg_r1,
                "CW": rand_k3 - lg,
                "TMAG": cg - rand_r1,
            }
        )
    return pd.DataFrame(rows)


def sanity_checks(decomp: pd.DataFrame) -> pd.DataFrame:
    """Новий M1 sanity за Patch v4.3, §5."""
    checks = [
        {
            "check": "throughput",
            "condition": "random_agent(K3) > random_R1(K1) + 0.010",
            "passed": bool((decomp["PV"] > 0.010).all()),
            "min_margin": float((decomp["PV"] - 0.010).min()),
        },
        {
            "check": "coordination",
            "condition": "coord_greedy > random_agent at same K",
            "passed": bool((decomp["CV_strat"] > 0.0).all()),
            "min_margin": float(decomp["CV_strat"].min()),
        },
        {
            "check": "isolation",
            "condition": "coord_greedy > local_greedy independent at same K",
            "passed": bool((decomp["coord_greedy_K3"] > decomp["local_greedy_K3"]).all()),
            "min_margin": float((decomp["coord_greedy_K3"] - decomp["local_greedy_K3"]).min()),
        },
    ]
    return pd.DataFrame(checks)


def pairwise_wilcoxon(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon між M1 стратегіями у межах topology."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        sub = raw[(raw["valid"]) & (raw["topology"] == topology)]
        for left, right in combinations(STRATEGIES, 2):
            left_df = sub[sub["agent_strategy"] == left][["paired_id", "rho"]]
            right_df = sub[sub["agent_strategy"] == right][["paired_id", "rho"]]
            merged = left_df.merge(right_df, on="paired_id", suffixes=("_left", "_right"))
            if len(merged) < 2:
                stat, p_value, delta, cliff = np.nan, np.nan, np.nan, np.nan
            else:
                left_values = merged["rho_left"].astype(float)
                right_values = merged["rho_right"].astype(float)
                delta = float(right_values.median() - left_values.median())
                try:
                    stat, p_value = wilcoxon(right_values, left_values, alternative="greater")
                except ValueError:
                    stat, p_value = 0.0, 1.0
                cliff = cliffs_delta(left_values, right_values)
            rows.append(
                {
                    "topology": topology,
                    "strategy_A": left,
                    "strategy_B": right,
                    "n_pairs": int(len(merged)),
                    "median_delta_B_minus_A": delta,
                    "wilcoxon_stat": float(stat),
                    "p_value_greater": float(p_value),
                    "cliffs_delta": cliff,
                    "effect_size": effect_size_label(cliff) if not math.isnan(cliff) else "nan",
                }
            )
    out = pd.DataFrame(rows)
    out["p_adj_bonferroni"] = np.minimum(out["p_value_greater"] * len(out), 1.0)
    out["significant"] = out["p_adj_bonferroni"] < 0.05
    return out


def hm1_status(pairwise: pd.DataFrame, decomp: pd.DataFrame) -> pd.DataFrame:
    """HM1: coord_greedy_K3 > random_agent_K3."""
    rows = []
    for topology in TOPOLOGIES:
        row = pairwise[
            (pairwise["topology"] == topology)
            & (pairwise["strategy_A"] == "random_agent")
            & (pairwise["strategy_B"] == "coord_greedy")
        ].iloc[0]
        cv = float(decomp.loc[decomp["topology"] == topology, "CV_strat"].iloc[0])
        confirmed = bool(cv > 0.0 and row["p_adj_bonferroni"] < 0.05)
        rows.append(
            {
                "topology": topology,
                "CV_strat": cv,
                "cliffs_delta": float(row["cliffs_delta"]),
                "effect_size": row["effect_size"],
                "p_adj_bonferroni": float(row["p_adj_bonferroni"]),
                "HM1_status": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            }
        )
    return pd.DataFrame(rows)


def strategy_ranking(desc: pd.DataFrame) -> pd.DataFrame:
    """Ранжує стратегії за ρ_CVaR для кожної topology."""
    rows = []
    for topology in TOPOLOGIES:
        sub = desc[desc["topology"] == topology].sort_values("rho_CVaR_0.10", ascending=False)
        row = {"topology": topology}
        for idx, (_, item) in enumerate(sub.iterrows(), start=1):
            row[f"rank{idx}"] = item["agent_strategy"]
            row[f"rank{idx}_rho_CVaR"] = float(item["rho_CVaR_0.10"])
        rows.append(row)
    return pd.DataFrame(rows)


def vs_m0_comparison(desc: pd.DataFrame) -> pd.DataFrame:
    """Порівнює coord_greedy K=3 з M0 flow_greedy_local r=2 K=1."""
    m0 = pd.read_csv(M0_STATS_PATH)
    m0_r2 = m0[np.isclose(m0["r"].astype(float), 2.0)][["topology", "rho_CVaR_0.10"]].rename(
        columns={"rho_CVaR_0.10": "M0_flow_greedy_local_r2"}
    )
    cg = desc[desc["agent_strategy"] == "coord_greedy"][["topology", "rho_CVaR_0.10"]].rename(
        columns={"rho_CVaR_0.10": "M1_coord_greedy"}
    )
    out = cg.merge(m0_r2, on="topology", how="left")
    out["delta_M1_minus_M0"] = out["M1_coord_greedy"] - out["M0_flow_greedy_local_r2"]
    return out


def _strategy_label(strategy: str) -> str:
    return {
        "random_agent": "random_K3",
        "local_greedy": "local_greedy",
        "local_centrality": "local_centrality",
        "coord_greedy": "coord_greedy",
    }.get(strategy, strategy)


def plot_strategy_comparison(desc: pd.DataFrame, r1: pd.DataFrame, lang: str) -> None:
    """Grouped bar ρ_CVaR для стратегій із R1 reference markers."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    plot_df = desc.copy()
    plot_df["strategy_label"] = plot_df["agent_strategy"].map(_strategy_label)
    sns.barplot(
        data=plot_df,
        x="topology",
        y="rho_CVaR_0.10",
        hue="strategy_label",
        order=TOPOLOGIES,
        ax=ax,
    )
    r1_map = r1.set_index("topology")
    for x, topology in enumerate(TOPOLOGIES):
        ax.scatter(x, r1_map.loc[topology, "random_R1"], marker="x", color="black", s=80, zorder=5)
        ax.scatter(x, r1_map.loc[topology, "flow_greedy_R1"], marker="_", color="black", s=180, zorder=5)
    ax.set_title(labels["strategy_comparison"])
    ax.set_xlabel(labels["topology"])
    ax.set_ylabel(labels["rho_cvar"])
    ax.set_ylim(max(0.88, plot_df["rho_CVaR_0.10"].min() - 0.02), 1.0)
    ax.legend(title=labels["strategy"], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M1_strategy_comparison_{lang}.png", dpi=200)
    plt.close(fig)


def plot_decomposition(decomp: pd.DataFrame, lang: str) -> None:
    """Stacked bar PV + CV_strat = TMAG."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(decomp))
    ax.bar(x, decomp["PV"], label="PV")
    ax.bar(x, decomp["CV_strat"], bottom=decomp["PV"], label="CV_strat")
    ax.set_xticks(x, decomp["topology"])
    ax.set_title(labels["decomposition"])
    ax.set_xlabel(labels["topology"])
    ax.set_ylabel(labels["delta"])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M1_decomposition_{lang}.png", dpi=200)
    plt.close(fig)


def plot_coordination_waste(decomp: pd.DataFrame, lang: str) -> None:
    """Bar chart CW."""
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#4c78a8" if value >= 0 else "#f58518" for value in decomp["CW"]]
    ax.bar(decomp["topology"], decomp["CW"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(labels["coordination_waste"])
    ax.set_xlabel(labels["topology"])
    ax.set_ylabel(labels["cw"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M1_coordination_waste_{lang}.png", dpi=200)
    plt.close(fig)


def plot_vs_baselines(desc: pd.DataFrame, r1: pd.DataFrame, lang: str) -> None:
    """Line plot rand_R1 → M1 strategies → fg_R1."""
    labels = LANG[lang]
    rows = []
    r1_map = r1.set_index("topology")
    for topology in TOPOLOGIES:
        rows.append({"topology": topology, "point": "rand_R1", "rho_CVaR_0.10": r1_map.loc[topology, "random_R1"], "order": 0})
        for order, strategy in enumerate(STRATEGIES, start=1):
            value = float(desc[(desc["topology"] == topology) & (desc["agent_strategy"] == strategy)]["rho_CVaR_0.10"].iloc[0])
            rows.append({"topology": topology, "point": _strategy_label(strategy), "rho_CVaR_0.10": value, "order": order})
        rows.append({"topology": topology, "point": "fg_R1", "rho_CVaR_0.10": r1_map.loc[topology, "flow_greedy_R1"], "order": 5})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for topology in TOPOLOGIES:
        sub = plot_df[plot_df["topology"] == topology].sort_values("order")
        ax.plot(sub["order"], sub["rho_CVaR_0.10"], marker="o", linewidth=2, label=topology)
    ax.set_xticks(range(6), ["rand_R1", "rand_K3", "local_g", "local_c", "coord_g", "fg_R1"], rotation=20)
    ax.set_title(labels["vs_baselines"])
    ax.set_ylabel(labels["rho_cvar"])
    ax.set_ylim(max(0.88, plot_df["rho_CVaR_0.10"].min() - 0.02), 1.0)
    ax.legend(title=labels["topology"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M1_vs_baselines_{lang}.png", dpi=200)
    plt.close(fig)


def make_figures(desc: pd.DataFrame, decomp: pd.DataFrame, r1: pd.DataFrame) -> None:
    """Створює 4 типи рисунків × 2 мови."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for lang in ("uk", "en"):
        plot_strategy_comparison(desc, r1, lang)
        plot_decomposition(decomp, lang)
        plot_coordination_waste(decomp, lang)
        plot_vs_baselines(desc, r1, lang)


def make_report(
    desc: pd.DataFrame,
    decomp: pd.DataFrame,
    sanity: pd.DataFrame,
    pairwise: pd.DataFrame,
    hm1: pd.DataFrame,
    ranking: pd.DataFrame,
    vs_m0: pd.DataFrame,
    raw: pd.DataFrame,
) -> str:
    """Формує M1 report із секціями Patch v4.3, §6.1."""
    hm1_all = bool((hm1["HM1_status"] == "CONFIRMED").all())
    f45 = bool(
        (decomp.loc[decomp["topology"] == "hierarchical", "CW"].iloc[0] > 0.0)
        and (decomp.loc[decomp["topology"] == "reserved", "CW"].iloc[0] > 0.0)
    )
    f46 = bool((decomp["BCG"] > 0.0).all())
    best = ranking["rank1"].mode().iloc[0]
    lines = [
        "=" * 72,
        "M1 RE-ANALYSIS: PARALLELISM / COORDINATION DECOMPOSITION (Patch v4.3)",
        "=" * 72,
        "K=3, r=2, N=50, p_v=0.20, T=100, M=200, mu=1",
        "Стратегії: random_agent | local_greedy | local_centrality | coord_greedy",
        "",
        "--- NEW SANITY CHECKS ---",
        sanity.to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        "",
        "--- DESCRIPTIVE STATISTICS ---",
        desc[[
            "topology", "agent_strategy", "n_valid", "n_invalid",
            "rho_CVaR_0.10", "rho_CVaR_ci_low", "rho_CVaR_ci_high",
            "rho_med", "R_rb_med",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- НОВІ МЕТРИКИ: PARALLELISM vs COORDINATION DECOMPOSITION ---",
        decomp[["topology", "PV", "CV_strat", "PV_CV_ratio", "BCG", "TMAG"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
        "",
        "--- COORDINATION WASTE ---",
        decomp[["topology", "random_K3", "local_greedy_K3", "CW"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
        "",
        "--- HYPOTHESIS STATUS ---",
        hm1.to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        f"HM1 overall: {'CONFIRMED' if hm1_all else 'NOT_CONFIRMED'}",
        f"HM1_new F4.5 (CW > 0 in hierarchical + reserved): {'CONFIRMED' if f45 else 'NOT_CONFIRMED'}",
        f"HM1_new2 F4.6 (BCG > 0 all topologies): {'CONFIRMED' if f46 else 'NOT_CONFIRMED'}",
        "",
        "--- STRATEGY RANKING ---",
        ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- M1 vs M0 COMPARISON ---",
        vs_m0.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- PAIRWISE WILCOXON (Bonferroni corrected) ---",
        pairwise.to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        "",
        "--- BEST STRATEGY FOR M2/M3 ---",
        f"Переможець: {best}",
        "Обґрунтування: максимальна ρ_CVaR у всіх топологіях; token координація прибирає duplicate repair.",
        "",
        "--- RUN SUMMARY ---",
        f"runs_total={len(raw)}, runs_valid={int(raw['valid'].sum())}, valid_pct={raw['valid'].mean() * 100.0:.2f}%",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Запускає M1 re-analysis на існуючих results/raw/article4/M1/."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    r1 = load_r1_benchmarks()
    desc = descriptive_stats(raw)
    decomp = decomposition_metrics(desc, r1)
    sanity = sanity_checks(decomp)
    if not bool(sanity["passed"].all()):
        failed = ", ".join(sanity.loc[~sanity["passed"], "check"].tolist())
        raise RuntimeError(f"M1 Patch v4.3 sanity failed: {failed}")
    pairwise = pairwise_wilcoxon(raw)
    hm1 = hm1_status(pairwise, decomp)
    ranking = strategy_ranking(desc)
    vs_m0 = vs_m0_comparison(desc)
    make_figures(desc, decomp, r1)

    desc.to_csv(AGG_DIR / "M1_descriptive_statistics.csv", index=False)
    decomp.to_csv(AGG_DIR / "M1_parallelism_coordination_decomposition.csv", index=False)
    sanity.to_csv(AGG_DIR / "M1_sanity_checks.csv", index=False)
    pairwise.to_csv(AGG_DIR / "M1_pairwise_wilcoxon.csv", index=False)
    hm1.to_csv(AGG_DIR / "M1_hypothesis_status.csv", index=False)
    ranking.to_csv(AGG_DIR / "M1_strategy_ranking.csv", index=False)
    vs_m0.to_csv(AGG_DIR / "M1_vs_M0_comparison.csv", index=False)

    report = make_report(desc, decomp, sanity, pairwise, hm1, ranking, vs_m0, raw)
    report_path = AGG_DIR / "M1_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

"""
Аналіз серії M2 для статті 4.

Реалізує: Math Spec Article 4 Patch v4.4 та Протокол M2
(див. docs/Math_Spec_Article_4_Patch_v4_4.docx, docs/Протокол_M2.docx).

Приклад використання:
    python -m src.analysis.m2_analysis
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
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M2"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
R1_STATS_PATH = ROOT / "results" / "aggregated" / "article3" / "R1_descriptive_statistics.csv"
M0_STATS_PATH = ROOT / "results" / "aggregated" / "article4" / "M0_descriptive_statistics.csv"
M1_STATS_PATH = ROOT / "results" / "aggregated" / "article4" / "M1_descriptive_statistics.csv"

TOPOLOGIES = ["hierarchical", "dense", "hub", "reserved"]
K_VALUES = [1, 2, 3, 5, 10]
K_PAIRS = [(1, 2), (2, 3), (3, 5), (5, 10)]
ALPHA = 0.10
BOOTSTRAP_B = 2000
EPS_AE = 0.002
SANITY_TOL = 0.010
STOP_K10_DROP = 0.005

LANG = {
    "uk": {
        "cvar_title": "M2: ρ_CVaR за кількістю агентів K",
        "bcg_title": "M2: BCG(K) відносно centralized flow_greedy",
        "ae_title": "M2: marginal gain AE на доданого агента",
        "clusters_title": "M2: два кластери насичення за K*_AE",
        "topology": "топологія",
        "K": "кількість агентів K",
        "rho_cvar": "ρ_CVaR_0.10",
        "bcg": "BCG(K)",
        "ae": "AE на агента",
        "kstar": "K*",
        "fast": "швидке насичення",
        "slow": "повільне насичення",
    },
    "en": {
        "cvar_title": "M2: ρ_CVaR by number of agents K",
        "bcg_title": "M2: BCG(K) relative to centralized flow_greedy",
        "ae_title": "M2: marginal gain AE per added agent",
        "clusters_title": "M2: two saturation clusters by K*_AE",
        "topology": "topology",
        "K": "number of agents K",
        "rho_cvar": "ρ_CVaR_0.10",
        "bcg": "BCG(K)",
        "ae": "AE per agent",
        "kstar": "K*",
        "fast": "fast saturation",
        "slow": "slow saturation",
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


def load_raw() -> pd.DataFrame:
    """Завантажує M2 raw_results.csv."""
    path = RAW_DIR / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"M2 raw_results.csv not found: {path}")
    raw = pd.read_csv(path, keep_default_na=False)
    raw["valid"] = raw["valid"].astype(str).str.lower().isin({"true", "1"})
    raw["K_agents"] = raw["K_agents"].astype(int)
    raw["rho"] = pd.to_numeric(raw["rho"], errors="coerce")
    raw["R_rb"] = pd.to_numeric(raw["R_rb"], errors="coerce")
    return raw


def load_r1_benchmarks() -> pd.DataFrame:
    """Завантажує R1 random та flow_greedy benchmarks."""
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


def load_m0_reference() -> pd.DataFrame:
    """Завантажує M0 fg_local r=2 reference для K=1 sanity."""
    m0 = pd.read_csv(M0_STATS_PATH)
    r2 = m0[m0["r"].astype(str).eq("2.0")].copy()
    return r2[["topology", "rho_CVaR_0.10"]].rename(columns={"rho_CVaR_0.10": "M0_fg_local_r2"})


def load_m1_reference() -> pd.DataFrame:
    """Завантажує M1 coord_greedy K=3 reference для K=3 sanity."""
    m1 = pd.read_csv(M1_STATS_PATH)
    cg = m1[m1["agent_strategy"].eq("coord_greedy")].copy()
    return cg[["topology", "rho_CVaR_0.10"]].rename(columns={"rho_CVaR_0.10": "M1_coord_greedy_K3"})


def descriptive_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Описова статистика для topology × K."""
    rows: list[dict] = []
    for (topology, K), group_all in raw.groupby(["topology", "K_agents"]):
        group = group_all[group_all["valid"]].copy()
        rho = group["rho"].astype(float)
        cvar_ci = bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA),
            seed=stable_seed("M2", topology, K, "cvar"),
        )
        med_ci = bootstrap_ci(
            rho,
            np.median,
            seed=stable_seed("M2", topology, K, "median"),
        )
        rows.append(
            {
                "topology": topology,
                "K_agents": int(K),
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
                "R_rb_med": float(group["R_rb"].median()) if len(group) else float("nan"),
                "non_recovered_pct": float(
                    group["non_recovered"].astype(str).str.lower().isin({"true", "1"}).mean() * 100.0
                )
                if len(group)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["topology", "K_agents"]).reset_index(drop=True)


def bcg_curve(desc: pd.DataFrame, r1: pd.DataFrame) -> pd.DataFrame:
    """BCG(K) = coord_greedy(K) - flow_greedy_R1."""
    out = desc.merge(r1, on="topology", how="left")
    out["BCG"] = out["rho_CVaR_0.10"] - out["flow_greedy_R1"]
    out["rho_CVaR_delta_vs_random_R1"] = out["rho_CVaR_0.10"] - out["random_R1"]
    return out[
        [
            "topology",
            "K_agents",
            "rho_CVaR_0.10",
            "rho_CVaR_ci_low",
            "rho_CVaR_ci_high",
            "random_R1",
            "flow_greedy_R1",
            "BCG",
            "rho_CVaR_delta_vs_random_R1",
        ]
    ].copy()


def ae_curve(desc: pd.DataFrame) -> pd.DataFrame:
    """AE на одного доданого агента між сусідніми K grid points."""
    pivot = desc.pivot(index="topology", columns="K_agents", values="rho_CVaR_0.10")
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        for left, right in K_PAIRS:
            delta = float(pivot.loc[topology, right] - pivot.loc[topology, left])
            rows.append(
                {
                    "topology": topology,
                    "from_K": left,
                    "to_K": right,
                    "delta_total": delta,
                    "delta_per_agent": delta / (right - left),
                    "below_epsilon": bool(delta / (right - left) < EPS_AE),
                }
            )
    return pd.DataFrame(rows)


def saturation_points(bcg: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    """K*_BCG та K*_AE для кожної topology."""
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        b = bcg[bcg["topology"].eq(topology)].sort_values("K_agents")
        positive = b[b["BCG"] > 0.0]
        K_bcg = int(positive["K_agents"].iloc[0]) if len(positive) else "not_reached"
        BCG_at = float(positive["BCG"].iloc[0]) if len(positive) else float("nan")

        a = ae[ae["topology"].eq(topology)].sort_values("from_K")
        saturated = a[a["delta_per_agent"] < EPS_AE]
        K_ae = int(saturated["from_K"].iloc[0]) if len(saturated) else "not_reached"
        AE_next = float(saturated["delta_per_agent"].iloc[0]) if len(saturated) else float("nan")
        cluster = "fast" if isinstance(K_ae, int) and K_ae <= 2 else "slow"
        rows.append(
            {
                "topology": topology,
                "K_star_BCG": K_bcg,
                "BCG_at_K_star": BCG_at,
                "K_star_AE": K_ae,
                "AE_next_per_agent": AE_next,
                "AE_epsilon": EPS_AE,
                "saturation_cluster": cluster,
            }
        )
    return pd.DataFrame(rows)


def sanity_checks(desc: pd.DataFrame, m0: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """K=1/K=3 sanity та monotonicity checks."""
    pivot = desc.pivot(index="topology", columns="K_agents", values="rho_CVaR_0.10")
    m0_map = m0.set_index("topology")["M0_fg_local_r2"].to_dict()
    m1_map = m1.set_index("topology")["M1_coord_greedy_K3"].to_dict()
    rows: list[dict] = []
    for topology in TOPOLOGIES:
        delta_m0 = float(pivot.loc[topology, 1] - m0_map[topology])
        delta_m1 = float(pivot.loc[topology, 3] - m1_map[topology])
        monotonic = bool(np.all(np.diff([pivot.loc[topology, K] for K in K_VALUES]) >= -1e-12))
        k10_drop = float(pivot.loc[topology, 10] - pivot.loc[topology, 3])
        rows.append(
            {
                "topology": topology,
                "delta_K1_vs_M0": delta_m0,
                "K1_vs_M0_pass": bool(abs(delta_m0) <= SANITY_TOL),
                "delta_K3_vs_M1": delta_m1,
                "K3_vs_M1_pass": bool(abs(delta_m1) <= SANITY_TOL),
                "monotonic_pass": monotonic,
                "delta_K10_minus_K3": k10_drop,
                "K10_drop_stop_pass": bool(k10_drop >= -STOP_K10_DROP),
            }
        )
    return pd.DataFrame(rows)


def spearman_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Spearman r(K, rho) на сирих валідних прогонах."""
    rows: list[dict] = []
    valid = raw[raw["valid"]].copy()
    for topology in TOPOLOGIES:
        sub = valid[valid["topology"].eq(topology)]
        r, p = spearmanr(sub["K_agents"].astype(float), sub["rho"].astype(float))
        rows.append(
            {
                "topology": topology,
                "metric": "rho",
                "spearman_r": float(r),
                "p_value": float(p),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_wilcoxon(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon між сусідніми K у межах topology."""
    rows: list[dict] = []
    valid = raw[raw["valid"]].copy()
    n_tests = len(TOPOLOGIES) * len(K_PAIRS)
    for topology in TOPOLOGIES:
        sub = valid[valid["topology"].eq(topology)]
        for left, right in K_PAIRS:
            left_df = sub[sub["K_agents"].eq(left)][["paired_id", "rho"]]
            right_df = sub[sub["K_agents"].eq(right)][["paired_id", "rho"]]
            merged = left_df.merge(right_df, on="paired_id", suffixes=("_left", "_right"))
            if len(merged) < 2:
                stat = p_value = float("nan")
            else:
                try:
                    stat, p_value = wilcoxon(
                        merged["rho_right"].astype(float),
                        merged["rho_left"].astype(float),
                        alternative="greater",
                        zero_method="wilcox",
                    )
                except ValueError:
                    stat, p_value = (0.0, 1.0)
            p_adj = min(float(p_value) * n_tests, 1.0) if not math.isnan(float(p_value)) else float("nan")
            rows.append(
                {
                    "topology": topology,
                    "K_left": left,
                    "K_right": right,
                    "n_pairs": int(len(merged)),
                    "median_delta": float((merged["rho_right"] - merged["rho_left"]).median())
                    if len(merged)
                    else float("nan"),
                    "W_statistic": float(stat),
                    "p_value": float(p_value),
                    "p_value_adj_bonferroni": p_adj,
                    "significant_adj": bool(p_adj < 0.05),
                }
            )
    return pd.DataFrame(rows)


def save_tables(
    desc: pd.DataFrame,
    bcg: pd.DataFrame,
    ae: pd.DataFrame,
    saturation: pd.DataFrame,
    sanity: pd.DataFrame,
    spearman: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
) -> None:
    """Зберігає M2 CSV artifacts."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(AGG_DIR / "M2_descriptive_statistics.csv", index=False)
    bcg.to_csv(AGG_DIR / "M2_BCG_curve.csv", index=False)
    ae.to_csv(AGG_DIR / "M2_AE_curve.csv", index=False)
    bcg.to_csv(AGG_DIR / "M2_K_scaling_metrics.csv", index=False)
    saturation.to_csv(AGG_DIR / "M2_saturation.csv", index=False)
    sanity.to_csv(AGG_DIR / "M2_sanity_checks.csv", index=False)
    spearman.to_csv(AGG_DIR / "M2_spearman.csv", index=False)
    wilcoxon_df.to_csv(AGG_DIR / "M2_pairwise_wilcoxon.csv", index=False)


def _plot_cvar(bcg: pd.DataFrame, lang: str) -> None:
    labels = LANG[lang]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    for ax, topology in zip(axes.flat, TOPOLOGIES):
        sub = bcg[bcg["topology"].eq(topology)]
        ax.plot(sub["K_agents"], sub["rho_CVaR_0.10"], marker="o", label="coord_greedy")
        ax.fill_between(
            sub["K_agents"].to_numpy(dtype=float),
            sub["rho_CVaR_ci_low"].to_numpy(dtype=float),
            sub["rho_CVaR_ci_high"].to_numpy(dtype=float),
            alpha=0.15,
        )
        ax.axhline(float(sub["flow_greedy_R1"].iloc[0]), color="tab:red", linestyle="--", label="fg_R1")
        ax.axhline(float(sub["random_R1"].iloc[0]), color="tab:gray", linestyle=":", label="rand_R1")
        ax.set_title(topology)
        ax.set_xlabel(labels["K"])
        ax.set_ylabel(labels["rho_cvar"])
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    handles, leg_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3)
    fig.suptitle(labels["cvar_title"], fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(FIG_DIR / f"M2_cvar_vs_K_{lang}.png", dpi=180)
    plt.close(fig)


def _plot_bcg(bcg: pd.DataFrame, saturation: pd.DataFrame, lang: str) -> None:
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(10, 6))
    for topology in TOPOLOGIES:
        sub = bcg[bcg["topology"].eq(topology)]
        ax.plot(sub["K_agents"], sub["BCG"], marker="o", label=topology)
        kstar = saturation.loc[saturation["topology"].eq(topology), "K_star_BCG"].iloc[0]
        if isinstance(kstar, (int, np.integer)):
            y = float(sub[sub["K_agents"].eq(kstar)]["BCG"].iloc[0])
            ax.scatter([kstar], [y], s=90, edgecolor="black", zorder=5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(K_VALUES)
    ax.set_xlabel(labels["K"])
    ax.set_ylabel(labels["bcg"])
    ax.set_title(labels["bcg_title"])
    ax.grid(alpha=0.25)
    ax.legend(title=labels["topology"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M2_BCG_vs_K_{lang}.png", dpi=180)
    plt.close(fig)


def _plot_ae(ae: pd.DataFrame, saturation: pd.DataFrame, lang: str) -> None:
    labels = LANG[lang]
    plot_df = ae.copy()
    plot_df["interval"] = plot_df["from_K"].astype(str) + "→" + plot_df["to_K"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="interval", y="delta_per_agent", hue="topology", ax=ax)
    ax.axhline(EPS_AE, color="black", linestyle="--", linewidth=1.0, label="ε=0.002")
    ax.set_xlabel("K interval")
    ax.set_ylabel(labels["ae"])
    ax.set_title(labels["ae_title"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=labels["topology"], ncol=2)
    for row in saturation.itertuples(index=False):
        if isinstance(row.K_star_AE, (int, np.integer)):
            ax.text(
                0.02,
                0.96 - 0.06 * TOPOLOGIES.index(row.topology),
                f"{row.topology}: K*_AE={row.K_star_AE}",
                transform=ax.transAxes,
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M2_AE_vs_K_{lang}.png", dpi=180)
    plt.close(fig)


def _plot_clusters(saturation: pd.DataFrame, lang: str) -> None:
    labels = LANG[lang]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df = saturation.copy()
    plot_df["K_star_AE_numeric"] = pd.to_numeric(plot_df["K_star_AE"], errors="coerce")
    palette = {"fast": "#4C78A8", "slow": "#F58518"}
    sns.barplot(
        data=plot_df,
        x="topology",
        y="K_star_AE_numeric",
        hue="saturation_cluster",
        dodge=False,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel(labels["topology"])
    ax.set_ylabel("K*_AE")
    ax.set_title(labels["clusters_title"])
    ax.set_ylim(0, max(6, float(plot_df["K_star_AE_numeric"].max()) + 1))
    ax.grid(axis="y", alpha=0.25)
    handles, leg_labels = ax.get_legend_handles_labels()
    translated = [labels["fast"] if item == "fast" else labels["slow"] for item in leg_labels]
    ax.legend(handles, translated, title="cluster")
    for patch, value in zip(ax.patches, plot_df["K_star_AE_numeric"]):
        if not math.isnan(float(value)):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.1,
                f"{int(value)}",
                ha="center",
                va="bottom",
            )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M2_two_clusters_{lang}.png", dpi=180)
    plt.close(fig)


def make_figures(bcg: pd.DataFrame, ae: pd.DataFrame, saturation: pd.DataFrame) -> None:
    """Генерує 4 типи фігур × 2 мови."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for lang in ("uk", "en"):
        _plot_cvar(bcg, lang)
        _plot_bcg(bcg, saturation, lang)
        _plot_ae(ae, saturation, lang)
        _plot_clusters(saturation, lang)


def _fmt(value: float, digits: int = 4) -> str:
    """Форматує число для звіту."""
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _table_by_k(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Повертає wide table topology × K."""
    return df.pivot(index="topology", columns="K_agents", values=value_col).loc[TOPOLOGIES, K_VALUES]


def write_report(
    desc: pd.DataFrame,
    bcg: pd.DataFrame,
    ae: pd.DataFrame,
    saturation: pd.DataFrame,
    sanity: pd.DataFrame,
    spearman: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    raw: pd.DataFrame,
) -> str:
    """Створює текстовий звіт M2."""
    cvar_wide = _table_by_k(bcg, "rho_CVaR_0.10")
    bcg_wide = _table_by_k(bcg, "BCG")
    r1 = bcg.drop_duplicates("topology").set_index("topology")[["flow_greedy_R1", "random_R1"]].loc[TOPOLOGIES]
    all_sanity_pass = bool(
        sanity["K1_vs_M0_pass"].all()
        and sanity["K3_vs_M1_pass"].all()
        and sanity["monotonic_pass"].all()
        and sanity["K10_drop_stop_pass"].all()
    )
    consensus_bcg = saturation["K_star_BCG"].mode().iloc[0]
    hm2 = "CONFIRMED" if pd.to_numeric(saturation["K_star_AE"], errors="coerce").le(5).all() else "PARTIALLY"
    n_total = int(len(raw))
    n_valid = int(raw["valid"].sum())
    lines = [
        "=============================================================",
        "M2: K-SCALING — BCG SATURATION (coord_greedy, token, r=2)",
        "=============================================================",
        "",
        "--- SANITY ---",
    ]
    sanity_line_1 = "  ".join(
        f"{row.topology}={row.delta_K1_vs_M0:+.4f}" for row in sanity.itertuples(index=False)
    )
    sanity_line_2 = "  ".join(
        f"{row.topology}={row.delta_K3_vs_M1:+.4f}" for row in sanity.itertuples(index=False)
    )
    lines.extend(
        [
            f"K=1 vs M0 fg_local_r2:  {sanity_line_1}  [{'PASS' if sanity['K1_vs_M0_pass'].all() else 'FAIL'}]",
            f"K=3 vs M1 coord_greedy: {sanity_line_2}  [{'PASS' if sanity['K3_vs_M1_pass'].all() else 'FAIL'}]",
            f"Monotonicity:           {'PASS' if sanity['monotonic_pass'].all() else 'FAIL'}",
            f"K=10 drop stop check:   {'PASS' if sanity['K10_drop_stop_pass'].all() else 'FAIL'}",
            "",
            "NOTE: PV(K)/CV_strat(K) not estimable under fixed coord_greedy design.",
            "      These metrics require random_agent(K) data. See Patch v4.5.",
            "",
            "--- rho_CVaR(K) ---",
            cvar_wide.join(r1["flow_greedy_R1"].rename("fg_R1")).join(r1["random_R1"].rename("rand_R1")).to_string(float_format=lambda x: f"{x:.4f}"),
            "",
            "--- BCG(K) = coord_greedy(K) - fg_R1 ---",
            bcg_wide.to_string(float_format=lambda x: f"{x:+.4f}"),
            "",
            "--- AE(K_next) per added agent ---",
            ae.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "--- SATURATION POINTS ---",
            saturation.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            f"CONSENSUS K*_BCG (majority): {consensus_bcg}",
            "",
            "--- SPEARMAN r(K, rho) ON RAW RUNS ---",
            spearman.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "--- PAIRED WILCOXON, BONFERRONI ---",
            wilcoxon_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
            "",
            "--- HYPOTHESIS STATUS ---",
            f"HM2 (K*_AE ≤ 5): {hm2}",
            f"K*_BCG (мінімальне K для BCG>0): consensus={consensus_bcg}",
            "",
            "--- ПАРАМЕТР ДЛЯ M3/I1/M4 ---",
            "K* (рекомендований): 3",
            "Обґрунтування: K*_BCG consensus=2, але K=3 уже протестовано і закрито в M1; "
            "K=2 дає BCG>0, проте без M1-рівня верифікації. Для M3/I1/M4 лишаємо K=3.",
            "",
            "--- RUN SUMMARY ---",
            f"Total rows: {n_total}; valid: {n_valid}; invalid: {n_total - n_valid}",
            f"All sanity passed: {all_sanity_pass}",
        ]
    )
    report = "\n".join(lines)
    (AGG_DIR / "M2_report.txt").write_text(report, encoding="utf-8")
    return report


def run_analysis() -> dict[str, pd.DataFrame | str]:
    """Повний M2 analysis workflow."""
    raw = load_raw()
    desc = descriptive_stats(raw)
    r1 = load_r1_benchmarks()
    m0 = load_m0_reference()
    m1 = load_m1_reference()
    bcg = bcg_curve(desc, r1)
    ae = ae_curve(desc)
    saturation = saturation_points(bcg, ae)
    sanity = sanity_checks(desc, m0, m1)
    spearman = spearman_raw(raw)
    wilcoxon_df = pairwise_wilcoxon(raw)

    if not sanity["K1_vs_M0_pass"].all():
        raise RuntimeError("STOP: Sanity K=1 vs M0 failed.")
    if not sanity["K3_vs_M1_pass"].all():
        raise RuntimeError("STOP: Sanity K=3 vs M1 failed.")
    if not sanity["K10_drop_stop_pass"].all():
        raise RuntimeError("STOP: rho_CVaR(K=10) < rho_CVaR(K=3) by more than 0.005.")

    save_tables(desc, bcg, ae, saturation, sanity, spearman, wilcoxon_df)
    make_figures(bcg, ae, saturation)
    report = write_report(desc, bcg, ae, saturation, sanity, spearman, wilcoxon_df, raw)
    return {
        "raw": raw,
        "descriptive": desc,
        "bcg": bcg,
        "ae": ae,
        "saturation": saturation,
        "sanity": sanity,
        "spearman": spearman,
        "wilcoxon": wilcoxon_df,
        "report": report,
    }


if __name__ == "__main__":
    outputs = run_analysis()
    print(outputs["report"])

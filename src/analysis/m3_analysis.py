"""
Аналіз серії M3 для статті 4.

Реалізує: Math Spec Article 4 §3.5, §3.6 та Протокол M3, §7
(див. docs/Math_Spec_Article_4_v4_1.docx, docs/Протокол_M3.docx).

Приклад використання:
    python -m src.analysis.m3_analysis
"""

from __future__ import annotations

import itertools
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.metrics.stochastic import (
    compute_rho_CVaR,
    compute_rho_IQR,
    compute_rho_med,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M3"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
RAW_PATH = RAW_DIR / "raw_results.csv"
M1_STATS_PATH = AGG_DIR / "M1_descriptive_statistics.csv"
ALPHA_CVAR = 0.10
BOOTSTRAP_B = 2000
SANITY_TOL = 0.010
MODES = ["independent", "token", "auction", "broadcast"]
COORD_MODES = ["token", "auction", "broadcast"]
TOPO_ORDER = ["hierarchical", "dense", "hub", "reserved"]
MODE_LABELS = {
    "uk": {
        "independent": "Незалежні",
        "token": "Токен",
        "auction": "Аукціон",
        "broadcast": "Broadcast",
    },
    "en": {
        "independent": "Independent",
        "token": "Token",
        "auction": "Auction",
        "broadcast": "Broadcast",
    },
}
TOPO_LABELS = {
    "uk": {
        "hierarchical": "Ієрархічна",
        "dense": "Щільна",
        "hub": "Хабова",
        "reserved": "Резервована",
    },
    "en": {
        "hierarchical": "Hierarchical",
        "dense": "Dense",
        "hub": "Hub",
        "reserved": "Reserved",
    },
}
TEXT = {
    "uk": {
        "mode_comparison": "M3: порівняння механізмів координації",
        "coordination_value": "M3: цінність координації відносно independent",
        "cost_benefit": "M3: виграш координації та комунікаційна складність",
        "cliff_heatmap": "M3: Cliff δ між режимами координації",
        "topology": "Топологія",
        "mode": "Режим координації",
        "cvar": "ρ_CVaR_0.10",
        "cv": "CV(mode)",
        "cost": "Відносна комунікаційна складність",
        "efficiency": "CV / cost",
    },
    "en": {
        "mode_comparison": "M3: coordination mechanism comparison",
        "coordination_value": "M3: coordination value relative to independent",
        "cost_benefit": "M3: coordination gain and communication cost",
        "cliff_heatmap": "M3: Cliff δ between coordination modes",
        "topology": "Topology",
        "mode": "Coordination mode",
        "cvar": "ρ_CVaR_0.10",
        "cv": "CV(mode)",
        "cost": "Relative communication cost",
        "efficiency": "CV / cost",
    },
}


def _clean(values: pd.Series) -> np.ndarray:
    """Повертає float-масив без NaN."""
    arr = values.astype(float).to_numpy()
    return arr[~np.isnan(arr)]


def _bootstrap_ci(
    values: np.ndarray,
    estimator,
    seed: int,
    B: int = BOOTSTRAP_B,
) -> tuple[float, float]:
    """Percentile bootstrap CI для довільного estimator(values)."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = estimator(sample)
    return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


def _stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap, незалежний від PYTHONHASHSEED."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff δ = P(x>y) - P(x<y)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float(np.sum(np.sign(diff)) / diff.size)


def _effect_size(delta: float) -> str:
    """Інтерпретація Cliff δ за Romano et al. thresholds."""
    if math.isnan(delta):
        return "NA"
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.330:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def load_data() -> pd.DataFrame:
    """Завантажує M3 raw_results.csv і залишає валідні прогони для метрик."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"M3 raw results not found: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def descriptive_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    """ρ_CVaR(mode) × topology з bootstrap CI."""
    rows: list[dict] = []
    for (topology, mode), group_all in raw.groupby(["topology", "coordination_mode"]):
        group = group_all[group_all["valid"]].copy()
        rho = _clean(group["rho"])
        cvar_ci = _bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA_CVAR),
            seed=_stable_seed("M3", topology, mode, "cvar"),
        )
        med_ci = _bootstrap_ci(
            rho,
            compute_rho_med,
            seed=_stable_seed("M3", topology, mode, "median"),
        )
        rows.append({
            "topology": topology,
            "coordination_mode": mode,
            "n_total": int(len(group_all)),
            "n_valid": int(len(group)),
            "n_invalid": int(len(group_all) - len(group)),
            "rho_med": compute_rho_med(rho),
            "rho_med_ci_low": med_ci[0],
            "rho_med_ci_high": med_ci[1],
            "rho_CVaR_0.10": compute_rho_CVaR(rho, alpha=ALPHA_CVAR),
            "rho_CVaR_ci_low": cvar_ci[0],
            "rho_CVaR_ci_high": cvar_ci[1],
            "rho_IQR": compute_rho_IQR(rho),
            "R_rb_med": float(group["R_rb"].median()) if len(group) else float("nan"),
            "non_recovered_pct": float(group["non_recovered"].mean() * 100.0) if len(group) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["topology", "coordination_mode"]).reset_index(drop=True)


def coordination_value(desc: pd.DataFrame) -> pd.DataFrame:
    """CV(mode) = ρ_CVaR(mode) - ρ_CVaR(independent)."""
    rows: list[dict] = []
    base = desc[desc["coordination_mode"] == "independent"].set_index("topology")
    for row in desc.to_dict("records"):
        if row["coordination_mode"] == "independent":
            continue
        independent = float(base.loc[row["topology"], "rho_CVaR_0.10"])
        rows.append({
            "topology": row["topology"],
            "coordination_mode": row["coordination_mode"],
            "rho_CVaR_0.10": float(row["rho_CVaR_0.10"]),
            "independent_rho_CVaR": independent,
            "CV": float(row["rho_CVaR_0.10"]) - independent,
        })
    return pd.DataFrame(rows).sort_values(["topology", "coordination_mode"]).reset_index(drop=True)


def pairwise_wilcoxon(raw: pd.DataFrame) -> pd.DataFrame:
    """Парні Wilcoxon-тести між усіма mode-парами у межах topology."""
    rows: list[dict] = []
    mode_pairs = list(itertools.combinations(MODES, 2))
    n_tests = len(mode_pairs) * len(TOPO_ORDER)
    valid = raw[raw["valid"]].copy()
    for topology in TOPO_ORDER:
        topo = valid[valid["topology"] == topology]
        for mode_a, mode_b in mode_pairs:
            a = topo[topo["coordination_mode"] == mode_a][["paired_id", "rho"]]
            b = topo[topo["coordination_mode"] == mode_b][["paired_id", "rho"]]
            paired = a.merge(b, on="paired_id", suffixes=("_a", "_b"))
            x = paired["rho_a"].astype(float).to_numpy()
            y = paired["rho_b"].astype(float).to_numpy()
            if len(paired) == 0 or np.allclose(x, y):
                stat = 0.0
                p_value = 1.0
            else:
                stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            delta = cliffs_delta(x, y)
            rows.append({
                "topology": topology,
                "mode_A": mode_a,
                "mode_B": mode_b,
                "n_pairs": int(len(paired)),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "p_value_adj_bonferroni": float(min(1.0, p_value * n_tests)),
                "significant": bool((p_value * n_tests) < 0.05),
                "cliffs_delta_A_minus_B": delta,
                "effect_size": _effect_size(delta),
            })
    return pd.DataFrame(rows)


def mode_ranking(desc: pd.DataFrame) -> pd.DataFrame:
    """Ранжування режимів за ρ_CVaR окремо для кожної topology."""
    rows: list[dict] = []
    for topology, group in desc.groupby("topology"):
        ordered = group.sort_values(["rho_CVaR_0.10", "coordination_mode"], ascending=[False, True])
        row = {"topology": topology}
        for rank, mode in enumerate(ordered["coordination_mode"], start=1):
            row[f"rank{rank}"] = mode
        rows.append(row)
    return pd.DataFrame(rows).sort_values("topology").reset_index(drop=True)


def cost_benefit(raw: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Оцінює CV / communication cost за асимптотикою з Протоколу M3."""
    valid = raw[raw["valid"]].copy()
    failed_mean = valid.groupby("topology")["N_failed_initial"].mean().to_dict()
    rows: list[dict] = []
    for row in cv.itertuples(index=False):
        K = 3
        if row.coordination_mode == "token":
            cost = float(K)
        elif row.coordination_mode == "auction":
            cost = float(K * failed_mean.get(row.topology, 0.0))
        elif row.coordination_mode == "broadcast":
            cost = float(K * K)
        else:
            cost = 0.0
        rows.append({
            "topology": row.topology,
            "coordination_mode": row.coordination_mode,
            "CV": float(row.CV),
            "communication_cost": cost,
            "CV_per_cost": float(row.CV / cost) if cost > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["topology", "coordination_mode"]).reset_index(drop=True)


def sanity_checks(desc: pd.DataFrame) -> pd.DataFrame:
    """M3 sanity проти M1 local_greedy і coord_greedy."""
    m1 = pd.read_csv(M1_STATS_PATH)
    m1_local = m1[m1["agent_strategy"] == "local_greedy"].set_index("topology")["rho_CVaR_0.10"]
    m1_coord = m1[m1["agent_strategy"] == "coord_greedy"].set_index("topology")["rho_CVaR_0.10"]
    m3 = desc.pivot(index="topology", columns="coordination_mode", values="rho_CVaR_0.10")
    rows: list[dict] = []
    for topology in TOPO_ORDER:
        independent_delta = float(m3.loc[topology, "independent"] - m1_local.loc[topology])
        token_delta = float(m3.loc[topology, "token"] - m1_coord.loc[topology])
        ordering_pass = bool(all(
            float(m3.loc[topology, mode]) > float(m3.loc[topology, "independent"])
            for mode in COORD_MODES
        ))
        rows.append({
            "topology": topology,
            "M3_independent": float(m3.loc[topology, "independent"]),
            "M1_local_greedy": float(m1_local.loc[topology]),
            "delta_independent_vs_M1": independent_delta,
            "independent_vs_M1_pass": bool(abs(independent_delta) <= SANITY_TOL),
            "M3_token": float(m3.loc[topology, "token"]),
            "M1_coord_greedy": float(m1_coord.loc[topology]),
            "delta_token_vs_M1": token_delta,
            "token_vs_M1_pass": bool(abs(token_delta) <= SANITY_TOL),
            "ordering_pass": ordering_pass,
        })
    return pd.DataFrame(rows)


def hypothesis_status(desc: pd.DataFrame) -> dict[str, object]:
    """Формальна HM3 перевірка за Протоколом M3."""
    pivot = desc.pivot(index="topology", columns="coordination_mode", values="rho_CVaR_0.10")
    token_gt_independent = bool((pivot["token"] > pivot["independent"]).all())
    advanced_ge_token = bool((pivot[["auction", "broadcast"]].max(axis=1) >= pivot["token"]).all())
    all_coord_gt_independent = bool(
        all((pivot[mode] > pivot["independent"]).all() for mode in COORD_MODES)
    )
    return {
        "token_gt_independent": token_gt_independent,
        "advanced_ge_token": advanced_ge_token,
        "all_coord_gt_independent": all_coord_gt_independent,
        "HM3_status": "CONFIRMED" if token_gt_independent and advanced_ge_token else "NOT_CONFIRMED",
    }


def _add_labels(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Додає людиночитні labels для графіків."""
    out = df.copy()
    out["topology_label"] = out["topology"].map(TOPO_LABELS[lang])
    out["mode_label"] = out["coordination_mode"].map(MODE_LABELS[lang])
    return out


def plot_mode_comparison(desc: pd.DataFrame, lang: str) -> None:
    """Bar chart ρ_CVaR × mode × topology."""
    data = _add_labels(desc, lang)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=data, x="topology_label", y="rho_CVaR_0.10", hue="mode_label", ax=ax)
    ax.set_title(TEXT[lang]["mode_comparison"])
    ax.set_xlabel(TEXT[lang]["topology"])
    ax.set_ylabel(TEXT[lang]["cvar"])
    ax.set_ylim(max(0.90, data["rho_CVaR_0.10"].min() - 0.01), 1.0)
    ax.legend(title=TEXT[lang]["mode"], ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M3_mode_comparison_{lang}.png", dpi=200)
    plt.close(fig)


def plot_coordination_value(cv: pd.DataFrame, lang: str) -> None:
    """Bar chart CV(mode) для token/auction/broadcast."""
    data = _add_labels(cv, lang)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=data, x="topology_label", y="CV", hue="mode_label", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(TEXT[lang]["coordination_value"])
    ax.set_xlabel(TEXT[lang]["topology"])
    ax.set_ylabel(TEXT[lang]["cv"])
    ax.legend(title=TEXT[lang]["mode"], ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M3_coordination_value_{lang}.png", dpi=200)
    plt.close(fig)


def plot_cost_benefit(cost: pd.DataFrame, lang: str) -> None:
    """Scatter communication complexity vs CV."""
    data = _add_labels(cost, lang)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=data,
        x="communication_cost",
        y="CV",
        hue="mode_label",
        style="topology_label",
        s=90,
        ax=ax,
    )
    for row in data.itertuples(index=False):
        ax.annotate(row.topology_label, (row.communication_cost, row.CV), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(TEXT[lang]["cost_benefit"])
    ax.set_xlabel(TEXT[lang]["cost"])
    ax.set_ylabel(TEXT[lang]["cv"])
    ax.legend(title="", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M3_cost_benefit_{lang}.png", dpi=200)
    plt.close(fig)


def plot_cliff_heatmap(pairwise: pd.DataFrame, lang: str) -> None:
    """Heatmap Cliff δ між режимами, усереднено по топологіях."""
    rows = []
    for mode_a, mode_b in itertools.permutations(MODES, 2):
        if mode_a == mode_b:
            continue
        direct = pairwise[(pairwise["mode_A"] == mode_a) & (pairwise["mode_B"] == mode_b)]
        reverse = pairwise[(pairwise["mode_A"] == mode_b) & (pairwise["mode_B"] == mode_a)]
        values = []
        if len(direct):
            values.extend(direct["cliffs_delta_A_minus_B"].astype(float).tolist())
        if len(reverse):
            values.extend((-reverse["cliffs_delta_A_minus_B"].astype(float)).tolist())
        rows.append({"mode_A": mode_a, "mode_B": mode_b, "delta": float(np.nanmean(values))})
    mat = pd.DataFrame(rows).pivot(index="mode_A", columns="mode_B", values="delta").reindex(index=MODES, columns=MODES)
    np.fill_diagonal(mat.values, 0.0)
    mat.index = [MODE_LABELS[lang][mode] for mode in mat.index]
    mat.columns = [MODE_LABELS[lang][mode] for mode in mat.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title(TEXT[lang]["cliff_heatmap"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"M3_cliff_heatmap_{lang}.png", dpi=200)
    plt.close(fig)


def make_figures(desc: pd.DataFrame, cv: pd.DataFrame, pairwise: pd.DataFrame, cost: pd.DataFrame) -> None:
    """Створює 4 типи рисунків двома мовами."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for lang in ("uk", "en"):
        plot_mode_comparison(desc, lang)
        plot_coordination_value(cv, lang)
        plot_cost_benefit(cost, lang)
        plot_cliff_heatmap(pairwise, lang)


def make_report(
    desc: pd.DataFrame,
    cv: pd.DataFrame,
    ranking: pd.DataFrame,
    sanity: pd.DataFrame,
    cost: pd.DataFrame,
    hm3: dict,
    elapsed_seconds: float | None = None,
) -> str:
    """Формує M3_report.txt за структурою Протоколу §7.3/§10."""
    pivot = desc.pivot(index="topology", columns="coordination_mode", values="rho_CVaR_0.10").reindex(TOPO_ORDER)
    cv_pivot = cv.pivot(index="topology", columns="coordination_mode", values="CV").reindex(TOPO_ORDER)
    sanity_ind = "PASS" if sanity["independent_vs_M1_pass"].all() else "FAIL"
    sanity_token = "PASS" if sanity["token_vs_M1_pass"].all() else "FAIL"
    sanity_order = "PASS" if sanity["ordering_pass"].all() else "FAIL"

    avg_cv_per_cost = cost.groupby("coordination_mode")["CV_per_cost"].mean().sort_values(ascending=False)
    best_cost_mode = str(avg_cv_per_cost.index[0])
    best_cvar_counts = ranking[[f"rank{i}" for i in range(1, 5)]].iloc[:, 0].value_counts()
    best_cvar_mode = str(best_cvar_counts.index[0])
    recommended = "token" if best_cost_mode == "token" else best_cvar_mode
    if hm3["HM3_status"] != "CONFIRMED" and best_cost_mode == "token":
        recommended = "token"

    lines = [
        "=== M3: COORDINATION MECHANISM COMPARISON ===",
        "local_greedy base, K=3, r=2",
        "",
        "--- SANITY ---",
        f"independent vs M1 local_greedy: {sanity_ind}",
        f"token vs M1 coord_greedy:       {sanity_token}",
        f"ordering (all coord > indep):   {sanity_order}",
        "",
        "--- rho_CVaR(mode) ---",
        pivot[MODES].to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "--- CV(mode) = mode - independent ---",
        cv_pivot[COORD_MODES].to_string(float_format=lambda x: f"{x:+.4f}"),
        "",
        "--- MODE RANKING ---",
        ranking.to_string(index=False),
        "",
        "--- HYPOTHESIS HM3 ---",
        f"HM3 (auction|broadcast >= token > independent): {hm3['HM3_status']}",
        f"token > independent: {hm3['token_gt_independent']}",
        f"max(auction,broadcast) >= token: {hm3['advanced_ge_token']}",
        "",
        "--- COST-BENEFIT ---",
        cost.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        f"Best average CV/cost: {best_cost_mode}",
        f"Best raw rho_CVaR most often: {best_cvar_mode}",
        "",
        "--- BEST MODE FOR I1/M4 ---",
        f"Recommended mode: {recommended}",
        "Rationale: token has near-identical CVaR to auction/broadcast with much lower communication cost."
        if recommended == "token"
        else "Rationale: selected by best aggregate rho_CVaR despite higher communication cost.",
    ]
    if elapsed_seconds is not None:
        lines.extend(["", f"Elapsed seconds: {elapsed_seconds:.1f}"])
    return "\n".join(lines) + "\n"


def run_analysis() -> dict[str, object]:
    """Запускає повний аналіз M3 і зберігає CSV/TXT/PNG artifacts."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    desc = descriptive_statistics(raw)
    cv = coordination_value(desc)
    pairwise = pairwise_wilcoxon(raw)
    ranking = mode_ranking(desc)
    cost = cost_benefit(raw, cv)
    sanity = sanity_checks(desc)
    hm3 = hypothesis_status(desc)

    failed_sanity = sanity[
        ~sanity["independent_vs_M1_pass"]
        | ~sanity["token_vs_M1_pass"]
        | ~sanity["ordering_pass"]
    ]
    if len(failed_sanity):
        raise RuntimeError(
            "M3 STOP: sanity/order check failed:\n"
            + failed_sanity.to_string(index=False)
        )

    desc.to_csv(AGG_DIR / "M3_descriptive_statistics.csv", index=False)
    cv.to_csv(AGG_DIR / "M3_coordination_value.csv", index=False)
    pairwise.to_csv(AGG_DIR / "M3_pairwise_wilcoxon.csv", index=False)
    ranking.to_csv(AGG_DIR / "M3_mode_ranking.csv", index=False)
    cost.to_csv(AGG_DIR / "M3_cost_benefit.csv", index=False)
    sanity.to_csv(AGG_DIR / "M3_sanity_checks.csv", index=False)
    make_figures(desc, cv, pairwise, cost)
    report = make_report(desc, cv, ranking, sanity, cost, hm3)
    (AGG_DIR / "M3_report.txt").write_text(report, encoding="utf-8")
    return {
        "raw": raw,
        "desc": desc,
        "cv": cv,
        "pairwise": pairwise,
        "ranking": ranking,
        "cost": cost,
        "sanity": sanity,
        "hm3": hm3,
        "report": report,
    }


if __name__ == "__main__":
    result = run_analysis()
    print(result["report"])

"""
Аналіз серії I1 для статті 4.

Реалізує: Math Spec Article 4 §3.6, §4.3 та Протокол I1, §7
(див. docs/Math_Spec_Article_4_v4_1.docx, docs/Протокол_I1.docx).

Приклад використання:
    python -m src.analysis.i1_analysis
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "I1"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
RAW_PATH = RAW_DIR / "raw_results.csv"
M2_STATS_PATH = AGG_DIR / "M2_descriptive_statistics.csv"
M3_CV_PATH = AGG_DIR / "M3_coordination_value.csv"
TOPO_ORDER = ["hierarchical", "dense", "hub", "reserved"]
ALPHA_CVAR = 0.10
BOOTSTRAP_B = 2000
SANITY_TOL = 0.010
NEGATIVE_IV_TOL = -0.005
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
        "iv_vs_cv": "I1: цінність інформації проти координації",
        "paired_diff": "I1: парні різниці ρ(r=∞) − ρ(r=2)",
        "scatter": "I1: ρ(r=2) проти ρ(r=∞)",
        "invariance": "I1: visibility invariance у multi-agent recovery",
        "topology": "Топологія",
        "value": "Значення",
        "metric": "Метрика",
        "diff": "ρ(r=∞) − ρ(r=2)",
        "rho_r2": "ρ(r=2)",
        "rho_inf": "ρ(r=∞)",
        "iv": "IV",
        "cv": "CV(token)",
    },
    "en": {
        "iv_vs_cv": "I1: information value versus coordination value",
        "paired_diff": "I1: paired differences ρ(r=∞) − ρ(r=2)",
        "scatter": "I1: ρ(r=2) versus ρ(r=∞)",
        "invariance": "I1: visibility invariance in multi-agent recovery",
        "topology": "Topology",
        "value": "Value",
        "metric": "Metric",
        "diff": "ρ(r=∞) − ρ(r=2)",
        "rho_r2": "ρ(r=2)",
        "rho_inf": "ρ(r=∞)",
        "iv": "IV",
        "cv": "CV(token)",
    },
}


def _stable_seed(*parts: object) -> int:
    """Стабільний seed для bootstrap."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _clean(values: pd.Series | np.ndarray) -> np.ndarray:
    """Повертає float array без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def _bootstrap_ci(values: np.ndarray, estimator, seed: int, B: int = BOOTSTRAP_B) -> tuple[float, float]:
    """Percentile bootstrap CI."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = estimator(sample)
    return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


def _paired_cliff_delta(r_inf: np.ndarray, r2: np.ndarray) -> float:
    """Парний Cliff-like δ: частка positive мінус частка negative differences."""
    diff = r_inf - r2
    if len(diff) == 0:
        return float("nan")
    return float((np.sum(diff > 0.0) - np.sum(diff < 0.0)) / len(diff))


def load_raw() -> pd.DataFrame:
    """Завантажує I1 raw_results.csv."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"I1 raw results not found: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def _paired_valid(raw: pd.DataFrame) -> pd.DataFrame:
    """Повертає wide table з валідними r2/r_inf парами."""
    valid = raw[raw["valid"]].copy()
    wide = valid.pivot_table(
        index=["topology", "paired_run_id"],
        columns="r_label",
        values="rho",
        aggfunc="first",
    ).reset_index()
    return wide.dropna(subset=["r2", "r_inf"]).copy()


def descriptive_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    """ρ_CVaR(r=2), ρ_CVaR(r=∞) з bootstrap CI."""
    rows: list[dict] = []
    for (topology, r_label), group_all in raw.groupby(["topology", "r_label"]):
        group = group_all[group_all["valid"]].copy()
        rho = _clean(group["rho"])
        cvar_ci = _bootstrap_ci(
            rho,
            lambda sample: compute_rho_CVaR(sample, alpha=ALPHA_CVAR),
            seed=_stable_seed("I1", topology, r_label, "cvar"),
        )
        med_ci = _bootstrap_ci(
            rho,
            compute_rho_med,
            seed=_stable_seed("I1", topology, r_label, "median"),
        )
        rows.append({
            "topology": topology,
            "r_label": r_label,
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
        })
    return pd.DataFrame(rows).sort_values(["topology", "r_label"]).reset_index(drop=True)


def information_value(raw: pd.DataFrame) -> pd.DataFrame:
    """IV = ρ_CVaR(r=∞) − ρ_CVaR(r=2), bootstrap на парах."""
    wide = _paired_valid(raw)
    rows: list[dict] = []
    for topology in TOPO_ORDER:
        group = wide[wide["topology"] == topology].copy()
        r2 = _clean(group["r2"])
        r_inf = _clean(group["r_inf"])
        if len(group) == 0:
            continue

        def estimator(indices: np.ndarray) -> float:
            return compute_rho_CVaR(r_inf[indices], ALPHA_CVAR) - compute_rho_CVaR(r2[indices], ALPHA_CVAR)

        rng = np.random.default_rng(_stable_seed("I1", topology, "iv"))
        boot = np.empty(BOOTSTRAP_B, dtype=float)
        n = len(group)
        for i in range(BOOTSTRAP_B):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            boot[i] = estimator(idx)

        rho_r2 = compute_rho_CVaR(r2, ALPHA_CVAR)
        rho_inf = compute_rho_CVaR(r_inf, ALPHA_CVAR)
        iv = rho_inf - rho_r2
        rows.append({
            "topology": topology,
            "n_pairs": int(n),
            "rho_CVaR_r2": rho_r2,
            "rho_CVaR_r_inf": rho_inf,
            "IV": iv,
            "IV_ci_low": float(np.quantile(boot, 0.025)),
            "IV_ci_high": float(np.quantile(boot, 0.975)),
            "paired_diff_mean": float(np.mean(r_inf - r2)),
            "paired_diff_median": float(np.median(r_inf - r2)),
            "cliff_d": _paired_cliff_delta(r_inf, r2),
            "IV_nonnegative_pass": bool(iv >= NEGATIVE_IV_TOL),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(raw: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank r=2 vs r=∞ у межах topology."""
    wide = _paired_valid(raw)
    rows: list[dict] = []
    for topology in TOPO_ORDER:
        group = wide[wide["topology"] == topology]
        r2 = _clean(group["r2"])
        r_inf = _clean(group["r_inf"])
        if len(group) == 0 or np.allclose(r2, r_inf):
            stat = 0.0
            p_value = 1.0
        else:
            stat, p_value = stats.wilcoxon(r_inf, r2, zero_method="wilcox", alternative="two-sided")
        rows.append({
            "topology": topology,
            "n_pairs": int(len(group)),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p_value),
            "significant_0.05": bool(p_value < 0.05),
        })
    return pd.DataFrame(rows)


def sanity_checks(raw: pd.DataFrame, desc: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """I1 sanity: парність, r=2≈M2 K=3, IV не суттєво негативний."""
    counts = raw.groupby("paired_run_id").size()
    pairing_pass = bool((counts == 2).all())
    labels_ok = bool(raw.groupby("paired_run_id")["r_label"].apply(lambda s: set(s) == {"r2", "r_inf"}).all())
    m2 = pd.read_csv(M2_STATS_PATH)
    m2_k3 = m2[m2["K_agents"] == 3].set_index("topology")["rho_CVaR_0.10"]
    r2 = desc[desc["r_label"] == "r2"].set_index("topology")["rho_CVaR_0.10"]
    rows: list[dict] = []
    for topology in TOPO_ORDER:
        delta = float(r2.loc[topology] - m2_k3.loc[topology])
        iv_value = float(iv.set_index("topology").loc[topology, "IV"])
        rows.append({
            "topology": topology,
            "pairing_count_pass": pairing_pass,
            "pairing_labels_pass": labels_ok,
            "I1_r2_CVaR": float(r2.loc[topology]),
            "M2_K3_CVaR": float(m2_k3.loc[topology]),
            "delta_r2_vs_M2": delta,
            "r2_vs_M2_pass": bool(abs(delta) <= SANITY_TOL),
            "IV": iv_value,
            "IV_nonnegative_pass": bool(iv_value >= NEGATIVE_IV_TOL),
        })
    return pd.DataFrame(rows)


def iv_vs_cv(iv: pd.DataFrame) -> pd.DataFrame:
    """Порівнює IV з CV(token) з M3."""
    cv = pd.read_csv(M3_CV_PATH)
    token_cv = cv[cv["coordination_mode"] == "token"].set_index("topology")["CV"]
    rows: list[dict] = []
    for row in iv.itertuples(index=False):
        cv_value = float(token_cv.loc[row.topology])
        rows.append({
            "topology": row.topology,
            "IV": float(row.IV),
            "CV_token_M3": cv_value,
            "IV_minus_CV": float(row.IV - cv_value),
            "IV_lt_CV": bool(row.IV < cv_value),
            "conclusion": "coordination > visibility" if row.IV < cv_value else "visibility >= coordination",
        })
    return pd.DataFrame(rows)


def _add_topology_labels(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Додає labels для рисунків."""
    out = df.copy()
    out["topology_label"] = out["topology"].map(TOPO_LABELS[lang])
    return out


def plot_iv_vs_cv(compare: pd.DataFrame, lang: str) -> None:
    """Grouped bar IV та CV(token)."""
    data = compare.melt(
        id_vars=["topology"],
        value_vars=["IV", "CV_token_M3"],
        var_name="metric",
        value_name="value",
    )
    data["metric"] = data["metric"].map({"IV": TEXT[lang]["iv"], "CV_token_M3": TEXT[lang]["cv"]})
    data = _add_topology_labels(data, lang)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.barplot(data=data, x="topology_label", y="value", hue="metric", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(TEXT[lang]["iv_vs_cv"])
    ax.set_xlabel(TEXT[lang]["topology"])
    ax.set_ylabel(TEXT[lang]["value"])
    ax.legend(title=TEXT[lang]["metric"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"I1_IV_vs_CV_{lang}.png", dpi=200)
    plt.close(fig)


def plot_paired_diff(raw: pd.DataFrame, lang: str) -> None:
    """Distribution paired differences."""
    wide = _paired_valid(raw)
    wide["diff"] = wide["r_inf"] - wide["r2"]
    data = _add_topology_labels(wide, lang)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.violinplot(data=data, x="topology_label", y="diff", inner="quartile", cut=0, ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(TEXT[lang]["paired_diff"])
    ax.set_xlabel(TEXT[lang]["topology"])
    ax.set_ylabel(TEXT[lang]["diff"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"I1_paired_diff_{lang}.png", dpi=200)
    plt.close(fig)


def plot_rho_scatter(raw: pd.DataFrame, lang: str) -> None:
    """Scatter ρ(r=2) vs ρ(r=∞)."""
    wide = _paired_valid(raw)
    data = _add_topology_labels(wide, lang)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    sns.scatterplot(data=data, x="r2", y="r_inf", hue="topology_label", alpha=0.7, s=35, ax=ax)
    lo = min(float(data["r2"].min()), float(data["r_inf"].min())) - 0.002
    hi = min(1.0, max(float(data["r2"].max()), float(data["r_inf"].max())) + 0.002)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(TEXT[lang]["scatter"])
    ax.set_xlabel(TEXT[lang]["rho_r2"])
    ax.set_ylabel(TEXT[lang]["rho_inf"])
    ax.legend(title=TEXT[lang]["topology"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"I1_rho_r2_vs_inf_{lang}.png", dpi=200)
    plt.close(fig)


def plot_visibility_invariance(iv: pd.DataFrame, lang: str) -> None:
    """IV by topology з emphasis на near-zero visibility value."""
    data = _add_topology_labels(iv, lang)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.barplot(data=data, x="topology_label", y="IV", ax=ax, color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(0.005, color="#B55", linewidth=1, linestyle="--")
    ax.axhline(-0.005, color="#B55", linewidth=1, linestyle="--")
    ax.set_title(TEXT[lang]["invariance"])
    ax.set_xlabel(TEXT[lang]["topology"])
    ax.set_ylabel(TEXT[lang]["iv"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"I1_visibility_invariance_{lang}.png", dpi=200)
    plt.close(fig)


def make_figures(raw: pd.DataFrame, iv: pd.DataFrame, compare: pd.DataFrame) -> None:
    """Створює 4 типи рисунків двома мовами."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for lang in ("uk", "en"):
        plot_iv_vs_cv(compare, lang)
        plot_paired_diff(raw, lang)
        plot_rho_scatter(raw, lang)
        plot_visibility_invariance(iv, lang)


def make_report(
    desc: pd.DataFrame,
    iv: pd.DataFrame,
    wilcoxon: pd.DataFrame,
    compare: pd.DataFrame,
    sanity: pd.DataFrame,
    raw: pd.DataFrame,
) -> str:
    """Формує I1_report.txt за шаблоном Протоколу §10."""
    desc_pivot = desc.pivot(index="topology", columns="r_label", values="rho_CVaR_0.10")
    info = iv.set_index("topology")
    comp = compare.set_index("topology")
    sanity_r2 = "PASS" if sanity["r2_vs_M2_pass"].all() else "FAIL"
    sanity_iv = "PASS" if sanity["IV_nonnegative_pass"].all() else "FAIL"
    sanity_pair = "PASS" if (sanity["pairing_count_pass"].all() and sanity["pairing_labels_pass"].all()) else "FAIL"
    hi1 = "CONFIRMED" if compare["IV_lt_CV"].all() else "NOT_CONFIRMED"
    valid_count = int(raw["valid"].sum())
    total_count = int(len(raw))
    pair_count = int(raw["paired_run_id"].nunique())

    lines = [
        "=== I1: INFORMATION VALUE (paired r=2 vs r=inf) ===",
        "coord_greedy, token, K=3",
        "",
        "--- SANITY ---",
        f"r=2 vs M2 K=3: {sanity_r2}",
        f"IV >= 0 (r=inf >= r=2): {sanity_iv}",
        f"pairing structure (2 rows/pair): {sanity_pair}",
        "",
        "--- INFORMATION VALUE ---",
        "topology       rho_r2   rho_inf  IV       cliff_d",
    ]
    for topology in TOPO_ORDER:
        lines.append(
            f"{topology:<13} "
            f"{desc_pivot.loc[topology, 'r2']:.4f}   "
            f"{desc_pivot.loc[topology, 'r_inf']:.4f}   "
            f"{info.loc[topology, 'IV']:+.4f}   "
            f"{info.loc[topology, 'cliff_d']:+.2f}"
        )
    lines.extend([
        "",
        "--- IV vs CV(token from M3) ---",
        "topology       IV(I1)   CV(M3)   IV<CV?",
    ])
    for topology in TOPO_ORDER:
        lines.append(
            f"{topology:<13} "
            f"{comp.loc[topology, 'IV']:+.4f}   "
            f"{comp.loc[topology, 'CV_token_M3']:.4f}   "
            f"{'yes' if comp.loc[topology, 'IV_lt_CV'] else 'no'}"
        )
    lines.extend([
        "",
        "--- HI1 STATUS ---",
        f"HI1 (IV < CV): {hi1}",
        "",
        "--- ВИСНОВОК ДЛЯ СТАТТІ ---",
        "Координація token має значно більшу практичну цінність, ніж перехід від r=2 до повної видимості. "
        "IV близька до нуля для всіх топологій; це підтримує visibility invariance у multi-agent режимі.",
        "",
        "--- WILCOXON ---",
        wilcoxon.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        "",
        f"ПРОГОНИ: {total_count} ({pair_count} пар); N_valid={valid_count}",
    ])
    return "\n".join(lines) + "\n"


def run_analysis() -> dict[str, object]:
    """Виконує повний I1 аналіз і зберігає CSV/TXT/PNG artifacts."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    desc = descriptive_statistics(raw)
    iv = information_value(raw)
    wilcoxon = paired_wilcoxon(raw)
    sanity = sanity_checks(raw, desc, iv)
    compare = iv_vs_cv(iv)

    if not sanity["pairing_count_pass"].all() or not sanity["pairing_labels_pass"].all():
        raise RuntimeError("I1 STOP: paired_run_id structure is invalid.")
    if not sanity["r2_vs_M2_pass"].all():
        raise RuntimeError("I1 STOP: r=2 vs M2 K=3 sanity failed:\n" + sanity.to_string(index=False))
    if (iv["IV"] < NEGATIVE_IV_TOL).any():
        raise RuntimeError("I1 STOP: IV is substantially negative:\n" + iv.to_string(index=False))

    desc.to_csv(AGG_DIR / "I1_descriptive_statistics.csv", index=False)
    iv.to_csv(AGG_DIR / "I1_information_value.csv", index=False)
    wilcoxon.to_csv(AGG_DIR / "I1_paired_wilcoxon.csv", index=False)
    compare.to_csv(AGG_DIR / "I1_IV_vs_CV.csv", index=False)
    sanity.to_csv(AGG_DIR / "I1_sanity_checks.csv", index=False)
    make_figures(raw, iv, compare)
    report = make_report(desc, iv, wilcoxon, compare, sanity, raw)
    (AGG_DIR / "I1_report.txt").write_text(report, encoding="utf-8")
    return {
        "raw": raw,
        "desc": desc,
        "iv": iv,
        "wilcoxon": wilcoxon,
        "compare": compare,
        "sanity": sanity,
        "report": report,
    }


if __name__ == "__main__":
    result = run_analysis()
    print(result["report"])

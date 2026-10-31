"""
Аналіз серії S5: масштабний аналіз resilience.

Гіпотеза H6: якісні висновки H1/H2 (рейтинг топологій) зберігаються
при збільшенні N від 30 до 120.

Що робить:
  1. Завантажує raw_results.csv серії S5.
  2. Описова статистика: ρ_med, ρ_CVaR, R_rb для кожного (topology, N).
  3. H6: Spearman correlation між рейтингами топологій при різних N.
  4. Додатковий аналіз: абсолютна зміна ρ з N по топологіях.
  5. Зберігає results/aggregated/S5_hypothesis_test.txt.
  6. Графіки:
       results/figures/S5_rho_vs_N.png
       results/figures/S5_cvar_vs_N.png
       results/figures/S5_ranking_stability.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "results" / "raw" / "S5" / "raw_results.csv"
AGG_DIR = ROOT / "results" / "aggregated"
FIG_DIR = ROOT / "results" / "figures"
AGG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_CVAR = 0.10
BOOTSTRAP_B = 2000
N_VALUES = [30, 50, 80, 120]
TOPO_ORDER = ["hierarchical", "reserved", "hub", "dense"]
TOPO_COLORS = {
    "hierarchical": "#2271b5",
    "reserved":     "#e69f00",
    "hub":          "#009e73",
    "dense":        "#cc79a7",
}
TOPO_MARKERS = {"hierarchical": "o", "reserved": "s", "hub": "^", "dense": "D"}


# ── Допоміжні функції ─────────────────────────────────────────────────────────
def cvar(arr: np.ndarray, alpha: float = ALPHA_CVAR) -> float:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    threshold = np.percentile(arr, alpha * 100)
    tail = arr[arr <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else threshold


def bootstrap_ci(arr: np.ndarray, stat_fn, B: int = BOOTSTRAP_B,
                 ci: float = 0.95) -> tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(B)]
    lo = (1 - ci) / 2 * 100
    return float(np.percentile(stats, lo)), float(np.percentile(stats, 100 - lo))


# ── Завантаження та описова статистика ────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df = df[df["series"] == "S5"].copy()
    df["valid"] = df["valid"].astype(bool)
    df["N"] = df["N"].astype(int)
    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["valid"]]
    rows = []
    for topo in TOPO_ORDER:
        for N in N_VALUES:
            sub = valid[(valid["topology"] == topo) & (valid["N"] == N)]
            n = len(sub)
            if n == 0:
                rows.append({"topology": topo, "N": N, "n": 0,
                              "rho_med": np.nan, "rho_CVaR": np.nan,
                              "rho_IQR": np.nan, "R_rb_med": np.nan,
                              "ci_lo": np.nan, "ci_hi": np.nan})
                continue
            rho = sub["rho"].values
            rb = sub["R_rb"].values
            rho_med = float(np.median(rho))
            rho_cvar = cvar(rho)
            rho_iqr = float(np.percentile(rho, 75) - np.percentile(rho, 25))
            rb_med = float(np.median(rb))
            ci_lo, ci_hi = bootstrap_ci(rho, np.median)
            rows.append({"topology": topo, "N": N, "n": n,
                         "rho_med": rho_med, "rho_CVaR": rho_cvar,
                         "rho_IQR": rho_iqr, "R_rb_med": rb_med,
                         "ci_lo": ci_lo, "ci_hi": ci_hi})
    return pd.DataFrame(rows)


# ── H6: Spearman correlation між рейтингами ───────────────────────────────────
def ranking_analysis(desc: pd.DataFrame) -> dict:
    """
    Для кожного N будуємо рейтинг топологій за ρ_med та ρ_CVaR
    (1 = найкраща, 4 = найгірша).
    Порівнюємо рейтинги між різними N через Spearman correlation.
    """
    rho_ranks = {}   # N → list of ranks (topology order from TOPO_ORDER)
    cvar_ranks = {}

    for N in N_VALUES:
        sub = desc[desc["N"] == N].set_index("topology")
        # Сортуємо за спаданням ρ_med (більший = кращий = нижчий rank)
        rho_sorted = sub["rho_med"].dropna().sort_values(ascending=False)
        cvar_sorted = sub["rho_CVaR"].dropna().sort_values(ascending=False)

        rho_rank = {topo: rank + 1 for rank, topo in enumerate(rho_sorted.index)}
        cvar_rank = {topo: rank + 1 for rank, topo in enumerate(cvar_sorted.index)}

        # Вектор рангів у фіксованому порядку TOPO_ORDER
        rho_ranks[N] = [rho_rank.get(t, np.nan) for t in TOPO_ORDER]
        cvar_ranks[N] = [cvar_rank.get(t, np.nan) for t in TOPO_ORDER]

    # Попарний Spearman між N-значеннями
    spearman_rho = {}
    spearman_cvar = {}
    for i, N1 in enumerate(N_VALUES):
        for j, N2 in enumerate(N_VALUES):
            if i >= j:
                continue
            r_rho, p_rho = spearmanr(rho_ranks[N1], rho_ranks[N2])
            r_cvar, p_cvar = spearmanr(cvar_ranks[N1], cvar_ranks[N2])
            spearman_rho[(N1, N2)] = (float(r_rho), float(p_rho))
            spearman_cvar[(N1, N2)] = (float(r_cvar), float(p_cvar))

    return {
        "rho_ranks": rho_ranks,
        "cvar_ranks": cvar_ranks,
        "spearman_rho": spearman_rho,
        "spearman_cvar": spearman_cvar,
    }


# ── Абсолютна зміна ρ з N ────────────────────────────────────────────────────
def scale_trend(desc: pd.DataFrame) -> dict:
    """Spearman correlation між N і ρ_med для кожної топології."""
    trends = {}
    for topo in TOPO_ORDER:
        sub = desc[desc["topology"] == topo].dropna(subset=["rho_med"])
        if len(sub) < 2:
            trends[topo] = {"r": np.nan, "p": np.nan, "direction": "?"}
            continue
        r, p = spearmanr(sub["N"], sub["rho_med"])
        if abs(r) < 0.3:
            direction = "flat"
        elif r > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        trends[topo] = {"r": float(r), "p": float(p), "direction": direction}
    return trends


# ── Запис звіту ───────────────────────────────────────────────────────────────
def write_report(desc: pd.DataFrame, ranking: dict, trends: dict,
                 out_path: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("СЕРІЯ S5: МАСШТАБНИЙ АНАЛІЗ — ВПЛИВ N НА RESILIENCE")
    lines.append("Гіпотеза H6: рейтинг топологій за ρ зберігається при зміні N")
    lines.append("=" * 72)
    lines.append(f"\nДизайн: 4 топології × 4 N = 16 конфігурацій × 100 прогонів")
    lines.append(f"N ∈ {N_VALUES}, p_v=0.20, T=100, M=100")

    # ── Описова статистика ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 72)
    lines.append("ОПИСОВА СТАТИСТИКА")
    lines.append("─" * 72)
    header = f"{'Topology':<14} {'N':>4}  {'n':>5}  {'ρ_med':>7}  {'ρ_CVaR':>7}  {'ρ_IQR':>7}  {'R_rb_med':>8}  {'CI 95%'}"
    lines.append(header)
    lines.append("─" * 72)
    for topo in TOPO_ORDER:
        for N in N_VALUES:
            row = desc[(desc["topology"] == topo) & (desc["N"] == N)]
            if len(row) == 0 or row.iloc[0]["n"] == 0:
                lines.append(f"{topo:<14} {N:>4}  {'N/A':>5}")
                continue
            r = row.iloc[0]
            ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            lines.append(
                f"{topo:<14} {N:>4}  {int(r['n']):>5}  "
                f"{r['rho_med']:.4f}  {r['rho_CVaR']:.4f}  "
                f"{r['rho_IQR']:.4f}  {r['R_rb_med']:.4f}  {ci}"
            )

    # ── Рейтинги ──────────────────────────────────────────────────────────
    lines.append("\n" + "─" * 72)
    lines.append("РЕЙТИНГИ ТОПОЛОГІЙ ЗА ρ_med (1=найкраща, 4=найгірша)")
    lines.append("─" * 72)
    header2 = f"{'Topology':<14}" + "".join(f"  N={N:>3}" for N in N_VALUES)
    lines.append(header2)
    for i, topo in enumerate(TOPO_ORDER):
        ranks = [ranking["rho_ranks"][N][i] for N in N_VALUES]
        rank_str = "".join(f"  {int(r):>5}" if not np.isnan(r) else "    N/A" for r in ranks)
        lines.append(f"{topo:<14}{rank_str}")

    lines.append("\nРЕЙТИНГИ ТОПОЛОГІЙ ЗА ρ_CVaR")
    lines.append(header2)
    for i, topo in enumerate(TOPO_ORDER):
        ranks = [ranking["cvar_ranks"][N][i] for N in N_VALUES]
        rank_str = "".join(f"  {int(r):>5}" if not np.isnan(r) else "    N/A" for r in ranks)
        lines.append(f"{topo:<14}{rank_str}")

    # ── Spearman між рейтингами ────────────────────────────────────────────
    lines.append("\n" + "─" * 72)
    lines.append("SPEARMAN CORRELATION МІЖ РЕЙТИНГАМИ (різні N)")
    lines.append("─" * 72)
    lines.append(f"{'Pair':>12}  {'ρ_med r':>8}  {'p':>10}  {'CVaR r':>8}  {'p':>10}  H6 підтв.?")
    lines.append("─" * 72)

    all_rho_r = []
    h6_confirmed = 0
    h6_total = 0
    for (N1, N2) in [(N_VALUES[i], N_VALUES[j])
                     for i in range(len(N_VALUES))
                     for j in range(i+1, len(N_VALUES))]:
        r_rho, p_rho = ranking["spearman_rho"][(N1, N2)]
        r_cvar, p_cvar = ranking["spearman_cvar"][(N1, N2)]
        h6_ok = r_rho > 0.8
        h6_total += 1
        if h6_ok:
            h6_confirmed += 1
        all_rho_r.append(r_rho)
        verdict = "YES" if h6_ok else "no"
        lines.append(
            f"N={N1:3} vs N={N2:3}  {r_rho:>+8.3f}  {p_rho:>10.4e}  "
            f"{r_cvar:>+8.3f}  {p_cvar:>10.4e}  {verdict:>9}"
        )

    lines.append(f"\nH6 підтверджена для {h6_confirmed}/{h6_total} пар (поріг r>0.80).")
    min_r = min(all_rho_r)
    lines.append(f"Мінімальний Spearman (ρ_med): {min_r:+.3f}")

    # ── Абсолютний тренд ──────────────────────────────────────────────────
    lines.append("\n" + "─" * 72)
    lines.append("АБСОЛЮТНИЙ ТРЕНД ρ_med з N (Spearman r між N і ρ_med)")
    lines.append("─" * 72)
    for topo in TOPO_ORDER:
        t = trends[topo]
        lines.append(
            f"  {topo:<14}: r={t['r']:+.3f}  p={t['p']:.4e}  → {t['direction']}"
        )

    # ── Висновок ──────────────────────────────────────────────────────────
    lines.append("\n" + "─" * 72)
    lines.append("ВИСНОВОК")
    lines.append("─" * 72)

    if h6_confirmed == h6_total:
        verdict = "ПОВНІСТЮ ПІДТВЕРДЖЕНА"
    elif h6_confirmed >= h6_total * 0.7:
        verdict = "ПЕРЕВАЖНО ПІДТВЕРДЖЕНА"
    else:
        verdict = "НЕ ПІДТВЕРДЖЕНА"

    lines.append(f"\nH6 {verdict}: {h6_confirmed}/{h6_total} пар мають Spearman > 0.80")
    lines.append("\nКлючові спостереження:")

    # Dynamic observations based on data
    for topo in TOPO_ORDER:
        t = trends[topo]
        rho_vals = [desc[(desc["topology"] == topo) & (desc["N"] == N)]["rho_med"].values
                    for N in N_VALUES]
        rho_vals = [v[0] for v in rho_vals if len(v) > 0 and not np.isnan(v[0])]
        if len(rho_vals) >= 2:
            delta = rho_vals[-1] - rho_vals[0]
            lines.append(f"  {topo:<14}: Δρ (N=30→120) = {delta:+.4f}, "
                         f"тренд={t['direction']} (r={t['r']:+.3f})")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Звіт збережено: {out_path}")


# ── Рисунок 1: ρ_med vs N ─────────────────────────────────────────────────────
def plot_rho_vs_N(desc: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for topo in TOPO_ORDER:
        sub = desc[desc["topology"] == topo].sort_values("N")
        valid = sub.dropna(subset=["rho_med"])
        if valid.empty:
            continue
        Ns = valid["N"].values
        med = valid["rho_med"].values
        ci_lo = valid["ci_lo"].values
        ci_hi = valid["ci_hi"].values

        ax.plot(Ns, med, color=TOPO_COLORS[topo], marker=TOPO_MARKERS[topo],
                linewidth=2, markersize=7, label=topo)
        ax.fill_between(Ns, ci_lo, ci_hi, alpha=0.15, color=TOPO_COLORS[topo])

    ax.set_xlabel("Розмір мережі N", fontsize=12)
    ax.set_ylabel("ρ_med (медіана resilience)", fontsize=12)
    ax.set_title("S5: ρ_med як функція N (95% CI Bootstrap)", fontsize=13)
    ax.set_xticks(N_VALUES)
    ax.set_xlim(N_VALUES[0] - 5, N_VALUES[-1] + 5)
    ax.set_ylim(bottom=max(0, desc["rho_med"].min() - 0.02))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ρ_med vs N збережено: {out_path}")


# ── Рисунок 2: ρ_CVaR vs N ───────────────────────────────────────────────────
def plot_cvar_vs_N(desc: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for topo in TOPO_ORDER:
        sub = desc[desc["topology"] == topo].sort_values("N")
        valid = sub.dropna(subset=["rho_CVaR"])
        if valid.empty:
            continue
        Ns = valid["N"].values
        cvar_vals = valid["rho_CVaR"].values

        ax.plot(Ns, cvar_vals, color=TOPO_COLORS[topo], marker=TOPO_MARKERS[topo],
                linewidth=2, markersize=7, label=topo, linestyle="--")

    ax.set_xlabel("Розмір мережі N", fontsize=12)
    ax.set_ylabel("ρ_CVaR₁₀% (середнє нижніх 10%)", fontsize=12)
    ax.set_title("S5: ρ_CVaR₁₀% як функція N — ризик найгіршого сценарію", fontsize=13)
    ax.set_xticks(N_VALUES)
    ax.set_xlim(N_VALUES[0] - 5, N_VALUES[-1] + 5)
    ax.set_ylim(bottom=max(0, desc["rho_CVaR"].min() - 0.02))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ρ_CVaR vs N збережено: {out_path}")


# ── Рисунок 3: Ranking stability heatmap ─────────────────────────────────────
def plot_ranking_stability(desc: pd.DataFrame, ranking: dict, out_path: Path) -> None:
    """
    2 heatmap поруч: рейтинги за ρ_med та за ρ_CVaR.
    Рядки = топології, стовпці = N.
    Колір: 1 (зелений) → 4 (червоний).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, (metric_key, title) in enumerate([
        ("rho_ranks", "Рейтинг за ρ_med"),
        ("cvar_ranks", "Рейтинг за ρ_CVaR"),
    ]):
        ax = axes[ax_idx]
        ranks_matrix = np.array([
            [ranking[metric_key][N][i] for N in N_VALUES]
            for i in range(len(TOPO_ORDER))
        ], dtype=float)

        im = ax.imshow(ranks_matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=1, vmax=len(TOPO_ORDER))

        ax.set_xticks(range(len(N_VALUES)))
        ax.set_xticklabels([f"N={N}" for N in N_VALUES], fontsize=10)
        ax.set_yticks(range(len(TOPO_ORDER)))
        ax.set_yticklabels(TOPO_ORDER, fontsize=10)

        for i in range(len(TOPO_ORDER)):
            for j in range(len(N_VALUES)):
                val = ranks_matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=9)
                else:
                    color = "white" if val in (1, 4) else "black"
                    ax.text(j, i, f"#{int(val)}", ha="center", va="center",
                            fontsize=12, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Рейтинг (1=кращий)")
        ax.set_title(title, fontsize=11)

    fig.suptitle("S5: Стабільність рейтингу топологій при зміні N", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ranking stability збережено: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Завантаження даних S5...")
    df = load_data()
    print(f"  Рядків: {len(df)}, valid: {df['valid'].sum()}")

    print("\nОписова статистика...")
    desc = descriptive_stats(df)

    print("\nАналіз рейтингів (H6)...")
    ranking = ranking_analysis(desc)

    print("\nТренди ρ з N...")
    trends = scale_trend(desc)

    print("\nЗапис звіту...")
    write_report(desc, ranking, trends, AGG_DIR / "S5_hypothesis_test.txt")

    print("\nГенерація рисунків...")
    plot_rho_vs_N(desc, FIG_DIR / "S5_rho_vs_N.png")
    plot_cvar_vs_N(desc, FIG_DIR / "S5_cvar_vs_N.png")
    plot_ranking_stability(desc, ranking, FIG_DIR / "S5_ranking_stability.png")

    print("\nГотово!")


if __name__ == "__main__":
    main()

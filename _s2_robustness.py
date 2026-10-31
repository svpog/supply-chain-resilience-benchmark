"""
Додатковий аналіз для §5.2: чи діє резервування на МИТТЄВУ робастність R_rb.

Мотивація. Статистичний додаток (блок B) показав, що за інтегральною rho
жодна пара рівнів K_res не дає значущої різниці (0 з 6 після Бонферроні,
Краскел-Волліс p = 0.067). Патч v1.4 стверджує, що ефект резервування
існує, але живе у глибині просідання R_rb, а не в rho, бо при mu=1 і
T=100 мережа встигає повністю відновитись і rho «розводнюється».

Тут це твердження перевіряється тим самим апаратом, що й усе інше:
Краскел-Волліс, попарні Манна-Вітні з Бонферроні, Cliff's delta,
bootstrap-CI різниці медіан. Додатково рахується відношення
|D rho_CVaR| / |D rho_med| для перевірки відносної тези §5.2.

Пише: results/aggregated/S2_robustness_analysis.txt
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from src.metrics.stochastic import compute_rho_CVaR

RAW = Path(__file__).parent / "results" / "raw" / "S2" / "raw_results.csv"
OUT = Path(__file__).parent / "results" / "aggregated" / "S2_robustness_analysis.txt"
B = 10000
SEED = 20261031
ALPHA_CVAR = 0.10


def cliff_delta(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def cliff_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33: return "small"
    if ad < 0.474: return "medium"
    return "large"


def boot_ci_diff(a, b, stat, rng, B=B):
    a = np.asarray(a); b = np.asarray(b)
    out = np.empty(B)
    for i in range(B):
        out[i] = stat(rng.choice(b, len(b), replace=True)) - stat(rng.choice(a, len(a), replace=True))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    df = pd.read_csv(RAW)
    # K_res — параметр ЛИШЕ резервованої топології; у решти трьох він інертний,
    # тому серія S2 у статті аналізується на topology == "reserved".
    df = df[df["topology"] == "reserved"]
    valid = df[df["valid"].astype(str).str.lower().isin(["true", "1"])].copy()
    levels = sorted(valid["K_res"].unique())
    rng = np.random.default_rng(SEED)
    L: list[str] = []
    w = L.append

    w("=" * 100)
    w("СЕРІЯ S2: ДЕ САМЕ ЖИВЕ ЕФЕКТ РЕЗЕРВУВАННЯ")
    w("Додатковий аналіз миттєвої робастності R_rb для §5.2")
    w("=" * 100)
    w("")
    w(f"Дані: {RAW.name}, валідних прогонів {len(valid)} на {len(levels)} рівнів K_res (топологія reserved).")
    w(f"Bootstrap: перцентильний, B={B}, seed={SEED}. CVaR: alpha={ALPHA_CVAR}.")
    w("")

    # ── описова ─────────────────────────────────────────────────────────
    w("-" * 100)
    w("1. ОПИСОВА СТАТИСТИКА ЗА РІВНЯМИ РЕЗЕРВУВАННЯ")
    w("-" * 100)
    w(f"{'K_res':>6}{'n':>6}{'rho_med':>11}{'rho_CVaR':>11}{'rho_IQR':>10}"
      f"{'R_rb_med':>11}{'R_rb_mean':>11}{'R_rb_CVaR':>11}")
    w("-" * 100)
    desc = {}
    for k in levels:
        g = valid[valid["K_res"] == k]
        rho = g["rho"].dropna().values
        rrb = g["R_rb"].dropna().values
        d = dict(n=len(g), rho=rho, rrb=rrb,
                 rho_med=float(np.median(rho)),
                 rho_cvar=float(compute_rho_CVaR(rho.tolist(), alpha=ALPHA_CVAR)),
                 rho_iqr=float(np.percentile(rho, 75) - np.percentile(rho, 25)),
                 rrb_med=float(np.median(rrb)), rrb_mean=float(np.mean(rrb)),
                 rrb_cvar=float(compute_rho_CVaR(rrb.tolist(), alpha=ALPHA_CVAR)))
        desc[k] = d
        w(f"{int(k):>6}{d['n']:>6}{d['rho_med']:>11.4f}{d['rho_cvar']:>11.4f}"
          f"{d['rho_iqr']:>10.4f}{d['rrb_med']:>11.4f}{d['rrb_mean']:>11.4f}{d['rrb_cvar']:>11.4f}")
    w("")

    # ── глобальні тести ─────────────────────────────────────────────────
    w("-" * 100)
    w("2. ГЛОБАЛЬНІ ТЕСТИ КРАСКЕЛА–ВОЛЛІСА")
    w("-" * 100)
    for metric, key in [("rho", "rho"), ("R_rb", "rrb")]:
        samples = [desc[k][key] for k in levels]
        H, p = kruskal(*samples)
        verdict = "H0 відхиляється" if p < 0.05 else "H0 НЕ відхиляється"
        w(f"  {metric:<6}: H = {H:8.4f}, df = {len(levels)-1}, p = {p:.4e}  ->  {verdict}")
    w("")

    # ── попарні ─────────────────────────────────────────────────────────
    pairs = list(combinations(levels, 2))
    m = len(pairs)
    w("-" * 100)
    w(f"3. ПОПАРНІ ТЕСТИ МАННА–ВІТНІ ДЛЯ R_rb (поріг Бонферроні alpha = 0.05/{m} = {0.05/m:.5f})")
    w("-" * 100)
    w(f"{'Пара':<14}{'n1':>5}{'n2':>5}{'p_raw':>11}{'p_adj':>11}{'sig':>5}"
      f"{'Cliff d':>10}{'ефект':>12}{'D R_rb_med':>12}{'95% CI':>22}")
    w("-" * 100)
    n_sig = 0
    for a, b in pairs:
        x, y = desc[a]["rrb"], desc[b]["rrb"]
        _, p_raw = mannwhitneyu(x, y, alternative="two-sided")
        p_adj = min(1.0, m * p_raw)
        d = cliff_delta(x, y)
        dm = float(np.median(y) - np.median(x))
        lo, hi = boot_ci_diff(x, y, np.median, rng)
        sig = "***" if p_adj < 0.05 else ""
        n_sig += 1 if p_adj < 0.05 else 0
        w(f"{'K=%d vs K=%d' % (a, b):<14}{len(x):>5}{len(y):>5}{p_raw:>11.4f}{p_adj:>11.4f}{sig:>5}"
          f"{d:>+10.4f}{cliff_label(d):>12}{dm:>+12.4f}   [{lo:+.4f},{hi:+.4f}]")
    w("")
    w(f"  Значущих після Бонферроні: {n_sig} з {m}.")
    w("")

    # ── відносна теза §5.2 ──────────────────────────────────────────────
    w("-" * 100)
    w("4. ВІДНОСНА ТЕЗА §5.2: ЧИ СИЛЬНІШЕ РЕЗЕРВУВАННЯ ДІЄ НА ХВІСТ, НІЖ НА МЕДІАНУ")
    w("-" * 100)
    w(f"{'Пара':<14}{'D rho_med':>12}{'D rho_CVaR':>13}{'відношення':>13}")
    w("-" * 100)
    ratios = []
    for a, b in pairs:
        dmed = desc[b]["rho_med"] - desc[a]["rho_med"]
        dcv = desc[b]["rho_cvar"] - desc[a]["rho_cvar"]
        r = abs(dcv) / abs(dmed) if abs(dmed) > 1e-9 else float("inf")
        if np.isfinite(r):
            ratios.append(r)
        w(f"{'K=%d vs K=%d' % (a, b):<14}{dmed:>+12.4f}{dcv:>+13.4f}"
          f"{(('%.1fx' % r) if np.isfinite(r) else 'н/д'):>13}")
    w("")
    if ratios:
        w(f"  Відношення |D rho_CVaR|/|D rho_med| на скінченних парах: "
          f"від {min(ratios):.1f}x до {max(ratios):.1f}x, медіана {np.median(ratios):.1f}x.")
    w("")

    # ── контраст R_rb проти rho на краях ────────────────────────────────
    w("-" * 100)
    w("5. КОНТРАСТ K=0 ПРОТИ K=3")
    w("-" * 100)
    a, b = levels[0], levels[-1]
    for name, key, stat in [("rho_med", "rho", np.median),
                            ("R_rb_med", "rrb", np.median),
                            ("R_rb_mean", "rrb", np.mean)]:
        x, y = desc[a][key], desc[b][key]
        diff = float(stat(y) - stat(x))
        lo, hi = boot_ci_diff(x, y, stat, rng)
        crosses = "перетинає нуль" if lo <= 0 <= hi else "НЕ перетинає нуль"
        w(f"  D {name:<10} = {diff:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {crosses}")
    w("")
    w("=" * 100)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nЗаписано: {OUT}")


if __name__ == "__main__":
    main()

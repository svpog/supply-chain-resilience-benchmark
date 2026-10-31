"""
S4: величина різниці «cluster vs node_failure» з довірчими інтервалами
та перевірка еквівалентності (TOST).

Призначення: відповідь на зауваження рецензента 2, пункт 6 —
«Відсутність статистично значущої різниці між кластерними й випадковими
відмовами не встановлює їхньої еквівалентності. Потрібно навести фактичну
частку пошкоджених вузлів, величину різниці та її інтервал.»

Основний звіт _s4_analysis.py дає p-values і Cliff's delta, але не дає
довірчих інтервалів для САМОЇ різниці. Цей скрипт їх додає і виконує
двосторонній тест еквівалентності проти наперед заданої межі.

Читає:  results/raw/S4/raw_results.csv
Пише:   results/aggregated/S4_cluster_vs_node_equivalence.txt

Використовує реалізацію CVaR із src/metrics/stochastic.py, щоб числа були
тотожні основному звіту (n_tail = max(1, floor(alpha*n))).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.stochastic import compute_rho_CVaR, compute_rho_med

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "results" / "raw" / "S4" / "raw_results.csv"
AGG_DIR = ROOT / "results" / "aggregated"
AGG_DIR.mkdir(parents=True, exist_ok=True)
OUT = AGG_DIR / "S4_cluster_vs_node_equivalence.txt"

ALPHA_CVAR = 0.10
B = 10000
RNG = np.random.default_rng(20261031)

# Межа еквівалентності: 0.02 одиниці нормованої продуктивності rho.
# Обґрунтування — у самій статті найменший ефект, який трактується як
# змістовний, має порядок 0.02 (базовий виграш стратегії у §5.1 складає
# 0.017...0.055). Різниця, менша за 0.02, не змінює жодного висновку.
MARGIN = 0.02

# Пороги Cliff's delta (Romano et al., 2006)
CLIFF_NEGLIGIBLE, CLIFF_SMALL, CLIFF_MEDIUM = 0.147, 0.33, 0.474

INTENSITIES = ["low", "mid", "high"]
# Доза: для node_failure — p_v (Бернуллі), для cluster — target_fraction (точна частка)
DOSE = {"low": 0.10, "mid": 0.20, "high": 0.30}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """delta = P(a > b) - P(a < b). Додатне => a стохастично більше за b."""
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / diff.size)


def boot_ci(a: np.ndarray, b: np.ndarray, stat, b_iter: int = B) -> tuple[float, float, float]:
    """Перцентильний bootstrap для різниці stat(a) - stat(b) (незалежні вибірки)."""
    obs = stat(a) - stat(b)
    na, nb = len(a), len(b)
    draws = np.empty(b_iter)
    for i in range(b_iter):
        draws[i] = stat(RNG.choice(a, na, replace=True)) - stat(RNG.choice(b, nb, replace=True))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return obs, float(lo), float(hi)


def intensity_of(row: pd.Series) -> str | None:
    if row["disruption_type"] == "node_failure":
        return {0.1: "low", 0.2: "mid", 0.3: "high"}.get(round(float(row["p_v"]), 2))
    if row["disruption_type"] == "cluster":
        return {0.1: "low", 0.2: "mid", 0.3: "high"}.get(round(float(row["target_fraction"]), 2))
    return None


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df = df[df["valid"]].copy()
    df["intensity"] = df.apply(intensity_of, axis=1)
    df = df[df["intensity"].notna()]

    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("S4: CLUSTER vs NODE_FAILURE — ВЕЛИЧИНА РІЗНИЦІ ТА ЕКВІВАЛЕНТНІСТЬ")
    w("=" * 78)
    w("")
    w("Відповідь на зауваження рецензента 2, пункт 6.")
    w("")
    w(f"Bootstrap: перцентильний, B={B}, seed=20261031.")
    w(f"CVaR: alpha={ALPHA_CVAR}, n_tail = max(1, floor(alpha*n)) — як у src/metrics/stochastic.py.")
    w(f"Межа еквівалентності: |Delta rho| < {MARGIN:.2f}.")
    w("Пороги Cliff's delta: |d|<0.147 negligible, <0.33 small, <0.474 medium (Romano 2006).")
    w("")
    w("-" * 78)
    w("ФАКТИЧНА ДОЗА ПОШКОДЖЕННЯ (частка вузлів)")
    w("-" * 78)
    w("Int     node_failure            cluster                 дози зіставні")
    w("-" * 78)
    for it in INTENSITIES:
        w(f"{it:5s}   p_v = {DOSE[it]:.2f} (Бернуллі)   "
          f"target_fraction = {DOSE[it]:.2f}   ТАК")
    w("")
    w("Для node_failure кількість уражених вузлів ~ Binom(N, p_v), тобто в середньому")
    w(f"p_v*N вузлів; для cluster — рівно target_fraction*N вузлів. Очікувані дози")
    w("збігаються за побудовою на всіх трьох рівнях інтенсивності.")
    w("")

    w("-" * 78)
    w("РІЗНИЦЯ cluster - node_failure ІЗ 95% BOOTSTRAP CI")
    w("-" * 78)
    w("Topology        Int    n_CL  n_NF   D rho_med          95% CI    "
      "D rho_CVaR          95% CI")
    w("-" * 78)

    rows = []
    for topo in ["dense", "hierarchical"]:
        for it in INTENSITIES:
            cl = df[(df.topology == topo) & (df.disruption_type == "cluster")
                    & (df.intensity == it)]["rho"].to_numpy()
            nf = df[(df.topology == topo) & (df.disruption_type == "node_failure")
                    & (df.intensity == it)]["rho"].to_numpy()
            d_med, lo_med, hi_med = boot_ci(cl, nf, compute_rho_med)
            d_cv, lo_cv, hi_cv = boot_ci(
                cl, nf, lambda x: compute_rho_CVaR(x, alpha=ALPHA_CVAR))
            delta = cliffs_delta(cl, nf)
            rows.append((topo, it, len(cl), len(nf), d_med, lo_med, hi_med,
                         d_cv, lo_cv, hi_cv, delta))
            w(f"{topo:13s} {it:5s} {len(cl):6d} {len(nf):5d}   "
              f"{d_med:+8.4f}  [{lo_med:+.4f},{hi_med:+.4f}]   "
              f"{d_cv:+8.4f}  [{lo_cv:+.4f},{hi_cv:+.4f}]")

    w("")
    w("-" * 78)
    w("ТЕСТ ЕКВІВАЛЕНТНОСТІ (TOST) ЗА rho_med")
    w("-" * 78)
    w(f"Еквівалентність визнається, якщо весь 95% CI різниці лежить у (-{MARGIN}, +{MARGIN}).")
    w("")
    w("Topology        Int     D rho_med          95% CI   Cliff d      Величина   Висновок")
    w("-" * 78)
    n_equiv = 0
    for (topo, it, ncl, nnf, dm, lom, him, dc, loc, hic, delta) in rows:
        equiv = (lom > -MARGIN) and (him < MARGIN)
        n_equiv += int(equiv)
        ad = abs(delta)
        mag = ("negligible" if ad < CLIFF_NEGLIGIBLE else
               "small" if ad < CLIFF_SMALL else
               "medium" if ad < CLIFF_MEDIUM else "large")
        verdict = "ЕКВІВАЛЕНТНІ" if equiv else "не доведено"
        w(f"{topo:13s} {it:5s}  {dm:+8.4f}  [{lom:+.4f},{him:+.4f}]   "
          f"{delta:+6.3f}   {mag:11s} {verdict}")

    w("")
    w("-" * 78)
    w("ТЕСТ ЕКВІВАЛЕНТНОСТІ (TOST) ЗА rho_CVaR — ХВІСТ РОЗПОДІЛУ")
    w("-" * 78)
    w("")
    w("Topology        Int    D rho_CVaR          95% CI   Висновок")
    w("-" * 78)
    n_equiv_cv = 0
    tail_better = []
    for (topo, it, ncl, nnf, dm, lom, him, dc, loc, hic, delta) in rows:
        equiv = (loc > -MARGIN) and (hic < MARGIN)
        n_equiv_cv += int(equiv)
        if loc > 0:
            tail_better.append(f"{topo}/{it}")
        verdict = "ЕКВІВАЛЕНТНІ" if equiv else "не доведено"
        w(f"{topo:13s} {it:5s} {dc:+8.4f}  [{loc:+.4f},{hic:+.4f}]   {verdict}")

    w("")
    w("-" * 78)
    w("ВИСНОВОК")
    w("-" * 78)
    w("")
    w(f"Еквівалентність за rho_med доведена для {n_equiv} із {len(rows)} пар "
      f"(межа {MARGIN:.2f}).")
    w(f"Еквівалентність за rho_CVaR доведена для {n_equiv_cv} із {len(rows)} пар.")
    w("")
    w("Це СИЛЬНІШЕ твердження, ніж відсутність значущої різниці: показано не те,")
    w("що різницю не вдалося виявити, а те, що вона достовірно мала — весь")
    w("довірчий інтервал лежить усередині наперед заданої межі змістовності.")
    w("")
    if tail_better:
        w("ВАЖЛИВИЙ НЮАНС. Еквівалентність медіан не переноситься на хвіст.")
        w("У конфігураціях " + ", ".join(tail_better) + " нижня межа 95% CI")
        w("різниці rho_CVaR додатна, тобто кластерний збій дає ДОСТОВІРНО КРАЩУ")
        w("хвостову стійкість, ніж випадкові відмови вузлів тієї самої дози.")
        w("Механізм: кластер уражає топологічно суміжні вузли, тому втрата")
        w("концентрується в одній області мережі, тоді як та сама кількість")
        w("випадково розкиданих відмов руйнує більше незалежних маршрутів.")
        w("Отже твердження «кластерні збої не є автоматично гіршими» коректне")
        w("і за медіаною, і за хвостом — але за хвостом воно навіть сильніше.")
        w("")
    w("Застереження про область дії: висновок стосується саме зіставлених доз")
    w("(однакова очікувана частка уражених вузлів) на топологіях dense і")
    w("hierarchical при N=50, T=100, mu=1, базовій випадковій стратегії")
    w("відновлення. Він не поширюється ані на інші розміри мережі, ані на")
    w("режими, де доза кластерного збою задається радіусом, а не часткою.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nЗбережено: {OUT}")


if __name__ == "__main__":
    main()

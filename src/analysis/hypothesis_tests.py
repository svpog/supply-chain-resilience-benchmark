"""
Статистичне тестування гіпотез resilience (Math Spec, розділ 8).

Реалізує:
    H1: тип топології статистично значимо впливає на ρ
        (Kruskal-Wallis + pairwise Mann-Whitney U + Cliff's delta)

Приклад використання:
    >>> from src.analysis.hypothesis_tests import run_H1_analysis
    >>> import pandas as pd
    >>> raw_df = pd.read_csv("results/raw/S1/raw_results.csv")
    >>> report = run_H1_analysis(raw_df)
    >>> print(report)
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════════════════
# Cliff's delta
# ════════════════════════════════════════════════════════════════════════════

def cliffs_delta(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """
    Cliff's delta — непараметрична оцінка розміру ефекту.

    δ = (# пар де x > y  −  # пар де x < y) / (n_x · n_y)
    δ ∈ [−1, 1].

    Інтерпретація |δ| (Romano et al., 2006):
        < 0.147  — мізерний (negligible)
        0.147–0.330 — малий (small)
        0.330–0.474 — середній (medium)
        ≥ 0.474  — великий (large)

    Args:
        x: перший масив значень.
        y: другий масив значень.

    Returns:
        Cliff's delta ∈ [−1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Векторизований підрахунок через broadcasting (швидше ніж подвійний цикл)
    diff = x[:, None] - y[None, :]  # матриця (n_x, n_y)
    dominance = float(np.sum(np.sign(diff)))
    return dominance / (len(x) * len(y))


def interpret_cliffs_delta(d: float) -> str:
    """Текстова інтерпретація |Cliff's delta|."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.330:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


# ════════════════════════════════════════════════════════════════════════════
# Kruskal-Wallis
# ════════════════════════════════════════════════════════════════════════════

def kruskal_wallis_test(groups: dict[str, list[float]]) -> dict[str, Any]:
    """
    Kruskal-Wallis H-тест: перевіряє, чи хоча б одна група відрізняється
    від решти за медіаною (непараметричний аналог ANOVA).

    Нульова гіпотеза H0: всі групи мають однаковий розподіл ρ.
    Альтернатива H1: принаймні одна група відрізняється.

    Args:
        groups: словник {назва_групи: список ρ}.

    Returns:
        dict з ключами: H_statistic, p_value, df, reject_H0, group_sizes.
    """
    arrays = [np.array(v, dtype=float) for v in groups.values()]
    H, p = stats.kruskal(*arrays)
    df = len(groups) - 1
    return {
        "H_statistic": float(H),
        "p_value": float(p),
        "df": df,
        "reject_H0": bool(p < 0.05),
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }


# ════════════════════════════════════════════════════════════════════════════
# Pairwise Mann-Whitney U + Bonferroni
# ════════════════════════════════════════════════════════════════════════════

def mannwhitney_pairwise(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Попарний Mann-Whitney U-тест з корекцією Bonferroni.

    Для k груп генерує C(k,2) порівнянь. Скоригований рівень значущості:
        α_adj = alpha / C(k,2)

    Args:
        groups: словник {назва_групи: список ρ}.
        alpha: вихідний рівень значущості (дефолт: 0.05).

    Returns:
        DataFrame з колонками:
            group_A, group_B, n_A, n_B,
            U_statistic, p_value, p_value_adj, significant,
            cliffs_delta, effect_size.
    """
    names = list(groups.keys())
    pairs = list(itertools.combinations(names, 2))
    n_comparisons = len(pairs)
    alpha_adj = alpha / n_comparisons

    rows = []
    for a, b in pairs:
        xa = np.array(groups[a], dtype=float)
        xb = np.array(groups[b], dtype=float)

        U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        p_adj = min(1.0, p * n_comparisons)  # Bonferroni correction
        cd = cliffs_delta(xa, xb)

        rows.append({
            "group_A": a,
            "group_B": b,
            "n_A": len(xa),
            "n_B": len(xb),
            "U_statistic": float(U),
            "p_value": float(p),
            "p_value_adj": float(p_adj),
            "significant": bool(p_adj < alpha),
            "cliffs_delta": float(cd),
            "effect_size": interpret_cliffs_delta(cd),
        })

    df = pd.DataFrame(rows)
    df.attrs["alpha"] = alpha
    df.attrs["alpha_adj"] = alpha_adj
    df.attrs["n_comparisons"] = n_comparisons
    return df


# ════════════════════════════════════════════════════════════════════════════
# Форматування звіту
# ════════════════════════════════════════════════════════════════════════════

def _fmt_p(p: float) -> str:
    """Форматує p-value: < 0.001 або точне значення."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def format_H1_report(
    kw_result: dict[str, Any],
    pw_df: pd.DataFrame,
    groups: dict[str, list[float]],
    series: str = "S1",
    group_col: str = "topology",
    alpha: float = 0.05,
) -> str:
    """
    Форматує повний текстовий звіт тестування гіпотези H1.

    Args:
        kw_result: результат kruskal_wallis_test.
        pw_df: результат mannwhitney_pairwise.
        groups: вихідні дані {назва: масив ρ}.
        series: назва серії (для заголовка).
        group_col: назва колонки групування.
        alpha: вихідний рівень значущості.

    Returns:
        Форматований текст звіту.
    """
    n_comparisons = pw_df.attrs.get("n_comparisons", len(pw_df))
    alpha_adj = alpha / n_comparisons
    lines = []

    # ── Заголовок ──────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append(f"HYPOTHESIS TEST H1 — Series {series}")
    lines.append(f"Гіпотеза: тип {group_col} статистично значимо впливає на ρ")
    lines.append(f"Тест: Kruskal-Wallis + pairwise Mann-Whitney U (Bonferroni)")
    lines.append(f"α = {alpha},  α_adj (Bonferroni) = {alpha:.4f}/{n_comparisons} = {alpha_adj:.4f}")
    lines.append("=" * 72)
    lines.append("")

    # ── Описова статистика ─────────────────────────────────────────────────
    lines.append("ОПИСОВА СТАТИСТИКА ρ:")
    lines.append(f"{'Топологія':<15}  {'N':>5}  {'Median':>8}  {'Mean':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    lines.append("-" * 65)
    for name, vals in sorted(groups.items()):
        arr = np.array(vals)
        lines.append(
            f"{name:<15}  {len(arr):>5}  {np.median(arr):>8.4f}  "
            f"{arr.mean():>8.4f}  {arr.std():>7.4f}  "
            f"{arr.min():>7.4f}  {arr.max():>7.4f}"
        )
    lines.append("")

    # ── Kruskal-Wallis ────────────────────────────────────────────────────
    lines.append("KRUSKAL-WALLIS TEST:")
    lines.append(f"  H-statistic = {kw_result['H_statistic']:.4f}")
    lines.append(f"  p-value     = {_fmt_p(kw_result['p_value'])}")
    lines.append(f"  df          = {kw_result['df']}")
    verdict = "ВІДХИЛЯЄМО H0" if kw_result["reject_H0"] else "НЕ ВІДХИЛЯЄМО H0"
    lines.append(f"  Висновок: {verdict} (α = {alpha})")
    lines.append("")

    # ── Pairwise Mann-Whitney + Cliff's delta ─────────────────────────────
    lines.append("PAIRWISE MANN-WHITNEY U (Bonferroni-скоригований):")
    hdr = (
        f"{'A':>12}  {'B':>12}  {'U':>9}  {'p-raw':>8}  "
        f"{'p-adj':>8}  {'Sig':>4}  {'δ':>7}  {'Effect':<12}"
    )
    lines.append(hdr)
    lines.append("-" * 80)
    for _, row in pw_df.iterrows():
        sig = "**" if row["significant"] else "  "
        lines.append(
            f"{row['group_A']:>12}  {row['group_B']:>12}  "
            f"{row['U_statistic']:>9.0f}  {_fmt_p(row['p_value']):>8}  "
            f"{_fmt_p(row['p_value_adj']):>8}  {sig:>4}  "
            f"{row['cliffs_delta']:>7.4f}  {row['effect_size']:<12}"
        )
    lines.append("  ** = significant after Bonferroni correction")
    lines.append("")

    # ── Загальний висновок ────────────────────────────────────────────────
    lines.append("ВИСНОВОК:")
    if kw_result["reject_H0"]:
        sig_pairs = pw_df[pw_df["significant"]]
        n_sig = len(sig_pairs)
        lines.append(
            f"  H1 ПІДТВЕРДЖЕНО: Kruskal-Wallis показав статистично значиму "
            f"різницю між топологіями (H={kw_result['H_statistic']:.2f}, "
            f"p{_fmt_p(kw_result['p_value'])})."
        )
        if n_sig > 0:
            large_eff = pw_df[pw_df["effect_size"].isin(["large", "medium"])]
            lines.append(
                f"  Після Bonferroni-корекції: {n_sig}/{n_comparisons} пар значимі."
            )
            if len(large_eff) > 0:
                pairs_str = ", ".join(
                    f"{r['group_A']} vs {r['group_B']} (δ={r['cliffs_delta']:.3f}, {r['effect_size']})"
                    for _, r in large_eff.iterrows()
                )
                lines.append(f"  Середній/великий ефект: {pairs_str}.")
        else:
            lines.append(
                f"  Після Bonferroni-корекції: жодна пара не значима — "
                f"різниці між топологіями малі за розміром ефекту."
            )
    else:
        lines.append(
            f"  H1 НЕ ПІДТВЕРДЖЕНО: Kruskal-Wallis не показав статистично "
            f"значимої різниці між топологіями (p={_fmt_p(kw_result['p_value'])})."
        )
    lines.append("=" * 72)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Головна функція
# ════════════════════════════════════════════════════════════════════════════

def run_H1_analysis(
    raw_df: pd.DataFrame,
    group_col: str = "topology",
    series: str = "S1",
    alpha: float = 0.05,
    output_path: str | None = None,
) -> str:
    """
    Повний аналіз гіпотези H1: Kruskal-Wallis + pairwise Mann-Whitney +
    Cliff's delta.

    Використовує лише valid прогони (valid=True).

    Args:
        raw_df: raw_results.csv як DataFrame.
        group_col: колонка для групування (дефолт: "topology").
        series: назва серії для звіту.
        alpha: рівень значущості.
        output_path: якщо задано — зберігає звіт у файл.

    Returns:
        Відформатований текст звіту.
    """
    valid = raw_df[raw_df["valid"].astype(bool)].copy()
    groups: dict[str, list[float]] = {}
    for name, grp in valid.groupby(group_col):
        vals = grp["rho"].dropna().tolist()
        if vals:
            groups[str(name)] = vals

    if len(groups) < 2:
        raise ValueError(f"Потрібно ≥ 2 групи, знайдено: {len(groups)}")

    kw_result = kruskal_wallis_test(groups)
    pw_df = mannwhitney_pairwise(groups, alpha=alpha)
    report = format_H1_report(kw_result, pw_df, groups,
                               series=series, group_col=group_col, alpha=alpha)

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")

    return report

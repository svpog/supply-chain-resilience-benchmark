"""
CVaR-діаграма: розподіл ρ з виділеною α-зоною.

Реалізує: Plan таблиць і рисунків (Етап 6)
(див. docs/протоколи_серій_та_метрики_перша_стаття_v3.xlsx).

Діаграма показує:
    - гістограму розподілу ρ по M прогонах;
    - вертикальну лінію VaR_α (порогове значення);
    - зафарбовану α-зону (CVaR: очікуване в хвості);
    - вертикальну лінію медіани для порівняння.

Приклад використання:
    >>> from src.visualization.cvar_plots import plot_cvar_diagram
    >>> rho = [0.7, 0.8, 0.6, 0.9, 0.5, ...]
    >>> fig = plot_cvar_diagram(rho, alpha=0.10,
    ...                         output_path="results/figures/cvar_S1_hierarchical.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_cvar_diagram(
    rho_values: list[float] | np.ndarray,
    alpha: float = 0.10,
    label: str = "",
    output_path: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
    save_dpi: int = 150,
    bins: int = 30,
    color: str = "steelblue",
) -> plt.Figure:
    """
    Гістограма розподілу ρ з виділеною CVaR-зоною.

    Авторська метрика ρ_CVaR_α відображається як середнє нижніх α·100% значень:
        - VaR_α  = percentile(ρ, α·100)  [вертикальна лінія]
        - CVaR_α = mean(ρ | ρ ≤ VaR_α)  [середнє у хвості]
    Разом із ρ_med дозволяє оцінити не лише «типову» поведінку,
    але й «ризик найгіршого сценарію».

    Args:
        rho_values: список/масив значень ρ зі M прогонів.
        alpha: рівень CVaR ∈ (0, 1]. Дефолт: 0.10 (нижні 10%).
        label: мітка для заголовка і легенди.
        output_path: якщо задано — зберігає PNG.
        title: заголовок (автоматичний якщо None).
        figsize: розмір фігури.
        save_dpi: роздільна здатність збереження.
        bins: кількість стовпчиків гістограми.
        color: колір гістограми.

    Returns:
        matplotlib.figure.Figure.
    """
    arr = np.array(rho_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        raise ValueError("rho_values порожній або містить лише NaN.")

    var_alpha = float(np.percentile(arr, alpha * 100))
    tail = arr[arr <= var_alpha]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var_alpha
    rho_med = float(np.median(arr))

    fig, ax = plt.subplots(figsize=figsize)

    # Гістограма: хвіст (≤ VaR_α) окремим кольором
    ax.hist(arr[arr > var_alpha],  bins=bins, color=color,  alpha=0.7,
            label="ρ > VaR_α")
    ax.hist(arr[arr <= var_alpha], bins=max(1, bins // 5), color="tomato", alpha=0.85,
            label=f"CVaR-зона (α={alpha:.0%})")

    # VaR_α
    ax.axvline(var_alpha, color="red", linewidth=1.8, linestyle="--",
               label=f"VaR_{alpha:.0%} = {var_alpha:.3f}")
    # CVaR
    ax.axvline(cvar, color="darkred", linewidth=1.5, linestyle=":",
               label=f"ρ_CVaR = {cvar:.3f}")
    # Медіана
    ax.axvline(rho_med, color="navy", linewidth=1.8, linestyle="-.",
               label=f"ρ_med = {rho_med:.3f}")

    ax.set_xlabel("ρ (інтегральна resilience)", fontsize=12)
    ax.set_ylabel("Кількість прогонів", fontsize=12)
    if title is None:
        title = f"Розподіл ρ" + (f" — {label}" if label else "")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig


def plot_cvar_comparison(
    rho_by_group: dict[str, list[float]],
    alpha: float = 0.10,
    output_path: str | None = None,
    title: str = "Порівняння розподілів ρ та CVaR-зон",
    figsize: tuple[int, int] | None = None,
    save_dpi: int = 150,
) -> plt.Figure:
    """
    Порівняльна CVaR-діаграма для декількох груп (топологій).

    Кожна група отримує окремий підграфік у сітці 2×N або 1×N.
    Дозволяє візуально оцінити, чи відрізняється ρ_CVaR між топологіями
    (основна мета серії S1 / S6).

    Args:
        rho_by_group: словник {назва_групи: список ρ}.
        alpha: рівень CVaR.
        output_path: шлях збереження PNG.
        title: загальний заголовок.
        figsize: розмір фігури; автоматично якщо None.
        save_dpi: роздільна здатність.

    Returns:
        matplotlib.figure.Figure.
    """
    groups = list(rho_by_group.keys())
    n = len(groups)
    if n == 0:
        raise ValueError("rho_by_group порожній.")

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 4 * nrows)

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        color = colors[idx % len(colors)]
        arr = np.array(rho_by_group[group], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            ax.set_title(f"{group}\n(немає даних)")
            ax.axis("off")
            continue

        var_alpha = float(np.percentile(arr, alpha * 100))
        cvar      = float(np.mean(arr[arr <= var_alpha])) if any(arr <= var_alpha) else var_alpha
        rho_med   = float(np.median(arr))

        ax.hist(arr[arr > var_alpha],  bins=25, color=color,  alpha=0.65)
        ax.hist(arr[arr <= var_alpha], bins=max(1, 5), color="tomato", alpha=0.85)

        ax.axvline(var_alpha, color="red",  linewidth=1.5, linestyle="--",
                   label=f"VaR = {var_alpha:.3f}")
        ax.axvline(cvar,      color="darkred", linewidth=1.4, linestyle=":",
                   label=f"CVaR = {cvar:.3f}")
        ax.axvline(rho_med,   color="navy", linewidth=1.5, linestyle="-.",
                   label=f"med = {rho_med:.3f}")

        ax.set_title(str(group), fontsize=11)
        ax.set_xlabel("ρ", fontsize=9)
        ax.set_ylabel("# прогонів", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Приховуємо порожні підграфіки
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig

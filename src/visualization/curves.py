"""
Криві Q(t) з median та IQR shading.

Реалізує: рисунок 2 з протоколів серій (Етап 6)
(див. docs/протоколи_серій_та_метрики_перша_стаття_v3.xlsx).

Вхід: Q_curves.parquet (run_id, config_id, t, Q) +
      raw_results.csv  (run_id, topology, valid, …).

Приклад використання:
    >>> from src.visualization.curves import plot_Q_curves
    >>> import pandas as pd
    >>> q_df  = pd.read_parquet("results/raw/S1/Q_curves.parquet")
    >>> raw_df = pd.read_csv("results/raw/S1/raw_results.csv")
    >>> fig = plot_Q_curves(q_df, raw_df,
    ...                     output_path="results/figures/Q_curves_S1.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_Q_curves(
    q_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    group_by: str = "topology",
    output_path: str | None = None,
    title: str = "Q(t) — медіана та IQR за топологіями",
    figsize: tuple[int, int] = (10, 6),
    save_dpi: int = 150,
    show_target: bool = True,
    q_target: float = 0.95,
) -> plt.Figure:
    """
    Будує криві Q(t): median + IQR shading для кожної групи (topology).

    Медіана та міжквартильний розмах обчислюються по всіх valid прогонах
    у кожному config_id. Дозволяє порівняти форму кривих відновлення
    для різних топологій (або інших параметрів).

    Args:
        q_df: DataFrame з Q_curves.parquet (колонки: run_id, t, Q).
        raw_df: DataFrame з raw_results.csv (колонки: run_id, topology, valid).
        group_by: колонка raw_df для розфарбовування ліній.
        output_path: якщо задано — зберігає PNG за цим шляхом.
        title: заголовок рисунку.
        figsize: розмір фігури (width, height) у дюймах.
        save_dpi: роздільна здатність для збереження.
        show_target: чи показувати горизонтальну лінію Q_target.
        q_target: поріг відновлення (дефолт 0.95).

    Returns:
        matplotlib.figure.Figure.
    """
    # Беремо run_id лише valid прогонів
    valid_ids = raw_df[raw_df["valid"].astype(bool)][["run_id", group_by]].drop_duplicates()

    merged = q_df.merge(valid_ids, on="run_id", how="inner")
    if merged.empty:
        raise ValueError("Немає valid прогонів для побудови кривих.")

    # Статистика (median, Q25, Q75) по групах і кроках часу
    g = merged.groupby([group_by, "t"])["Q"]
    stats = pd.concat([
        g.median().rename("median"),
        g.quantile(0.25).rename("q25"),
        g.quantile(0.75).rename("q75"),
    ], axis=1).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    groups = sorted(stats[group_by].unique())

    for i, group in enumerate(groups):
        color = colors[i % len(colors)]
        gdf = stats[stats[group_by] == group].sort_values("t")
        t   = gdf["t"].values
        med = gdf["median"].values
        q25 = gdf["q25"].values
        q75 = gdf["q75"].values

        ax.plot(t, med, label=str(group), color=color, linewidth=2)
        ax.fill_between(t, q25, q75, alpha=0.18, color=color)

    if show_target:
        ax.axhline(
            y=q_target, color="dimgray", linestyle="--",
            linewidth=1, alpha=0.7,
            label=f"Q_target = {q_target}",
        )

    ax.set_xlabel("Час t (кроки симуляції)", fontsize=12)
    ax.set_ylabel("Q(t) = F(t) / F(0)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig


def plot_Q_curves_grid(
    q_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    group_by: str = "topology",
    output_path: str | None = None,
    figsize: tuple[int, int] | None = None,
    save_dpi: int = 150,
    q_target: float = 0.95,
) -> plt.Figure:
    """
    Сітка підграфіків: один підграфік на групу (topology).

    Зручно для детального порівняння кривих відновлення між серіями.

    Args:
        q_df: DataFrame з Q_curves.parquet.
        raw_df: DataFrame з raw_results.csv.
        group_by: колонка raw_df для поділу по підграфіках.
        output_path: якщо задано — зберігає PNG.
        figsize: розмір фігури; якщо None — обчислюється автоматично.
        save_dpi: роздільна здатність збереження.
        q_target: поріг відновлення.

    Returns:
        matplotlib.figure.Figure.
    """
    valid_ids = raw_df[raw_df["valid"].astype(bool)][["run_id", group_by]].drop_duplicates()
    merged = q_df.merge(valid_ids, on="run_id", how="inner")
    if merged.empty:
        raise ValueError("Немає valid прогонів для побудови кривих.")

    groups = sorted(merged[group_by].unique())
    n = len(groups)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    colors = plt.cm.tab10.colors

    for idx, group in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        color = colors[idx % len(colors)]
        sub = merged[merged[group_by] == group]

        runs = sub.groupby(["run_id", "t"])["Q"].first().unstack("t")
        if runs.empty:
            continue

        t_vals = runs.columns.values
        med = runs.median(axis=0).values
        q25 = runs.quantile(0.25, axis=0).values
        q75 = runs.quantile(0.75, axis=0).values

        ax.plot(t_vals, med, color=color, linewidth=2)
        ax.fill_between(t_vals, q25, q75, alpha=0.2, color=color)
        ax.axhline(y=q_target, color="dimgray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(str(group), fontsize=11)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("Q(t)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    # Приховуємо порожні підграфіки
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Q(t): медіана + IQR за топологіями", fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig

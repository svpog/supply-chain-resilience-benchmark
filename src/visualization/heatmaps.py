"""
Heatmap: тип збою × інтенсивність → ρ_med (серія S4).

Реалізує: Plan таблиць і рисунків (Етап 6)
(див. docs/протоколи_серій_та_метрики_перша_стаття_v3.xlsx).

Тривимірна структура S4:
    - node_failure:     ряд × стовпець = topology × p_v
    - edge_degradation: ряд × стовпець = p_e × gamma (для кожної топології)
    - cluster:          ряд × стовпець = topology × r_clust

Приклад використання:
    >>> from src.visualization.heatmaps import plot_disruption_heatmap
    >>> import pandas as pd
    >>> agg_df = pd.read_csv("results/aggregated/aggregated_results.csv")
    >>> fig = plot_disruption_heatmap(agg_df,
    ...                               output_path="results/figures/heatmap_S4.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_disruption_heatmap(
    agg_df: pd.DataFrame,
    metric: str = "rho_med",
    output_path: str | None = None,
    figsize: tuple[int, int] = (16, 5),
    save_dpi: int = 150,
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """
    Три-панельний heatmap для S4: по одній панелі на тип збою.

    Панель 1 (node_failure):     topology × p_v → ρ_med
    Панель 2 (edge_degradation): p_e × gamma → ρ_med (усереднено по топологіях)
    Панель 3 (cluster):          topology × r_clust → ρ_med

    Args:
        agg_df: aggregated_results.csv з серії S4.
        metric: метрика для відображення (дефолт: rho_med).
        output_path: якщо задано — зберігає PNG.
        figsize: розмір фігури.
        save_dpi: роздільна здатність збереження.
        cmap: colormap (RdYlGn: червоний=погано, зелений=добре).

    Returns:
        matplotlib.figure.Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _plot_node_failure_panel(agg_df, axes[0], metric, cmap)
    _plot_edge_degradation_panel(agg_df, axes[1], metric, cmap)
    _plot_cluster_panel(agg_df, axes[2], metric, cmap)

    fig.suptitle(
        f"S4: тип збою × інтенсивність → {metric}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig


def _plot_node_failure_panel(
    agg_df: pd.DataFrame,
    ax: plt.Axes,
    metric: str,
    cmap: str,
) -> None:
    """Панель node_failure: topology × p_v → ρ_med."""
    sub = agg_df[agg_df["disruption_type"] == "node_failure"].copy()
    if sub.empty:
        ax.set_title("node_failure\n(немає даних)")
        ax.axis("off")
        return

    sub["p_v"] = sub["p_v"].astype(float)
    pivot = sub.pivot_table(
        index="topology", columns="p_v", values=metric, aggfunc="mean"
    )
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("node_failure\ntopology × p_v", fontsize=11)
    ax.set_xlabel("p_v", fontsize=10)
    ax.set_ylabel("topology", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def _plot_edge_degradation_panel(
    agg_df: pd.DataFrame,
    ax: plt.Axes,
    metric: str,
    cmap: str,
) -> None:
    """Панель edge_degradation: p_e × gamma → ρ_med (усереднено по топологіях)."""
    sub = agg_df[agg_df["disruption_type"] == "edge_degradation"].copy()
    if sub.empty:
        ax.set_title("edge_degradation\n(немає даних)")
        ax.axis("off")
        return

    sub["p_e"]   = sub["p_e"].astype(float)
    sub["gamma"] = sub["gamma"].astype(float)
    pivot = sub.pivot_table(
        index="p_e", columns="gamma", values=metric, aggfunc="mean"
    )
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("edge_degradation\np_e × gamma", fontsize=11)
    ax.set_xlabel("gamma", fontsize=10)
    ax.set_ylabel("p_e", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def _plot_cluster_panel(
    agg_df: pd.DataFrame,
    ax: plt.Axes,
    metric: str,
    cmap: str,
) -> None:
    """Панель cluster: topology × r_clust → ρ_med."""
    sub = agg_df[agg_df["disruption_type"] == "cluster"].copy()
    if sub.empty:
        ax.set_title("cluster\n(немає даних)")
        ax.axis("off")
        return

    sub["r_clust"] = sub["r_clust"].astype(float)
    pivot = sub.pivot_table(
        index="topology", columns="r_clust", values=metric, aggfunc="mean"
    )
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("cluster\ntopology × r_clust", fontsize=11)
    ax.set_xlabel("r_clust", fontsize=10)
    ax.set_ylabel("topology", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_topology_comparison_heatmap(
    agg_df: pd.DataFrame,
    x_col: str,
    metric: str = "rho_med",
    disruption_type: str = "node_failure",
    output_path: str | None = None,
    figsize: tuple[int, int] = (8, 4),
    save_dpi: int = 150,
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """
    Простий heatmap: topology × одна вісь інтенсивності → ρ_med.

    Зручний для швидкого порівняння двох топологій при конкретному
    типі збою.

    Args:
        agg_df: aggregated_results.csv.
        x_col: колонка для осі X (p_v, p_e, r_clust тощо).
        metric: метрика (дефолт: rho_med).
        disruption_type: тип збою для фільтрації.
        output_path: шлях збереження PNG.
        figsize: розмір фігури.
        save_dpi: роздільна здатність.
        cmap: colormap.

    Returns:
        matplotlib.figure.Figure.
    """
    sub = agg_df[agg_df["disruption_type"] == disruption_type].copy()
    if sub.empty:
        raise ValueError(f"Немає даних для disruption_type='{disruption_type}'")

    sub[x_col] = sub[x_col].astype(float)
    pivot = sub.pivot_table(
        index="topology", columns=x_col, values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title(f"{disruption_type}: topology × {x_col} → {metric}", fontsize=11)
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel("topology", fontsize=10)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig

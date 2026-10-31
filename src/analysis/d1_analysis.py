"""
D1 node-level diagnostic для слабкого K_res ефекту в R2.

Реалізує переформульовану D1-діагностику: чи recovery стратегії
ремонтують вузли, що розблоковують backup-маршрути, чи переважно
магістральні high-capacity ребра.

Приклад використання:
    python -m src.analysis.d1_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "D1"
AGG_DIR = ROOT / "results" / "aggregated" / "article3"

STRATEGIES = ["random", "centrality_betweenness", "flow_greedy"]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("inf") if numerator > 0 else float("nan")
    return float(numerator / denominator)


def load_actions() -> pd.DataFrame:
    path = RAW_DIR / "recovery_actions_log.parquet"
    if not path.exists():
        raise FileNotFoundError(f"D1 recovery log not found: {path}")
    actions = pd.read_parquet(path)
    if actions.empty:
        raise ValueError("D1 recovery log is empty.")
    actions["has_backup_neighbor"] = actions["has_backup_neighbor"].astype(bool)
    return actions


def baseline_by_seed(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Частка failed/repaired вузлів із backup-neighbor для кожного paired_id.

    Для D1 усі стратегії мають один і той самий failed_nodes набір. Тому
    de-dup по (paired_id, repaired_node_id) відновлює baseline candidate pool.
    """
    dedup = actions.drop_duplicates(["paired_id", "repaired_node_id"]).copy()
    return (
        dedup.groupby("paired_id", as_index=False)
        .agg(
            frac_failed_with_backup_neighbor=("has_backup_neighbor", "mean"),
            n_failed_nodes_logged=("repaired_node_id", "nunique"),
        )
    )


def per_seed_strategy_stats(actions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (strategy, paired_id), group in actions.groupby(["recovery_strategy", "paired_id"]):
        first3 = group[group["t"] <= 3]
        rows.append(
            {
                "recovery_strategy": strategy,
                "paired_id": paired_id,
                "has_backup_share": float(group["has_backup_neighbor"].mean()),
                "first3_share": float(first3["has_backup_neighbor"].mean()) if len(first3) else np.nan,
                "n_actions": int(len(group)),
                "trunk_capacity_sum": float(group["newly_unlocked_trunk_capacity"].sum()),
                "backup_capacity_sum": float(group["newly_unlocked_backup_capacity"].sum()),
            }
        )
    stats = pd.DataFrame(rows)
    return stats.merge(baseline, on="paired_id", how="left")


def aggregate_by_strategy(actions: pd.DataFrame, seed_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in STRATEGIES:
        group = actions[actions["recovery_strategy"] == strategy]
        seed_group = seed_stats[seed_stats["recovery_strategy"] == strategy]
        first3 = group[group["t"] <= 3]
        trunk_total = float(group["newly_unlocked_trunk_capacity"].sum())
        backup_total = float(group["newly_unlocked_backup_capacity"].sum())
        rows.append(
            {
                "recovery_strategy": strategy,
                "has_backup_neighbor_share": float(group["has_backup_neighbor"].mean()),
                "has_backup_neighbor_share_first3": (
                    float(first3["has_backup_neighbor"].mean()) if len(first3) else np.nan
                ),
                "mean_newly_unlocked_trunk_capacity_per_repair": float(
                    group["newly_unlocked_trunk_capacity"].mean()
                ),
                "mean_newly_unlocked_backup_capacity_per_repair": float(
                    group["newly_unlocked_backup_capacity"].mean()
                ),
                "first3_trunk_capacity_per_repair": float(
                    first3["newly_unlocked_trunk_capacity"].mean()
                ) if len(first3) else np.nan,
                "first3_backup_capacity_per_repair": float(
                    first3["newly_unlocked_backup_capacity"].mean()
                ) if len(first3) else np.nan,
                "ratio_trunk_to_backup": _safe_ratio(trunk_total, backup_total),
                "n_actions": int(len(group)),
                "n_seed_pairs": int(seed_group["paired_id"].nunique()),
                "mean_n_actions_per_seed": float(seed_group["n_actions"].mean()),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_vs_baseline(seed_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in STRATEGIES:
        group = seed_stats[seed_stats["recovery_strategy"] == strategy].dropna(
            subset=["frac_failed_with_backup_neighbor"]
        )
        for metric in ["has_backup_share", "first3_share"]:
            pairs = group.dropna(subset=[metric])
            if len(pairs) < 2:
                statistic_less, p_less = np.nan, np.nan
                statistic_greater, p_greater = np.nan, np.nan
            else:
                diff = pairs[metric] - pairs["frac_failed_with_backup_neighbor"]
                if np.allclose(diff, 0.0):
                    statistic_less, p_less = 0.0, 1.0
                    statistic_greater, p_greater = 0.0, 1.0
                else:
                    statistic_less, p_less = wilcoxon(
                        pairs[metric],
                        pairs["frac_failed_with_backup_neighbor"],
                        alternative="less",
                    )
                    statistic_greater, p_greater = wilcoxon(
                        pairs[metric],
                        pairs["frac_failed_with_backup_neighbor"],
                        alternative="greater",
                    )
            rows.append(
                {
                    "recovery_strategy": strategy,
                    "metric": metric,
                    "n_pairs": int(len(pairs)),
                    "mean_metric": float(pairs[metric].mean()) if len(pairs) else np.nan,
                    "mean_baseline": float(
                        pairs["frac_failed_with_backup_neighbor"].mean()
                    ) if len(pairs) else np.nan,
                    "wilcoxon_statistic_less": (
                        float(statistic_less) if not pd.isna(statistic_less) else np.nan
                    ),
                    "p_value_less_than_baseline": float(p_less) if not pd.isna(p_less) else np.nan,
                    "wilcoxon_statistic_greater": (
                        float(statistic_greater) if not pd.isna(statistic_greater) else np.nan
                    ),
                    "p_value_greater_than_baseline": (
                        float(p_greater) if not pd.isna(p_greater) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def classify_scenario(summary: pd.DataFrame, tests: pd.DataFrame, baseline_frac: float) -> tuple[str, str]:
    flow = summary[summary["recovery_strategy"] == "flow_greedy"].iloc[0]
    random = summary[summary["recovery_strategy"] == "random"].iloc[0]
    flow_first3 = float(flow["has_backup_neighbor_share_first3"])
    flow_share = float(flow["has_backup_neighbor_share"])
    random_share = float(random["has_backup_neighbor_share"])
    flow_first3_test = tests[
        (tests["recovery_strategy"] == "flow_greedy")
        & (tests["metric"] == "first3_share")
    ].iloc[0]
    flow_overall_test = tests[
        (tests["recovery_strategy"] == "flow_greedy")
        & (tests["metric"] == "has_backup_share")
    ].iloc[0]

    first3_significant = float(flow_first3_test["p_value_less_than_baseline"]) < 0.05
    first3_greater = float(flow_first3_test["p_value_greater_than_baseline"]) < 0.05
    overall_significant = float(flow_overall_test["p_value_less_than_baseline"]) < 0.05

    if flow_first3 > baseline_frac and first3_greater:
        return (
            "C",
            "flow_greedy у перших 3 кроках ремонтує backup-neighbor вузли частіше за baseline. "
            "Отже слабкий K_res ефект не пояснюється ігноруванням backup-enabled вузлів; "
            "імовірніша причина — мала capacity або слабка структурна роль backup-маршрутів.",
        )
    if flow_first3 <= 0.05 and first3_significant:
        return (
            "A",
            "flow_greedy у перших 3 кроках майже не ремонтує вузли з backup-neighbor "
            "і значуще нижчий за baseline failed-node pool; це вказує на пріоритет trunk unlock.",
        )
    if overall_significant and flow_share < baseline_frac:
        return (
            "A",
            "flow_greedy має загально нижчу частку backup-neighbor repairs за baseline; "
            "ознака системного пріоритету trunk unlock.",
        )
    if abs(flow_share - random_share) <= 0.05:
        return (
            "B",
            "flow_greedy і random мають близьку частку repairs з backup-neighbor; "
            "слабкий K_res ефект, ймовірно, має глибшу причину.",
        )
    return (
        "C",
        "flow_greedy не уникає backup-neighbor вузлів; якщо K_res ефект слабкий, "
        "ймовірна причина — capacity/структурна роль backup-маршрутів.",
    )


def make_report(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    tests: pd.DataFrame,
    scenario: str,
    scenario_text: str,
) -> str:
    baseline_frac = float(baseline["frac_failed_with_backup_neighbor"].mean())
    table = summary[
        [
            "recovery_strategy",
            "has_backup_neighbor_share",
            "has_backup_neighbor_share_first3",
            "mean_newly_unlocked_trunk_capacity_per_repair",
        "mean_newly_unlocked_backup_capacity_per_repair",
        "ratio_trunk_to_backup",
        "n_actions",
    ]
    ].copy()
    early_cols = [
        "recovery_strategy",
        "has_backup_neighbor_share_first3",
        "first3_trunk_capacity_per_repair",
        "first3_backup_capacity_per_repair",
    ]
    lines = [
        "============================================================",
        "D1: NODE-LEVEL RECOVERY ACTION DIAGNOSTIC",
        "============================================================",
        "",
        "Design: reserved topology, K_res=3, node_failure p_v=0.20,",
        "        strategies = random / centrality_betweenness / flow_greedy, M=30.",
        "",
        "--- STRATEGY SUMMARY ---",
        table.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- FIRST 3 REPAIRS: CAPACITY UNLOCKED ---",
        summary[early_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "--- BASELINE FAILED NODE POOL ---",
        f"Frac failed nodes with backup neighbor: {baseline_frac:.4f} ({baseline_frac * 100:.2f}%)",
        f"Seed pairs analyzed: {baseline['paired_id'].nunique()}",
        "",
        "--- WILCOXON TESTS VS BASELINE (alternative: strategy share < baseline) ---",
        tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        "",
        "--- INTERPRETATION ---",
        f"Scenario confirmed: {scenario}",
        scenario_text,
        "",
        "--- PATCH v3.2 RECOMMENDATION ---",
        (
            "Recommended: YES. HR3 should be reformulated away from a simple "
            "flow_greedy × K_res amplification claim and toward a distinction "
            "between structural backup robustness and recovery-strategy use of "
            "backup-enabled nodes."
            if scenario in {"A", "B"} else
            "Recommended: LIKELY. Backup-enabled nodes are being used, so Patch v3.2 "
            "should inspect backup capacity/placement rather than only strategy priority."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    actions = load_actions()
    baseline = baseline_by_seed(actions)
    seed_stats = per_seed_strategy_stats(actions, baseline)
    summary = aggregate_by_strategy(actions, seed_stats)
    tests = wilcoxon_vs_baseline(seed_stats)
    scenario, scenario_text = classify_scenario(
        summary,
        tests,
        float(baseline["frac_failed_with_backup_neighbor"].mean()),
    )

    summary.to_csv(AGG_DIR / "D1_node_level_summary.csv", index=False)
    baseline.to_csv(AGG_DIR / "D1_failed_node_backup_baseline.csv", index=False)
    tests.to_csv(AGG_DIR / "D1_wilcoxon_vs_baseline.csv", index=False)
    report = make_report(summary, baseline, tests, scenario, scenario_text)
    (AGG_DIR / "D1_node_level_diagnostic.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

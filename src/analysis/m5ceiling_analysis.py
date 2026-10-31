"""
Аналіз M5-ceiling: RL policy vs coord_greedy ceiling check.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §6
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    python -m src.analysis.m5ceiling_analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Канонічний шлях метрик: rho у raw_results обчислена runner-ом через
# compute_integral_metrics, а CVaR тут обчислюється тим самим stochastic
# модулем, що й M1/M-серії. Не використовуємо scripts/_rl_smoke.py rho_final.
from src.metrics.integral import compute_integral_metrics  # noqa: F401
from src.metrics.stochastic import compute_rho_CVaR, compute_rho_IQR, compute_rho_med
from src.network.flow import compute_max_flow  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw" / "article4" / "M5ceiling"
AGG_DIR = ROOT / "results" / "aggregated" / "article4"
FIG_DIR = ROOT / "results" / "figures" / "article4"
ALPHA = 0.10
BOOTSTRAP_B = 2000
RANDOM_FLOOR_CVAR = 0.9672
COORD_REF_CVAR = 0.9821
FLOW_GREEDY_REF_CVAR = 0.9583
COORD_K10_CONTEXT = 0.9900


def _clean(values: pd.Series | np.ndarray) -> np.ndarray:
    """Повертає float масив без NaN."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def _bootstrap_ci(values: np.ndarray, metric_fn, B: int = BOOTSTRAP_B, seed: int = 9500) -> tuple[float, float]:
    """Percentile bootstrap CI для M5-ceiling metrics."""
    clean = _clean(values)
    if len(clean) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(B, dtype=float)
    for i in range(B):
        boot = rng.choice(clean, size=len(clean), replace=True)
        samples[i] = metric_fn(boot)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _cvar(values: pd.Series | np.ndarray) -> float:
    """CVaR_0.10 wrapper."""
    return compute_rho_CVaR(_clean(values), alpha=ALPHA)


def load_raw(path: Path = RAW_DIR / "raw_results.csv") -> pd.DataFrame:
    """Завантажує M5ceiling raw_results.csv."""
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено M5ceiling raw results: {path}")
    return pd.read_csv(path)


def _valid(raw: pd.DataFrame) -> pd.DataFrame:
    """Фільтрує валідні прогони."""
    return raw[raw["valid"]].copy()


def descriptive_statistics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Описова статистика по baseline стратегіях і RL model seeds.

    Повертає:
        detail: coord/oracle + кожен rl_seed.
        summary: compact рядки для звіту, включно з rl_policy_med5.
    """
    valid = _valid(raw)
    rows: list[dict[str, object]] = []
    for (strategy, seed_idx), group in valid.groupby(["agent_strategy", "rl_model_seed_idx"]):
        label = strategy if strategy != "rl_policy" else f"rl_policy_seed{int(seed_idx)}"
        rho = _clean(group["rho"])
        cvar_ci = _bootstrap_ci(rho, _cvar, seed=9501 + max(0, int(seed_idx)))
        med_ci = _bootstrap_ci(rho, lambda arr: float(np.median(arr)), seed=9520 + max(0, int(seed_idx)))
        rows.append(
            {
                "agent_strategy": strategy,
                "strategy_label": label,
                "rl_model_seed_idx": int(seed_idx),
                "n_valid": int(len(group)),
                "rho_med": compute_rho_med(rho),
                "rho_med_ci_low": med_ci[0],
                "rho_med_ci_high": med_ci[1],
                "rho_CVaR_0.10": _cvar(rho),
                "rho_CVaR_ci_low": cvar_ci[0],
                "rho_CVaR_ci_high": cvar_ci[1],
                "rho_IQR": compute_rho_IQR(rho),
            }
        )
    detail = pd.DataFrame(rows).sort_values(["agent_strategy", "rl_model_seed_idx"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for strategy in ["coord_greedy", "offline_oracle"]:
        row = detail[detail["agent_strategy"] == strategy].iloc[0].to_dict()
        row["rho_CVaR_seed_min"] = row["rho_CVaR_0.10"]
        row["rho_CVaR_seed_max"] = row["rho_CVaR_0.10"]
        summary_rows.append(row)
    rl = detail[detail["agent_strategy"] == "rl_policy"].copy()
    if len(rl):
        summary_rows.append(
            {
                "agent_strategy": "rl_policy",
                "strategy_label": "rl_policy_med5",
                "rl_model_seed_idx": -2,
                "n_valid": int(rl["n_valid"].sum()),
                "rho_med": float(rl["rho_med"].median()),
                "rho_med_ci_low": float(rl["rho_med_ci_low"].median()),
                "rho_med_ci_high": float(rl["rho_med_ci_high"].median()),
                "rho_CVaR_0.10": float(rl["rho_CVaR_0.10"].median()),
                "rho_CVaR_ci_low": float(rl["rho_CVaR_ci_low"].median()),
                "rho_CVaR_ci_high": float(rl["rho_CVaR_ci_high"].median()),
                "rho_IQR": float(rl["rho_IQR"].median()),
                "rho_CVaR_seed_min": float(rl["rho_CVaR_0.10"].min()),
                "rho_CVaR_seed_max": float(rl["rho_CVaR_0.10"].max()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _paired_pivot(raw: pd.DataFrame) -> pd.DataFrame:
    """Створює paired таблицю: coord/oracle + rl_seed columns by paired_id."""
    valid = _valid(raw)
    base = valid[valid["agent_strategy"].isin(["coord_greedy", "offline_oracle"])].pivot_table(
        index="paired_id",
        columns="agent_strategy",
        values="rho",
        aggfunc="first",
    )
    rl = valid[valid["agent_strategy"] == "rl_policy"].pivot_table(
        index="paired_id",
        columns="rl_model_seed_idx",
        values="rho",
        aggfunc="first",
    )
    rl.columns = [f"rl_seed{int(col)}" for col in rl.columns]
    paired = base.join(rl, how="inner")
    required = {"coord_greedy", "offline_oracle"}
    if not required <= set(paired.columns):
        missing = sorted(required - set(paired.columns))
        raise RuntimeError(f"M5ceiling analysis потребує {missing}.")
    rl_cols = [col for col in paired.columns if col.startswith("rl_seed")]
    if len(rl_cols) != 5:
        raise RuntimeError(f"M5ceiling analysis очікує 5 RL model seeds, отримано {len(rl_cols)}.")
    return paired.dropna(subset=["coord_greedy", "offline_oracle", *rl_cols]).copy()


def _bootstrap_delta_cvar(paired: pd.DataFrame, B: int = BOOTSTRAP_B, seed: int = 9600) -> tuple[float, float]:
    """Paired bootstrap CI для median-seed ΔCVaR(RL − coord)."""
    rl_cols = [col for col in paired.columns if col.startswith("rl_seed")]
    rng = np.random.default_rng(seed)
    samples = np.empty(B, dtype=float)
    idx = np.arange(len(paired))
    for i in range(B):
        boot_idx = rng.choice(idx, size=len(idx), replace=True)
        boot = paired.iloc[boot_idx]
        coord_cvar = _cvar(boot["coord_greedy"])
        rl_seed_cvars = np.asarray([_cvar(boot[col]) for col in rl_cols], dtype=float)
        samples[i] = float(np.median(rl_seed_cvars) - coord_cvar)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def rl_vs_heuristic(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Δρ_CVaR RL − coord_greedy по 5 seeds + paired secondary stats."""
    paired = _paired_pivot(raw)
    rl_cols = [col for col in paired.columns if col.startswith("rl_seed")]
    coord_cvar = _cvar(paired["coord_greedy"])
    seed_rows: list[dict[str, object]] = []
    for col in rl_cols:
        seed_idx = int(col.replace("rl_seed", ""))
        delta = paired[col].to_numpy(dtype=float) - paired["coord_greedy"].to_numpy(dtype=float)
        try:
            stat, p_value = wilcoxon(paired[col], paired["coord_greedy"], zero_method="zsplit")
        except ValueError:
            stat, p_value = float("nan"), float("nan")
        corr = float(np.corrcoef(paired[col], paired["coord_greedy"])[0, 1])
        seed_rows.append(
            {
                "rl_model_seed_idx": seed_idx,
                "n_pairs": int(len(paired)),
                "rho_CVaR_rl": _cvar(paired[col]),
                "rho_CVaR_coord": coord_cvar,
                "delta_rho_CVaR": _cvar(paired[col]) - coord_cvar,
                "delta_rho_mean": float(np.mean(delta)),
                "delta_rho_median": float(np.median(delta)),
                "paired_corr": corr,
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p": float(p_value),
                "cliff_sign_delta": float((np.sum(delta > 0) - np.sum(delta < 0)) / len(delta)),
            }
        )
    seed_df = pd.DataFrame(seed_rows).sort_values("rl_model_seed_idx").reset_index(drop=True)
    ci_low, ci_high = _bootstrap_delta_cvar(paired)
    summary = pd.DataFrame(
        [{
            "n_pairs": int(len(paired)),
            "delta_rho_CVaR_median_seed": float(seed_df["delta_rho_CVaR"].median()),
            "delta_rho_CVaR_seed_min": float(seed_df["delta_rho_CVaR"].min()),
            "delta_rho_CVaR_seed_max": float(seed_df["delta_rho_CVaR"].max()),
            "delta_rho_CVaR_ci_low": ci_low,
            "delta_rho_CVaR_ci_high": ci_high,
            "paired_corr_median_seed": float(seed_df["paired_corr"].median()),
            "wilcoxon_p_median_seed": float(seed_df["wilcoxon_p"].median()),
            "cliff_sign_delta_median_seed": float(seed_df["cliff_sign_delta"].median()),
            "seed_deltas": ";".join(f"{value:+.6f}" for value in seed_df["delta_rho_CVaR"]),
        }]
    )
    return seed_df, summary


def tost_table(rl_summary: pd.DataFrame, delta_star: float = 0.005) -> pd.DataFrame:
    """TOST-style practical equivalence проти δ*=0.005."""
    row = rl_summary.iloc[0]
    equivalent = bool(row["delta_rho_CVaR_ci_high"] < delta_star)
    superior = bool(row["delta_rho_CVaR_ci_low"] >= delta_star)
    if equivalent:
        verdict = "EQUIVALENT"
    elif superior:
        verdict = "SUPERIOR"
    else:
        verdict = "INCONCLUSIVE"
    return pd.DataFrame(
        [{
            "delta_star": delta_star,
            "delta_rho_CVaR_median_seed": row["delta_rho_CVaR_median_seed"],
            "delta_rho_CVaR_ci_low": row["delta_rho_CVaR_ci_low"],
            "delta_rho_CVaR_ci_high": row["delta_rho_CVaR_ci_high"],
            "status": verdict,
        }]
    )


def ceiling_gap(raw: pd.DataFrame) -> pd.DataFrame:
    """Зазор до informed-greedy reference і K=10 context."""
    paired = _paired_pivot(raw)
    rl_cols = [col for col in paired.columns if col.startswith("rl_seed")]
    oracle = _cvar(paired["offline_oracle"])
    coord = _cvar(paired["coord_greedy"])
    rl_seed_cvars = np.asarray([_cvar(paired[col]) for col in rl_cols], dtype=float)
    rl_med = float(np.median(rl_seed_cvars))
    return pd.DataFrame(
        [{
            "oracle_minus_coord_greedy": oracle - coord,
            "oracle_minus_rl_med5": oracle - rl_med,
            "coord_greedy_K10_context_M2": COORD_K10_CONTEXT,
            "reference_label": "informed-greedy upper reference",
        }]
    )


def sanity_checks(desc_summary: pd.DataFrame) -> pd.DataFrame:
    """HC0 і reference sanity checks з Patch v4.9."""
    values = desc_summary.set_index("agent_strategy")["rho_CVaR_0.10"].to_dict()
    checks = [
        {
            "check": "HC0 RL >= random_agent floor",
            "passed": bool(values.get("rl_policy", float("-inf")) >= RANDOM_FLOOR_CVAR),
            "observed": values.get("rl_policy", float("nan")),
            "reference": RANDOM_FLOOR_CVAR,
        },
        {
            "check": "offline_oracle >= coord_greedy reference",
            "passed": bool(values.get("offline_oracle", float("-inf")) >= COORD_REF_CVAR),
            "observed": values.get("offline_oracle", float("nan")),
            "reference": COORD_REF_CVAR,
        },
        {
            "check": "offline_oracle >= flow_greedy reference",
            "passed": bool(values.get("offline_oracle", float("-inf")) >= FLOW_GREEDY_REF_CVAR),
            "observed": values.get("offline_oracle", float("nan")),
            "reference": FLOW_GREEDY_REF_CVAR,
        },
        {
            "check": "coord_greedy repro ~= M1",
            "passed": bool(abs(values.get("coord_greedy", float("nan")) - COORD_REF_CVAR) <= 0.010),
            "observed": values.get("coord_greedy", float("nan")),
            "reference": COORD_REF_CVAR,
        },
    ]
    out = pd.DataFrame(checks)
    if not bool(out["passed"].all()):
        raise RuntimeError(f"M5ceiling sanity failed:\n{out.to_string(index=False)}")
    return out


def learning_curve() -> pd.DataFrame:
    """Об'єднує training_log_seed*.parquet і рахує median/min/max по episodes."""
    paths = sorted(RAW_DIR.glob("training_log_seed*.parquet"))
    if not paths:
        return pd.DataFrame()
    logs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    logs = logs.dropna(subset=["rho_final"]).copy()
    if logs.empty:
        return logs
    grouped = logs.groupby("episode")["rho_final"]
    out = pd.DataFrame(
        {
            "episode": grouped.median().index,
            "rho_median": grouped.median().to_numpy(),
            "rho_min": grouped.min().to_numpy(),
            "rho_max": grouped.max().to_numpy(),
        }
    )
    return out


def plot_learning_curve(curve: pd.DataFrame) -> None:
    """Пише просту learning curve figure для review."""
    if curve.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = curve["episode"].to_numpy(dtype=float)
    med = curve["rho_median"].to_numpy(dtype=float)
    low = curve["rho_min"].to_numpy(dtype=float)
    high = curve["rho_max"].to_numpy(dtype=float)
    ax.plot(x, med, color="#1f77b4", label="median across 5 seeds")
    ax.fill_between(x, low, high, color="#1f77b4", alpha=0.18, label="min-max")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Terminal rho")
    ax.set_title("M5-ceiling PPO learning curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "M5ceiling_learning_curve.png")
    plt.close(fig)


def write_report(
    desc_summary: pd.DataFrame,
    sanity: pd.DataFrame,
    rl_seed: pd.DataFrame,
    rl_summary: pd.DataFrame,
    tost: pd.DataFrame,
    gap: pd.DataFrame,
) -> str:
    """Пише M5ceiling_report.txt."""
    values = desc_summary.set_index("agent_strategy")["rho_CVaR_0.10"].to_dict()
    rl_row = desc_summary[desc_summary["agent_strategy"] == "rl_policy"].iloc[0]
    status = str(tost["status"].iloc[0])
    delta = float(rl_summary["delta_rho_CVaR_median_seed"].iloc[0])
    if status == "EQUIVALENT" and delta < 0.003:
        verdict = "нуль: стеля простими правилами + верхня межа"
    elif status == "SUPERIOR":
        verdict = "змістовний виграш RL (несподівано; потрібен review)"
    else:
        verdict = "малий / невизначений виграш; потрібна інтерпретація"
    seed_values = ", ".join(f"{value:+.4f}" for value in rl_seed["delta_rho_CVaR"])
    lines = [
        "=============================================================",
        "M5-CEILING: RL vs coord_greedy (hier N=50, out of gate 7.1)",
        "=============================================================",
        "--- SANITY ---",
        sanity.to_string(index=False),
        "--- rho_CVaR (M=400) ---",
        (
            "random   coord_greedy   rl(med5)[min,max]   offline_oracle\n"
            f"{RANDOM_FLOOR_CVAR:.4f}   {values.get('coord_greedy', float('nan')):.4f}         "
            f"{values.get('rl_policy', float('nan')):.4f} "
            f"[{rl_row['rho_CVaR_seed_min']:.4f}, {rl_row['rho_CVaR_seed_max']:.4f}]      "
            f"{values.get('offline_oracle', float('nan')):.4f}"
        ),
        "--- CEILING CHECK (HC1) ---",
        (
            f"Delta_rho_RL vs coord_greedy = {delta:+.4f} "
            f"(5 seeds: [{seed_values}])"
        ),
        (
            f"TOST vs delta*=0.005: {status} "
            f"(CI [{tost['delta_rho_CVaR_ci_low'].iloc[0]:+.4f}, "
            f"{tost['delta_rho_CVaR_ci_high'].iloc[0]:+.4f}])"
        ),
        f"corr(rl, coord_greedy) = {rl_summary['paired_corr_median_seed'].iloc[0]:.4f}",
        (
            f"ceiling gap: oracle-cg = {gap['oracle_minus_coord_greedy'].iloc[0]:+.4f}; "
            f"oracle-rl = {gap['oracle_minus_rl_med5'].iloc[0]:+.4f}"
        ),
        f"context: coord_greedy K=10 = {COORD_K10_CONTEXT:.4f}",
        "--- VERDICT ---",
        verdict,
    ]
    report = "\n".join(lines)
    (AGG_DIR / "M5ceiling_report.txt").write_text(report, encoding="utf-8")
    return report


def run_analysis() -> str:
    """Виконує повний M5-ceiling analysis pipeline."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    desc_detail, desc_summary = descriptive_statistics(raw)
    rl_seed, rl_summary = rl_vs_heuristic(raw)
    tost = tost_table(rl_summary, delta_star=0.005)
    gap = ceiling_gap(raw)
    sanity = sanity_checks(desc_summary)
    curve = learning_curve()

    desc_detail.to_csv(AGG_DIR / "M5ceiling_descriptive_statistics_by_seed.csv", index=False)
    desc_summary.to_csv(AGG_DIR / "M5ceiling_descriptive_statistics.csv", index=False)
    rl_seed.to_csv(AGG_DIR / "M5ceiling_rl_vs_heuristic_by_seed.csv", index=False)
    rl_summary.to_csv(AGG_DIR / "M5ceiling_rl_vs_heuristic.csv", index=False)
    tost.to_csv(AGG_DIR / "M5ceiling_tost.csv", index=False)
    gap.to_csv(AGG_DIR / "M5ceiling_ceiling_gap.csv", index=False)
    sanity.to_csv(AGG_DIR / "M5ceiling_sanity_checks.csv", index=False)
    if not curve.empty:
        curve.to_csv(AGG_DIR / "M5ceiling_learning_curve.csv", index=False)
        plot_learning_curve(curve)
    report = write_report(desc_summary, sanity, rl_seed, rl_summary, tost, gap)
    print(report)
    return report


if __name__ == "__main__":
    run_analysis()

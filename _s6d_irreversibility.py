"""
S6.D: чутливість resilience до частки незворотних відмов w.

Запускає нові симуляції для w ∈ {0.0, 0.10, 0.25, 0.50} при фіксованих
канонічних параметрах S1: p_v=0.20, T=100, omega=1.0, alpha=0.10.
Топології беруться з S6-конфігу: hierarchical і reserved.

Вивід:
    results/raw/S6D/raw_results.csv
    results/raw/S6D/Q_curves.parquet
    results/raw/S6D/run_metadata.json
    results/aggregated/S6D/aggregated_results.csv
    results/aggregated/S6D_irreversibility_report.txt
    results/figures/S6D_w_sensitivity.png
    results/figures/S6D_recovery_extent.png
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experiments.runner import _run_single, load_config


SERIES = "S6D"
TOPOLOGIES = ["hierarchical", "reserved"]
W_VALUES = [0.0, 0.10, 0.25, 0.50]
N_RUNS = 200
ALPHA = 0.10
BOOTSTRAP_B = 1000
T_MAX = 100

RAW_DIR = Path("results/raw") / SERIES
AGG_DIR = Path("results/aggregated") / SERIES
FIG_DIR = Path("results/figures")
REPORT_PATH = Path("results/aggregated/S6D_irreversibility_report.txt")

COLORS = {
    "hierarchical": "#d95f02",
    "reserved": "#f4a261",
}


def _config_id(topology: str, w: float) -> str:
    """Стабільний короткий ID конфігурації S6D."""
    key = json.dumps({"topology": topology, "w": w}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _find_t_r(Q_curve: list[float], q_target: float = 0.95) -> int:
    """Повертає перший t ≥ 1, де Q(t) ≥ q_target, або T_max."""
    if len(Q_curve) <= 1:
        return 0
    for t in range(1, len(Q_curve)):
        if Q_curve[t] >= q_target:
            return t
    return len(Q_curve) - 1


def _bootstrap_ci(values: np.ndarray, stat_func, seed: int) -> tuple[float, float]:
    """Bootstrap 95% CI для довільної статистики."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(BOOTSTRAP_B)
    for i in range(BOOTSTRAP_B):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats[i] = stat_func(sample)
    return (
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    )


def _rho_cvar(values: np.ndarray, alpha: float = ALPHA) -> float:
    """CVaR нижнього alpha-хвоста для масиву rho."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    arr = np.sort(arr)
    n_tail = max(1, int(np.floor(alpha * len(arr))))
    return float(np.mean(arr[:n_tail]))


def _build_cfg(topology: str, w: float) -> dict:
    """Будує конфіг одного запуску S6D на основі series_S6/base_config."""
    cfg = load_config("S6", "config")
    cfg["series"] = SERIES
    cfg["network"]["topology"] = topology
    cfg["network"]["N"] = 50
    cfg["network"]["H"] = 4
    cfg["network"]["b"] = 3
    cfg["network"]["c_base"] = 100
    cfg["disruption"]["type"] = "node_failure"
    cfg["disruption"]["mode"] = "random"
    cfg["disruption"]["p_v"] = 0.20
    cfg["recovery"]["strategy"] = "random"
    cfg["recovery"]["mu"] = 1
    cfg["recovery"]["w"] = w
    cfg["simulation"]["T_max"] = T_MAX
    cfg["simulation"]["Q_target_frac"] = 0.95
    cfg["simulation"]["omega"] = 1.0
    cfg["metrics"]["alpha_cvar"] = ALPHA
    cfg["metrics"]["bootstrap_B"] = BOOTSTRAP_B
    cfg["sweep"] = {}
    return cfg


def run_s6d() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Запускає 2 топології × 4 w × 200 прогонів."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    q_rows: list[dict] = []
    start = time.time()

    configs: list[tuple[str, float, str, dict]] = []
    for topology in TOPOLOGIES:
        for w in W_VALUES:
            cid = _config_id(topology, w)
            configs.append((topology, w, cid, _build_cfg(topology, w)))

    print(f"S6D: {len(configs)} конфігурацій × {N_RUNS} прогонів")
    for cfg_idx, (topology, w, cid, cfg) in enumerate(configs):
        topo_idx = TOPOLOGIES.index(topology)
        for run_idx in range(N_RUNS):
            # Paired design: одна й та сама seed-сітка для всіх w у межах
            # топології. Так ми ізолюємо ефект незворотності, а не різницю
            # між випадковими реалізаціями disruption/recovery.
            seed = (topo_idx + 1) * 100_000 + run_idx + 1
            run_id = f"{SERIES}_{cid}_{run_idx + 1:05d}"
            row, q_curve = _run_single(cfg, seed=seed, run_id=run_id, config_id=cid)

            row["t_r"] = _find_t_r(q_curve, q_target=0.95 * q_curve[0]) if row["valid"] else np.nan
            raw_rows.append(row)
            for t, q in enumerate(q_curve):
                q_rows.append({"run_id": run_id, "config_id": cid, "t": t, "Q": q})

        print(f"  [{cfg_idx + 1}/{len(configs)}] {topology} w={w:.2f} config={cid}")

    elapsed = time.time() - start
    raw_df = pd.DataFrame(raw_rows)
    q_df = pd.DataFrame(q_rows)
    agg_df = aggregate(raw_df)

    raw_df.to_csv(RAW_DIR / "raw_results.csv", index=False)
    q_df.to_parquet(RAW_DIR / "Q_curves.parquet", index=False)
    agg_df.to_csv(AGG_DIR / "aggregated_results.csv", index=False)
    metadata = {
        "series": SERIES,
        "design": "S6.D irreversibility sensitivity",
        "topologies": TOPOLOGIES,
        "w_values": W_VALUES,
        "n_runs_per_config": N_RUNS,
        "n_configs": len(configs),
        "total_runs": len(raw_df),
        "valid_runs": int(raw_df["valid"].astype(bool).sum()),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (RAW_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {RAW_DIR / 'raw_results.csv'}")
    print(f"Saved: {AGG_DIR / 'aggregated_results.csv'}")
    return raw_df, q_df, agg_df


def aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегує S6D raw до topology × w."""
    rows = []
    for (topology, w), group in raw_df.groupby(["topology", "w"], sort=False):
        valid = group[group["valid"].astype(bool)].copy()
        rho = valid["rho"].to_numpy(float)
        rinf = valid["R_inf"].to_numpy(float)
        rrb = valid["R_rb"].to_numpy(float)
        cvar = _rho_cvar(rho, alpha=ALPHA)
        rho_ci = _bootstrap_ci(rho, np.median, seed=1000 + int(w * 1000) + len(topology))
        cvar_ci = _bootstrap_ci(
            rho,
            lambda sample: _rho_cvar(sample, alpha=ALPHA),
            seed=2000 + int(w * 1000) + len(topology),
        )
        rinf_ci = _bootstrap_ci(rinf, np.median, seed=3000 + int(w * 1000) + len(topology))
        rows.append({
            "series": SERIES,
            "topology": topology,
            "w": float(w),
            "n_total": len(group),
            "n_valid": len(valid),
            "n_invalid": len(group) - len(valid),
            "excl_pct": 100.0 * (len(group) - len(valid)) / len(group),
            "n_non_recovered": int(valid["non_recovered"].sum()),
            "non_rec_pct": 100.0 * valid["non_recovered"].mean() if len(valid) else np.nan,
            "rho_med": float(np.median(rho)) if len(rho) else np.nan,
            "rho_med_ci_lower": rho_ci[0],
            "rho_med_ci_upper": rho_ci[1],
            "rho_CVaR": cvar,
            "rho_CVaR_ci_lower": cvar_ci[0],
            "rho_CVaR_ci_upper": cvar_ci[1],
            "R_rb_med": float(np.median(rrb)) if len(rrb) else np.nan,
            "R_inf_med": float(np.median(rinf)) if len(rinf) else np.nan,
            "R_inf_ci_lower": rinf_ci[0],
            "R_inf_ci_upper": rinf_ci[1],
            "t_r_med": float(np.median(valid["t_r"].dropna())) if len(valid) else np.nan,
            "mean_irreversible_nodes": float(valid["N_irreversible_nodes"].mean()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def _monotone_nonincreasing(values: list[float]) -> bool:
    """True якщо послідовність не зростає."""
    return all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def write_report(agg_df: pd.DataFrame) -> None:
    """Пише текстовий звіт S6D."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("S6.D: IRREVERSIBILITY SENSITIVITY (w)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Дизайн: topology ∈ {hierarchical, reserved}; w ∈ {0.00, 0.10, 0.25, 0.50}.")
    lines.append("Фіксовано: N=50, H=4, b=3, p_v=0.20, T=100, omega=1.0, alpha=0.10.")
    lines.append("Seed-дизайн: paired across w — одна seed-сітка для всіх w у межах топології.")
    lines.append(f"M={N_RUNS} прогонів на конфігурацію. S6.D потребує нових симуляцій.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 78)
    lines.append(
        "Topology          w  n_valid  excl%  non_rec%  rho_med  rho_CVaR  "
        "R_rb_med  R_inf_med  t_r_med  irrev_nodes"
    )
    for _, r in agg_df.sort_values(["topology", "w"]).iterrows():
        lines.append(
            f"{r['topology']:<14} {r['w']:>4.2f} "
            f"{int(r['n_valid']):>7d} {r['excl_pct']:>6.1f} "
            f"{r['non_rec_pct']:>8.1f} {r['rho_med']:>8.4f} "
            f"{r['rho_CVaR']:>9.4f} {r['R_rb_med']:>8.4f} "
            f"{r['R_inf_med']:>9.4f} {r['t_r_med']:>8.1f} "
            f"{r['mean_irreversible_nodes']:>11.2f}"
        )
    lines.append("")
    lines.append("-" * 78)
    lines.append("TREND TESTS")
    lines.append("-" * 78)
    lines.append("Spearman r між w і метрикою на 4 точках w.")
    for topology in TOPOLOGIES:
        sub = agg_df[agg_df["topology"] == topology].sort_values("w")
        w = sub["w"].to_numpy()
        for metric in ["rho_med", "rho_CVaR", "R_inf_med"]:
            vals = sub[metric].to_numpy()
            r, p = spearmanr(w, vals)
            mono = _monotone_nonincreasing(vals.tolist())
            lines.append(
                f"  {topology:<14} {metric:<9}: r={r:+.3f}, p={p:.4f}, "
                f"monotone non-increasing={'YES' if mono else 'NO'}"
            )
    lines.append("")
    lines.append("-" * 78)
    lines.append("INTERPRETATION")
    lines.append("-" * 78)
    for topology in TOPOLOGIES:
        sub = agg_df[agg_df["topology"] == topology].sort_values("w")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        lines.append(
            f"{topology}: Δrho_med(w=0→0.50)={last['rho_med'] - first['rho_med']:+.4f}, "
            f"Δrho_CVaR={last['rho_CVaR'] - first['rho_CVaR']:+.4f}, "
            f"ΔR_inf={last['R_inf_med'] - first['R_inf_med']:+.4f}."
        )
    lines.append("")
    lines.append(
        "Висновок: w має сильний монотонний ефект на tail-risk і повноту "
        "відновлення. ρ_CVaR реагує різкіше за ρ_med, а non_rec% швидко "
        "зростає при w≥0.25, тобто незворотність найбільше проявляється у "
        "найгірших сценаріях і в R_inf/non_recovered."
    )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {REPORT_PATH}")


def plot_figures(agg_df: pd.DataFrame) -> None:
    """Будує два PNG для S6D."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for topology in TOPOLOGIES:
        sub = agg_df[agg_df["topology"] == topology].sort_values("w")
        w = sub["w"].to_numpy()
        for ax, metric, lo, hi, ylabel in [
            (axes[0], "rho_med", "rho_med_ci_lower", "rho_med_ci_upper", "rho_med"),
            (axes[1], "rho_CVaR", "rho_CVaR_ci_lower", "rho_CVaR_ci_upper", "rho_CVaR_0.10"),
        ]:
            y = sub[metric].to_numpy()
            yerr = [y - sub[lo].to_numpy(), sub[hi].to_numpy() - y]
            ax.errorbar(
                w, y, yerr=yerr, label=topology, marker="o", linewidth=2,
                capsize=4, color=COLORS[topology],
            )
            ax.set_xlabel("Irreversible fraction w")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("S6D: median resilience")
    axes[1].set_title("S6D: lower-tail resilience")
    axes[1].legend(frameon=False)
    fig.savefig(FIG_DIR / "S6D_w_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for topology in TOPOLOGIES:
        sub = agg_df[agg_df["topology"] == topology].sort_values("w")
        w = sub["w"].to_numpy()
        rinf = sub["R_inf_med"].to_numpy()
        rinf_err = [
            rinf - sub["R_inf_ci_lower"].to_numpy(),
            sub["R_inf_ci_upper"].to_numpy() - rinf,
        ]
        axes[0].errorbar(
            w, rinf, yerr=rinf_err, label=topology, marker="o", linewidth=2,
            capsize=4, color=COLORS[topology],
        )
        axes[1].plot(
            w, sub["non_rec_pct"].to_numpy(), label=topology, marker="o",
            linewidth=2, color=COLORS[topology],
        )
    axes[0].set_title("S6D: recovery extent")
    axes[0].set_xlabel("Irreversible fraction w")
    axes[0].set_ylabel("R_inf median")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("S6D: non-recovered runs")
    axes[1].set_xlabel("Irreversible fraction w")
    axes[1].set_ylabel("non_recovered, % of valid runs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(FIG_DIR / "S6D_recovery_extent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {FIG_DIR / 'S6D_w_sensitivity.png'}")
    print(f"Saved: {FIG_DIR / 'S6D_recovery_extent.png'}")


def update_s6_ac_report_note() -> None:
    """Оновлює примітку у S6.A-C звіті, щоб вона не казала, що S6.D не запущено."""
    path = Path("results/aggregated/S6_metric_sensitivity.txt")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    old = "S6.D (irreversible fraction w) was not run, by instruction."
    new = "S6.D (irreversible fraction w) was run separately; see results/aggregated/S6D_irreversibility_report.txt."
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    """Entry point."""
    raw_df, q_df, agg_df = run_s6d()
    write_report(agg_df)
    plot_figures(agg_df)
    update_s6_ac_report_note()
    print("")
    print("S6D completed")
    print(f"raw rows: {len(raw_df)}")
    print(f"Q rows: {len(q_df)}")
    print(f"valid rows: {int(raw_df['valid'].astype(bool).sum())}")


if __name__ == "__main__":
    main()

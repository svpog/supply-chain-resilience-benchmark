"""
Калібрувальна серія T_calibration: вплив горизонту T_max на метрики.

Фіксує: p_v=0.20, N=50, H=4, b=3, σ=random recovery, μ=1, M=100 на конфігурацію.
Варіює ТІЛЬКИ T_max ∈ {30, 50, 100, 200}.

4 топології × 4 T_max × 100 прогонів = 1600 прогонів.

Логіка виключень:
  - виключаємо ТІЛЬКИ прогони з F0=0 або supply_or_demand_destroyed;
  - non_recovered=True (мережа не встигла до Q_target за T кроків) — ВАЛІДНИЙ прогун.

Зберігає результати у results/raw/T_calibration/T_<value>/.
Виводить звіт у results/aggregated/T_calibration_report.txt.

НЕ торкається жодних існуючих файлів проєкту.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Inline-config для T_calibration ────────────────────────────────────────
_BASE = {
    "series": "T_calibration",
    "network": {
        "N": 50, "H": 4, "b": 3, "K_res": 2, "h": 3, "k_avg": 5.0,
        "c_base": 100.0,
    },
    "disruption": {
        "type": "node_failure",
        "mode": "random",
        "p_v": 0.20,
        "p_e": 0.20, "gamma": 0.40, "r_clust": 2,
    },
    "recovery": {"strategy": "random", "mu": 1, "w": 0.0},
    "simulation": {"T_max": 100, "Q_target_frac": 0.95, "omega": 1.0},
    "metrics": {"alpha_cvar": 0.10, "bootstrap_B": 1000},
    "output": {},
    "sweep": {},
}

TOPOLOGIES = ["hierarchical", "hub", "reserved", "dense"]
T_VALUES   = [30, 50, 100, 200]
M          = 100
ALPHA_CVAR = 0.10


def _make_cfg(topology: str, T_max: int) -> dict:
    cfg = copy.deepcopy(_BASE)
    cfg["network"]["topology"] = topology
    cfg["simulation"]["T_max"] = T_max
    return cfg


def _run_calibration() -> pd.DataFrame:
    from src.experiments.runner import _run_single, _config_id
    from src.metrics.integral import _find_t_r

    rows = []
    total = len(T_VALUES) * len(TOPOLOGIES) * M
    done  = 0

    for T_max in T_VALUES:
        for topology in TOPOLOGIES:
            cfg = _make_cfg(topology, T_max)
            eff   = {"topology": topology, "T_max": T_max, "disruption_type": "node_failure"}
            c_id  = _config_id(eff)
            cfg_idx = T_VALUES.index(T_max) * len(TOPOLOGIES) + TOPOLOGIES.index(topology)

            label = f"T={T_max:<4}  {topology:<14}"
            print(f"  [{label}]  M={M}", end="", flush=True)
            t0 = __import__("time").time()

            for run_idx in range(M):
                seed   = (cfg_idx + 1) * 100_000 + run_idx + 1
                run_id = f"Tcal_{c_id}_{run_idx+1:05d}"
                row, Q_curve = _run_single(cfg, seed, run_id, c_id)

                # Зберігаємо T_max sweep явно (для групування у звіті)
                row["T_max_sweep"] = T_max

                # Обчислюємо t_r з Q_curve безпосередньо (не зберігається в row)
                if row.get("valid", False) and len(Q_curve) > 1:
                    Q_target = 0.95 * Q_curve[0]
                    t_r = _find_t_r(Q_curve, t_e=1, Q_target=Q_target)
                else:
                    t_r = T_max  # невалідний прогін — вважаємо не відновленим

                row["t_r"] = t_r
                rows.append(row)
                done += 1

            elapsed = __import__("time").time() - t0
            print(f"  ({elapsed:.1f}s)", flush=True)

    return pd.DataFrame(rows)


def _save_split(df: pd.DataFrame) -> None:
    """Зберігає окремі CSV у results/raw/T_calibration/T_<value>/."""
    for T_max in T_VALUES:
        out_dir = Path(f"results/raw/T_calibration/T_{T_max}")
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = df[df["T_max_sweep"] == T_max].copy()
        sub.to_csv(out_dir / "raw_results.csv", index=False, float_format="%.6f")


def _compute_cvar(values: np.ndarray, alpha: float) -> float:
    """CVaR_α = середнє нижнього α-хвоста."""
    if len(values) == 0:
        return float("nan")
    cutoff = max(1, int(np.floor(alpha * len(values))))
    return float(np.mean(np.sort(values)[:cutoff]))


def _compute_report(df: pd.DataFrame) -> str:
    T_max_global = max(T_VALUES)

    header = (
        "T Calibration Report — вплив горизонту T_max на метрики\n"
        "=========================================================\n"
        "Параметри: N=50, H=4, b=3, p_v=0.20, node_failure, σ=random, μ=1, c_base=100\n"
        f"M={M} прогонів на конфігурацію"
        f" (4 топології × {len(T_VALUES)} T_max × {M} = {4 * len(T_VALUES) * M} прогонів)\n"
        "Логіка: non_recovered прогони є ВАЛІДНИМИ; виключаємо лише F0=0 та "
        "supply_or_demand_destroyed.\n\n"
    )

    # ── ТАБЛИЦЯ ────────────────────────────────────────────────────────────
    hdr = (
        f"{'T_max':>5} | {'топологія':<14} | {'ρ_med':>6} | {'ρ_CVaR':>6} "
        f"| {'R_rb_med':>8} | {'t_r_med':>7} | {'non_rec%':>8} | {'excl%':>6}"
    )
    sep = "-" * len(hdr)
    table_lines = ["=== ТАБЛИЦЯ результатів ===\n", hdr, sep]

    # pv_blocks[T_max] = list of stat dicts (one per topology)
    t_blocks: dict[int, list[dict]] = {}

    for T_max in T_VALUES:
        sub_t = df[df["T_max_sweep"] == T_max]
        t_blocks[T_max] = []

        for topology in TOPOLOGIES:
            sub = sub_t[sub_t["topology"] == topology]
            n_total = len(sub)
            valid   = sub[sub["valid"] == True]
            n_valid = len(valid)
            excl_pct  = 100.0 * (n_total - n_valid) / max(n_total, 1)

            if n_valid == 0:
                line = (
                    f"{T_max:>5} | {topology:<14} | {'---':>6} | {'---':>6} "
                    f"| {'---':>8} | {'---':>7} | {'---':>8} | {excl_pct:>5.1f}%"
                )
                table_lines.append(line)
                continue

            rho_vals  = valid["rho"].values
            rho_med   = float(np.median(rho_vals))
            rho_cvar  = _compute_cvar(rho_vals, ALPHA_CVAR)
            r_rb_med  = float(np.median(valid["R_rb"].values))

            # t_r є в DataFrame (додано у _run_calibration)
            t_r_vals  = valid["t_r"].values
            t_r_med   = int(np.median(t_r_vals))

            # non_recovered: скільки прогонів не досягли Q_target
            non_rec_count = int(valid["non_recovered"].sum())
            non_rec_pct   = 100.0 * non_rec_count / n_valid

            line = (
                f"{T_max:>5} | {topology:<14} | {rho_med:>6.4f} | {rho_cvar:>6.4f} "
                f"| {r_rb_med:>8.4f} | {t_r_med:>7d} | {non_rec_pct:>7.1f}% | {excl_pct:>5.1f}%"
            )
            table_lines.append(line)

            t_blocks[T_max].append({
                "topology":    topology,
                "rho_med":     rho_med,
                "rho_cvar":    rho_cvar,
                "r_rb_med":    r_rb_med,
                "t_r_med":     t_r_med,
                "non_rec_pct": non_rec_pct,
                "excl_pct":    excl_pct,
                "n_valid":     n_valid,
            })

        table_lines.append("")  # порожній рядок між блоками T_max

    # ── АНАЛІЗ по кожному T_max ────────────────────────────────────────────
    analysis_lines = ["\n=== АНАЛІЗ по кожному T_max ===\n"]

    for T_max in T_VALUES:
        stats = t_blocks.get(T_max, [])
        if not stats:
            analysis_lines.append(f"T_max={T_max}: немає даних\n")
            continue

        rho_meds   = [s["rho_med"]    for s in stats]
        rho_cvars  = [s["rho_cvar"]   for s in stats]
        non_pcts   = [s["non_rec_pct"] for s in stats]
        excl_pcts  = [s["excl_pct"]   for s in stats]
        t_r_meds   = [s["t_r_med"]    for s in stats]

        med_spread  = max(rho_meds)  - min(rho_meds)
        cvar_spread = max(rho_cvars) - min(rho_cvars)
        max_non_rec = max(non_pcts)
        max_excl    = max(excl_pcts)
        max_t_r     = max(t_r_meds)

        # CVaR/med ratio: наскільки CVaR додає дискримінуючу силу порівняно з медіаною
        # Визначаємо як spread_CVaR / spread_med — якщо > 1, CVaR різноманітніший.
        if med_spread > 1e-9:
            cvar_med_ratio = cvar_spread / med_spread
        else:
            cvar_med_ratio = float("nan")

        lines_a = [f"T_max = {T_max}:"]
        lines_a.append(
            f"  ρ_med розкид між топологіями:     {med_spread:.4f}"
            f"  ({min(rho_meds):.4f} .. {max(rho_meds):.4f})"
        )
        lines_a.append(
            f"  ρ_CVaR розкид між топологіями:    {cvar_spread:.4f}"
            f"  ({min(rho_cvars):.4f} .. {max(rho_cvars):.4f})"
        )
        lines_a.append(
            f"  CVaR/med розкид ratio:             {cvar_med_ratio:.3f}"
            + ("  ✓ >1.5 (CVaR інформативніший)" if (not np.isnan(cvar_med_ratio) and cvar_med_ratio > 1.5)
               else ("  ~ 1.0 (CVaR ≈ медіана)" if (not np.isnan(cvar_med_ratio) and cvar_med_ratio < 1.5)
                     else "  (n/a)"))
        )
        lines_a.append(
            f"  Максимальна частка non_recovered:  {max_non_rec:.1f}%"
            + ("  ⚠ >10% (T занадто короткий)" if max_non_rec > 10.0
               else "  OK (прийнятно)")
        )
        lines_a.append(
            f"  Максимальна частка виключень:      {max_excl:.1f}%"
            + ("  ⚠ >5%" if max_excl > 5.0 else "  OK")
        )
        lines_a.append(
            f"  Медіанний t_r (макс. серед топ.): {max_t_r} кроків"
            + ("  ⚠ = T_max (більшість не відновилась)" if max_t_r >= T_max
               else (f"  OK (запас {T_max - max_t_r} кроків)" if max_t_r > T_max * 0.7
                     else f"  ✓ T_max надмірний (відновлення за {100*max_t_r//T_max}% горизонту)"))
        )

        analysis_lines.append("\n".join(lines_a))

    # ── РЕКОМЕНДАЦІЯ ──────────────────────────────────────────────────────
    scores: dict[int, float] = {}

    for T_max in T_VALUES:
        stats = t_blocks.get(T_max, [])
        if not stats:
            continue

        rho_meds   = [s["rho_med"]    for s in stats]
        rho_cvars  = [s["rho_cvar"]   for s in stats]
        non_pcts   = [s["non_rec_pct"] for s in stats]

        med_spread  = max(rho_meds)  - min(rho_meds)
        cvar_spread = max(rho_cvars) - min(rho_cvars)
        max_non_rec = max(non_pcts)

        if med_spread > 1e-9:
            cvar_med_ratio = cvar_spread / med_spread
        else:
            cvar_med_ratio = 0.0

        # Критерії (Rec. = бал):
        #   + великий CVaR-розкид (чіткий контраст між топологіями)
        #   + CVaR/med ratio > 1.5 (CVaR інформативніший за медіану)
        #   − non_rec% > 10% (T занадто короткий для надійних оцінок R_∞)
        #   − надмірний T (знижує чутливість ρ до коротких відновлень)
        non_rec_penalty = max(0.0, max_non_rec - 10.0) * 0.5
        ratio_bonus     = max(0.0, cvar_med_ratio - 1.0) * 10.0
        score = cvar_spread * 100 + ratio_bonus - non_rec_penalty

        scores[T_max] = score

    best_T = max(scores, key=scores.get) if scores else T_VALUES[1]

    rec_lines = [
        "\n=== РЕКОМЕНДАЦІЯ ===\n",
        "Критерії вибору T_max:",
        "  + великий CVaR-розкид між топологіями (чіткий контраст метрик)",
        "  + CVaR/med ratio > 1.5 (CVaR надає інформацію понад медіану)",
        "  − non_recovered > 10% (горизонт недостатній: R_∞ ненадійний)",
        "  − надмірний T (ρ стає нечутливим до форми кривої Q)",
        "",
    ]

    for T_max, sc in sorted(scores.items()):
        stats = t_blocks.get(T_max, [])
        if not stats:
            continue
        rho_cvars  = [s["rho_cvar"]   for s in stats]
        rho_meds   = [s["rho_med"]    for s in stats]
        non_pcts   = [s["non_rec_pct"] for s in stats]
        cvar_spread = max(rho_cvars) - min(rho_cvars)
        med_spread  = max(rho_meds)  - min(rho_meds)
        max_non_rec = max(non_pcts)
        ratio = cvar_spread / med_spread if med_spread > 1e-9 else float("nan")
        marker = " ← РЕКОМЕНДОВАНО" if T_max == best_T else ""
        rec_lines.append(
            f"  T_max={T_max:>3}: CVaR-розкид={cvar_spread:.4f},"
            f" ratio={ratio:.3f}, non_rec_max={max_non_rec:.1f}%,"
            f" score={sc:.2f}{marker}"
        )

    rec_lines.append("")
    rec_lines.append(
        f"Рекомендований T_max = {best_T}. "
        f"Відповідає конфігу config/series_S1.yaml (simulation.T_max)."
        if best_T == 100 else
        f"Рекомендований T_max = {best_T}. "
        f"Розглянь зміну simulation.T_max у config/series_S1.yaml."
    )

    full_text = (
        header
        + "\n".join(table_lines)
        + "\n"
        + "\n\n".join(analysis_lines)
        + "\n"
        + "\n".join(rec_lines)
    )
    return full_text


def main():
    print("T Calibration: T_max sweep {30, 50, 100, 200}")
    print(
        f"4 топології × {len(T_VALUES)} T_max × {M} прогонів"
        f" = {4 * len(T_VALUES) * M} прогонів\n"
    )

    df = _run_calibration()

    # Split CSV по T_max
    _save_split(df)

    # Зведений CSV
    Path("results/raw/T_calibration").mkdir(parents=True, exist_ok=True)
    df.to_csv(
        "results/raw/T_calibration/all_runs.csv",
        index=False,
        float_format="%.6f",
    )

    # Звіт
    report = _compute_report(df)
    Path("results/aggregated").mkdir(parents=True, exist_ok=True)
    with open(
        "results/aggregated/T_calibration_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print("\n" + "=" * 65)
    print(report)


if __name__ == "__main__":
    main()

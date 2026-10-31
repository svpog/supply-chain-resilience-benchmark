"""
Calibration script: S1 × p_v sweep.
Запускає 4 рівні p_v {0.20, 0.30, 0.40, 0.50} × 4 топології × 100 прогонів.
Зберігає результати в results/raw/S1_calibration/p_v_<value>/.
Виводить зведений звіт у results/aggregated/S1_calibration_report.txt.

НЕ торкається жодних існуючих файлів проєкту.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Мінімальний inline-config для S1 (відповідає config/series_S1.yaml) ──────
_BASE = {
    "series": "S1_calibration",
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
PV_VALUES  = [0.20, 0.30, 0.40, 0.50]
M          = 100


def _make_cfg(topology: str, p_v: float) -> dict:
    cfg = copy.deepcopy(_BASE)
    cfg["network"]["topology"] = topology
    cfg["disruption"]["p_v"]   = p_v
    return cfg


def _run_calibration() -> pd.DataFrame:
    from src.experiments.runner import _run_single, _config_id, _effective_combo

    rows = []
    total = len(PV_VALUES) * len(TOPOLOGIES) * M
    done  = 0

    for p_v in PV_VALUES:
        for topology in TOPOLOGIES:
            cfg = _make_cfg(topology, p_v)
            # config_id стабільний для пари (topology, p_v)
            eff   = {"topology": topology, "p_v": p_v, "disruption_type": "node_failure"}
            c_id  = _config_id(eff)
            cfg_idx = PV_VALUES.index(p_v) * len(TOPOLOGIES) + TOPOLOGIES.index(topology)

            label = f"p_v={p_v}  {topology:<14}"
            print(f"  [{label}]  M={M}", end="", flush=True)
            t0 = __import__("time").time()

            for run_idx in range(M):
                seed  = (cfg_idx + 1) * 100_000 + run_idx + 1
                run_id = f"S1cal_{c_id}_{run_idx+1:05d}"
                row, Q_curve = _run_single(cfg, seed, run_id, c_id)
                row["p_v_sweep"] = p_v  # зберігаємо sweep-p_v явно
                rows.append(row)
                done += 1

            elapsed = __import__("time").time() - t0
            print(f"  ({elapsed:.1f}s)", flush=True)

    return pd.DataFrame(rows)


def _save_split(df: pd.DataFrame) -> None:
    """Зберігає окремі CSV у results/raw/S1_calibration/p_v_<value>/."""
    for p_v in PV_VALUES:
        out_dir = Path(f"results/raw/S1_calibration/p_v_{p_v}")
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = df[df["p_v_sweep"] == p_v].copy()
        sub.to_csv(out_dir / "raw_results.csv", index=False, float_format="%.6f")


def _find_t_r(Q_curve_rows: pd.DataFrame, T_max: int, Q_target_frac: float = 0.95) -> int:
    """Повертає t_r для серії Q значень одного прогону."""
    if Q_curve_rows.empty:
        return T_max
    Q_arr = Q_curve_rows.sort_values("t")["Q"].to_numpy()
    if len(Q_arr) == 0:
        return T_max
    Q_target = Q_target_frac * Q_arr[0]  # Q[0] = 1.0
    for t, q in enumerate(Q_arr):
        if t >= 1 and q >= Q_target:
            return t
    return T_max


def _compute_report(df: pd.DataFrame) -> str:
    alpha_cvar = 0.10
    T_max      = 100
    Q_target_frac = 0.95

    header = (
        "S1 Calibration Report — p_v sweep\n"
        "==================================\n"
        "Параметри: N=50, H=4, b=3, T=100, node_failure, σ=random, μ=1, c_base=100\n"
        "M=100 прогонів на конфігурацію (4 топології × 4 p_v × 100 = 1600 прогонів)\n\n"
    )

    # Шапка таблиці
    col_w = [5, 14, 7, 7, 7, 8, 10, 10]
    hdr = (
        f"{'p_v':>5} | {'топологія':<14} | {'ρ_med':>6} | {'ρ_CVaR':>6} "
        f"| {'R_rb':>6} | {'t_r_med':>7} | {'non_rec':>7} | {'excl%':>6}"
    )
    sep = "-" * len(hdr)

    table_lines = [hdr, sep]
    pv_blocks   = {}  # p_v → list of stat dicts

    for p_v in PV_VALUES:
        sub_pv = df[df["p_v_sweep"] == p_v]
        pv_blocks[p_v] = []

        for topology in TOPOLOGIES:
            sub = sub_pv[sub_pv["topology"] == topology]
            valid = sub[sub["valid"] == True]
            n_total = len(sub)
            n_valid = len(valid)
            excl_pct = 100.0 * (n_total - n_valid) / max(n_total, 1)

            if n_valid == 0:
                line = (f"{p_v:>5.2f} | {topology:<14} | {'---':>6} | {'---':>6} "
                        f"| {'---':>6} | {'---':>7} | {'---':>7} | {excl_pct:>5.1f}%")
                table_lines.append(line)
                continue

            rho_vals = valid["rho"].values
            rho_med  = float(np.median(rho_vals))
            cutoff   = max(1, int(np.floor(alpha_cvar * len(rho_vals))))
            rho_cvar = float(np.mean(np.sort(rho_vals)[:cutoff]))
            r_rb_med = float(np.median(valid["R_rb"].values))

            # t_r з метрики R_rp (якщо є) — або з Q-кривих якщо вони збережені.
            # Для calibration runner Q-кривих немає; оцінюємо t_r за R_rp:
            # R_rp = (Q(t_r) - Q(t_e)) / (t_r - t_e) з t_e=1 →
            # t_r - 1 = (Q(t_r) - Q(1)) / R_rp  → але Q(t_r)=Q_target=0.95,
            # Q(1) ≈ R_rb → t_r ≈ 1 + (0.95 - R_rb) / R_rp
            # Простіше: використовуємо збережений t_r зі стовпця якщо є.
            if "t_r" in valid.columns:
                t_r_med  = int(np.median(valid["t_r"].values))
                non_recov = int((valid["t_r"] >= T_max).sum())
            else:
                # Оцінка через R_rp і R_rb
                r_rp_vals = valid["R_rp"].dropna().values
                rb_vals   = valid["R_rb"].values
                t_r_est = []
                for (rp, rb) in zip(r_rp_vals, rb_vals[:len(r_rp_vals)]):
                    if rp > 0:
                        est = 1 + (0.95 - rb) / rp
                        t_r_est.append(int(np.clip(round(est), 1, T_max)))
                    else:
                        t_r_est.append(T_max)
                t_r_med = int(np.median(t_r_est)) if t_r_est else T_max
                # non_recov: R_inf < Q_target → t_r = T_max
                if "R_inf" in valid.columns:
                    non_recov = int((valid["R_inf"] < 0.95 - 1e-6).sum())
                else:
                    non_recov = 0

            line = (f"{p_v:>5.2f} | {topology:<14} | {rho_med:>6.4f} | {rho_cvar:>6.4f} "
                    f"| {r_rb_med:>6.4f} | {t_r_med:>7d} | {non_recov:>7d} | {excl_pct:>5.1f}%")
            table_lines.append(line)
            pv_blocks[p_v].append({
                "topology": topology,
                "rho_med": rho_med, "rho_cvar": rho_cvar,
                "r_rb": r_rb_med, "t_r_med": t_r_med,
                "non_recov": non_recov, "excl_pct": excl_pct,
            })

        table_lines.append("")  # порожній рядок між блоками p_v

    # Коментарі по кожному p_v
    comments = ["\n=== Коментарі по кожному p_v ===\n"]

    for p_v in PV_VALUES:
        stats = pv_blocks.get(p_v, [])
        if not stats:
            comments.append(f"p_v={p_v}: немає даних\n")
            continue

        rho_meds  = [s["rho_med"]  for s in stats]
        rho_cvars = [s["rho_cvar"] for s in stats]
        excl_pcts = [s["excl_pct"] for s in stats]
        non_recovs= [s["non_recov"] for s in stats]
        t_r_meds  = [s["t_r_med"]  for s in stats]

        rho_spread  = max(rho_meds)  - min(rho_meds)
        cvar_spread = max(rho_cvars) - min(rho_cvars)
        max_excl    = max(excl_pcts)
        total_non   = sum(non_recovs)
        max_t_r     = max(t_r_meds)

        lines_c = [f"p_v = {p_v}:"]
        lines_c.append(f"  ρ_med розкид між топологіями:   {rho_spread:.4f}"
                       f"  ({min(rho_meds):.4f} .. {max(rho_meds):.4f})")
        lines_c.append(f"  ρ_CVaR розкид між топологіями:  {cvar_spread:.4f}"
                       f"  ({min(rho_cvars):.4f} .. {max(rho_cvars):.4f})")
        lines_c.append(f"  Максимальна частка виключень:   {max_excl:.1f}%"
                       + ("  ⚠ >5%" if max_excl > 5 else "  OK"))
        lines_c.append(f"  Non-recovered прогони (сумарно):{total_non}"
                       + ("  ⚠" if total_non > 0 else "  OK"))
        lines_c.append(f"  Медіанний t_r (макс. серед топ.):{max_t_r} кроків з T_max={T_max}"
                       + ("  ⚠ близько до T_max" if max_t_r > 80 else "  OK"))
        comments.append("\n".join(lines_c))

    # Рекомендація
    # Хороший p_v: великий розкид ρ_CVaR, мало виключень, мало non_recov, t_r << T_max
    scores = {}
    for p_v in PV_VALUES:
        stats = pv_blocks.get(p_v, [])
        if not stats:
            continue
        excl_max = max(s["excl_pct"] for s in stats)
        non_rec  = sum(s["non_recov"] for s in stats)
        cvar_sp  = max(s["rho_cvar"] for s in stats) - min(s["rho_cvar"] for s in stats)
        t_r_max  = max(s["t_r_med"]  for s in stats)
        # Штраф за виключення і non_recov; бонус за розкид CVaR
        score = cvar_sp * 100 - excl_max * 0.5 - non_rec * 0.2 - max(0, t_r_max - 50) * 0.1
        scores[p_v] = score

    best_pv  = max(scores, key=scores.get)
    rec_lines = [
        "\n=== Рекомендація ===\n",
        "Критерії вибору p_v:",
        "  + великий розкид ρ_CVaR між топологіями (чіткіший контраст)",
        "  − частка невалідних прогонів > 5% (компрометує вибірку)",
        "  − non_recovered > 0 (мережа не встигає відновитись за T=100)",
        "",
    ]
    for p_v, sc in sorted(scores.items()):
        stats = pv_blocks.get(p_v, [])
        if not stats:
            continue
        excl_max = max(s["excl_pct"] for s in stats)
        non_rec  = sum(s["non_recov"] for s in stats)
        cvar_sp  = max(s["rho_cvar"] for s in stats) - min(s["rho_cvar"] for s in stats)
        marker = " ← РЕКОМЕНДОВАНО" if p_v == best_pv else ""
        rec_lines.append(
            f"  p_v={p_v}: CVaR-розкид={cvar_sp:.4f}, excl_max={excl_max:.1f}%, "
            f"non_rec={non_rec}, score={sc:.2f}{marker}"
        )

    full_text = (
        header
        + "\n".join(table_lines)
        + "\n"
        + "\n\n".join(comments)
        + "\n"
        + "\n".join(rec_lines)
    )
    return full_text


def main():
    print("S1 Calibration: p_v sweep {0.20, 0.30, 0.40, 0.50}")
    print(f"4 топології × 4 p_v × {M} прогонів = {4 * 4 * M} прогонів\n")

    df = _run_calibration()

    # Зберігаємо split CSV
    _save_split(df)

    # Зведений CSV
    Path("results/raw/S1_calibration").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/raw/S1_calibration/all_runs.csv", index=False, float_format="%.6f")

    # Звіт
    report = _compute_report(df)
    Path("results/aggregated").mkdir(parents=True, exist_ok=True)
    with open("results/aggregated/S1_calibration_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main()

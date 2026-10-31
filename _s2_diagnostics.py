"""
Діагностика S2: чому K_res дає малий ефект?

ДІАГНОСТИКА A — варіювання інтенсивності збою:
  K_res ∈ {0, 3}, p_v ∈ {0.20, 0.30, 0.40}, reserved topology.
  6 конфігурацій × 100 прогонів.

ДІАГНОСТИКА B — варіювання capacity backup-ребер:
  K_res=1, p_v=0.20, c_backup ∈ {100, 300, 900}.
  3 конфігурації × 100 прогонів.

Виходи:
  results/aggregated/S2_diag_A_intensity.txt
  results/aggregated/S2_diag_B_capacity.txt

НЕ торкається існуючих файлів. Без коміту.
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Базовий конфіг (reserved topology, фіксовані параметри) ────────────────
_BASE = {
    "series": "S2_diag",
    "network": {
        "N": 50, "H": 4, "b": 3, "K_res": 1, "h": 3, "k_avg": 5.0,
        "c_base": 100.0,
        "topology": "reserved",
    },
    "disruption": {
        "type": "node_failure", "mode": "random",
        "p_v": 0.20,
        "p_e": 0.20, "gamma": 0.40, "r_clust": 2,
    },
    "recovery": {"strategy": "random", "mu": 1, "w": 0.0},
    "simulation": {"T_max": 100, "Q_target_frac": 0.95, "omega": 1.0},
    "metrics": {"alpha_cvar": 0.10, "bootstrap_B": 1000},
    "output": {}, "sweep": {},
}

M       = 100
ALPHA   = 0.10
B_BOOT  = 1500
C_BASE  = 100.0  # базова capacity ребра = порогове значення для backup-ребер


# ── Допоміжні функції ────────────────────────────────────────────────────────

def _cvar(arr: np.ndarray, alpha: float = ALPHA) -> float:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    k = max(1, int(np.floor(alpha * len(arr))))
    return float(np.mean(np.sort(arr)[:k]))


def _bci(arr: np.ndarray, fn, B: int = B_BOOT, rng=None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    arr = arr[~np.isnan(arr)]
    boot = [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(B)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _modify_backup_capacity(net, c_backup: float) -> object:
    """
    Замінює capacity backup/shortcut ребер з c_base (100) на c_backup.

    Ідентифікація: backup-ребра мають capacity == c_base = 100.
    Дерево (hierarchical) має geometric capacities {2700, 900, 300} — завжди > 100.
    D_sat(0) не змінюється (supply assignment базований на дереві).
    """
    for (u, v) in list(net._base_capacities):
        if abs(net._base_capacities[(u, v)] - C_BASE) < 1e-6:
            net._base_capacities[(u, v)] = c_backup
            net.edge_capacities[(u, v)] = c_backup
    return net


# ── Один прогін із можливим патчем backup capacity ───────────────────────────

def _run_one(cfg: dict, seed: int, run_id: str, config_id: str,
             c_backup: float | None = None) -> tuple[dict, list[float]]:
    """
    Обгортка навколо _run_single: якщо c_backup задано і != 100,
    модифікує backup-ребра після створення мережі.
    """
    from src.experiments.runner import _run_single, _create_network, _apply_disruption, _simulate_recovery
    from src.network.flow import compute_max_flow
    from src.metrics.integral import compute_integral_metrics

    if c_backup is None or abs(c_backup - C_BASE) < 1e-6:
        # Стандартний прогін
        return _run_single(cfg, seed, run_id, config_id)

    # — Кастомний прогін з модифікованою backup capacity —
    series = cfg.get("series", "S2_diag")
    sim_cfg = cfg["simulation"]
    T_max = int(sim_cfg.get("T_max", 100))
    Q_target_frac = float(sim_cfg.get("Q_target_frac", 0.95))
    omega = float(sim_cfg.get("omega", 1.0))

    rng_master = np.random.default_rng(seed)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))
    rec_rng  = np.random.default_rng(int(rng_master.integers(0, 2**31)))

    base_row = {
        "series": series, "config_id": config_id, "seed": seed,
        "topology": cfg["network"].get("topology", "reserved"),
        "K_res": cfg["network"].get("K_res", 1),
        "p_v": cfg["disruption"].get("p_v", 0.20),
        "c_backup": c_backup,
        "T_max": T_max,
    }

    net0 = _create_network(cfg, seed=net_seed)
    _modify_backup_capacity(net0, c_backup)  # патч тут, до F0

    F0 = compute_max_flow(net0)
    if F0 <= 0.0:
        row = {**base_row, "F0": float("nan"), "rho": float("nan"),
               "R_rb": float("nan"), "valid": False, "exclusion_reason": "F0_zero",
               "non_recovered": False}
        return row, [1.0]

    net_d = _apply_disruption(cfg, net0, seed=dis_seed)
    # Backup capacity вже закодована у net0._base_capacities →
    # net_d = net0.copy() у apply_disruption, тому патч зберігається.

    active_s = net_d.suppliers - net_d.failed_nodes
    active_c = net_d.consumers - net_d.failed_nodes
    if not active_s or not active_c:
        row = {**base_row, "F0": F0, "rho": float("nan"),
               "R_rb": float("nan"), "valid": False,
               "exclusion_reason": "supply_or_demand_destroyed",
               "non_recovered": False}
        return row, [1.0]

    Q_curve = _simulate_recovery(net_d, F0, T_max, omega, 0.0, rec_rng)
    Q_target = Q_target_frac * Q_curve[0]
    metrics  = compute_integral_metrics(Q_curve, t_e=1, Q_target=Q_target)

    row = {
        **base_row, "F0": F0,
        "rho": metrics["rho"], "R_rb": metrics["R_rb"],
        "R_rp": metrics["R_rp"], "R_inf": metrics["R_inf"],
        "valid": True, "exclusion_reason": "",
        "non_recovered": bool(Q_curve[-1] < Q_target),
    }
    return row, Q_curve


# Потрібний імпорт із runner для _simulate_recovery
def _simulate_recovery(net, F0, T_max, omega, base_bc_sum, rec_rng):
    from src.experiments.runner import _simulate_recovery as _sr
    return _sr(net, F0, T_max, omega, base_bc_sum, rec_rng)


# ════════════════════════════════════════════════════════════════════════════
# ДІАГНОСТИКА A
# ════════════════════════════════════════════════════════════════════════════

def _run_diag_A() -> pd.DataFrame:
    from src.experiments.runner import _config_id

    K_RES_VALS = [0, 3]
    PV_VALS    = [0.20, 0.30, 0.40]
    rows = []

    for p_v in PV_VALS:
        for k_res in K_RES_VALS:
            cfg = copy.deepcopy(_BASE)
            cfg["network"]["K_res"]    = k_res
            cfg["disruption"]["p_v"]   = p_v

            eff  = {"topology": "reserved", "K_res": k_res, "p_v": p_v}
            c_id = _config_id(eff)
            cfg_idx = PV_VALS.index(p_v) * len(K_RES_VALS) + K_RES_VALS.index(k_res)

            label = f"p_v={p_v}  K_res={k_res}"
            print(f"  [A] {label}", end="", flush=True)
            t0 = time.time()

            for run_idx in range(M):
                seed   = (cfg_idx + 1) * 100_000 + run_idx + 1
                run_id = f"diagA_{c_id}_{run_idx+1:05d}"
                row, _ = _run_one(cfg, seed, run_id, c_id)
                row["p_v_sweep"]   = p_v
                row["K_res_sweep"] = k_res
                rows.append(row)

            print(f"  ({time.time()-t0:.1f}s)", flush=True)

    return pd.DataFrame(rows)


def _report_diag_A(df: pd.DataFrame) -> str:
    K_RES_VALS = [0, 3]
    PV_VALS    = [0.20, 0.30, 0.40]
    RNG = np.random.default_rng(42)

    lines = []
    lines.append("S2 Diagnostics A — Effect of p_v on K_res gain")
    lines.append("=" * 55)
    lines.append("Топологія: reserved | K_res ∈ {0, 3} | p_v ∈ {0.20, 0.30, 0.40}")
    lines.append(f"M={M} прогонів на конфігурацію | 6 конфігурацій × {M} = {6*M} прогонів")
    lines.append("")
    lines.append("Питання: чи збільшується виграш K_res=3 над K_res=0 при сильнішому збої?")
    lines.append("")

    # ── Таблиця ──────────────────────────────────────────────────────────────
    hdr = (f"{'p_v':>5} | {'K_res':>5} | {'n':>4} | {'ρ_med':>7} | "
           f"{'ρ_CVaR':>7} | {'R_rb_med':>9} | {'excl%':>6}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    stats = {}  # (p_v, k_res) → dict
    for p_v in PV_VALS:
        for k_res in K_RES_VALS:
            sub = df[(df["p_v_sweep"] == p_v) & (df["K_res_sweep"] == k_res)]
            valid = sub[sub["valid"] == True]
            n_tot = len(sub)
            n_val = len(valid)
            excl  = 100.0 * (n_tot - n_val) / max(n_tot, 1)

            if n_val == 0:
                stats[(p_v, k_res)] = None
                lines.append(f"{p_v:>5.2f} | {k_res:>5} | {0:>4} | {'---':>7} | "
                              f"{'---':>7} | {'---':>9} | {excl:>5.1f}%")
                continue

            arr     = valid["rho"].values
            r_rb    = valid["R_rb"].values
            rho_med = float(np.median(arr))
            rho_cv  = _cvar(arr)
            r_rb_m  = float(np.median(r_rb))

            stats[(p_v, k_res)] = {
                "rho_med": rho_med, "rho_cvar": rho_cv,
                "r_rb": r_rb_m, "excl": excl, "n": n_val,
                "arr": arr,
            }
            lines.append(f"{p_v:>5.2f} | {k_res:>5} | {n_val:>4} | {rho_med:>7.4f} | "
                         f"{rho_cv:>7.4f} | {r_rb_m:>9.4f} | {excl:>5.1f}%")

        lines.append("")  # blank between p_v blocks

    # ── Gains K_res=3 − K_res=0 per p_v ─────────────────────────────────────
    lines.append("=== Gain K_res=3 over K_res=0 ===")
    lines.append("")
    ghdr = f"{'p_v':>5} | {'Δρ_med':>8} | {'Δρ_CVaR':>9} | {'ΔR_rb':>7} | {'ratio Δ_CVaR/Δ_med':>20}"
    lines.append(ghdr)
    lines.append("-" * len(ghdr))

    for p_v in PV_VALS:
        s0 = stats.get((p_v, 0))
        s3 = stats.get((p_v, 3))
        if s0 is None or s3 is None:
            lines.append(f"{p_v:>5.2f} | {'---':>8} | {'---':>9} | {'---':>7} | {'---':>20}")
            continue
        d_med  = s3["rho_med"]  - s0["rho_med"]
        d_cvar = s3["rho_cvar"] - s0["rho_cvar"]
        d_rb   = s3["r_rb"]     - s0["r_rb"]
        ratio  = d_cvar / d_med if abs(d_med) > 1e-6 else float("nan")
        lines.append(f"{p_v:>5.2f} | {d_med:>+8.4f} | {d_cvar:>+9.4f} | {d_rb:>+7.4f} | {ratio:>20.3f}")

    # ── Bootstrap CI на gain ────────────────────────────────────────────────
    lines.append("")
    lines.append("=== Bootstrap 95% CI для ρ_med та ρ_CVaR (B=1500) ===")
    lines.append("")
    for p_v in PV_VALS:
        for k_res in K_RES_VALS:
            s = stats.get((p_v, k_res))
            if s is None:
                continue
            ml, mh = _bci(s["arr"], np.median, rng=RNG)
            cl, ch = _bci(s["arr"], lambda a: _cvar(a), rng=RNG)
            lines.append(f"  p_v={p_v} K_res={k_res}: "
                         f"ρ_med=[{ml:.4f},{mh:.4f}]  "
                         f"ρ_CVaR=[{cl:.4f},{ch:.4f}]")

    # ── Висновок ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=== Висновок Діагностики A ===")
    lines.append("")

    gains_med  = []
    gains_cvar = []
    for p_v in PV_VALS:
        s0 = stats.get((p_v, 0))
        s3 = stats.get((p_v, 3))
        if s0 and s3:
            gains_med.append(s3["rho_med"]  - s0["rho_med"])
            gains_cvar.append(s3["rho_cvar"] - s0["rho_cvar"])

    if len(gains_med) >= 2:
        mono_med  = all(gains_med[i]  <= gains_med[i+1]  for i in range(len(gains_med)-1))
        mono_cvar = all(gains_cvar[i] <= gains_cvar[i+1] for i in range(len(gains_cvar)-1))
        max_gain_med  = max(gains_med)
        max_gain_cvar = max(gains_cvar)

        lines.append(f"Gain ρ_med  монотонно зростає з p_v: {'✓ YES' if mono_med  else '✗ NO'}")
        lines.append(f"Gain ρ_CVaR монотонно зростає з p_v: {'✓ YES' if mono_cvar else '✗ NO'}")
        lines.append(f"Максимальний gain ρ_med:  {max_gain_med:+.4f}  (при p_v={PV_VALS[gains_med.index(max_gain_med)]})")
        lines.append(f"Максимальний gain ρ_CVaR: {max_gain_cvar:+.4f} (при p_v={PV_VALS[gains_cvar.index(max_gain_cvar)]})")
        lines.append("")

        threshold = 0.01  # значущий gain = 1 відсотковий пункт
        if max_gain_cvar > threshold:
            lines.append("→ Висновок (2) ВІРОГІДНИЙ: при сильнішому збої виграш K_res зростає.")
            lines.append("  p_v=0.20 дає занадто слабкий стрес для reserved topology.")
        else:
            lines.append("→ Висновок (1) ВІРОГІДНИЙ: K_res не дає суттєвого виграшу навіть при "
                         "сильнішому збої.")
            lines.append("  Random recovery нівелює виграш backup-ребер — це справжня знахідка.")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# ДІАГНОСТИКА B
# ════════════════════════════════════════════════════════════════════════════

def _run_diag_B() -> pd.DataFrame:
    from src.experiments.runner import _config_id

    C_BACKUP_VALS = [100.0, 300.0, 900.0]
    K_RES = 1
    rows = []

    for cb_idx, c_backup in enumerate(C_BACKUP_VALS):
        cfg = copy.deepcopy(_BASE)
        cfg["network"]["K_res"]  = K_RES
        cfg["disruption"]["p_v"] = 0.20

        eff  = {"topology": "reserved", "K_res": K_RES, "p_v": 0.20, "c_backup": c_backup}
        c_id = f"diagB_cb{int(c_backup):04d}"

        label = f"c_backup={int(c_backup)}"
        print(f"  [B] {label}", end="", flush=True)
        t0 = time.time()

        for run_idx in range(M):
            seed   = (cb_idx + 1) * 200_000 + run_idx + 1
            run_id = f"diagB_{c_id}_{run_idx+1:05d}"
            row, _ = _run_one(cfg, seed, run_id, c_id, c_backup=c_backup)
            row["c_backup_sweep"] = c_backup
            rows.append(row)

        print(f"  ({time.time()-t0:.1f}s)", flush=True)

    return pd.DataFrame(rows)


def _report_diag_B(df: pd.DataFrame) -> str:
    C_BACKUP_VALS = [100.0, 300.0, 900.0]
    RNG = np.random.default_rng(42)

    lines = []
    lines.append("S2 Diagnostics B — Effect of backup edge capacity on resilience")
    lines.append("=" * 60)
    lines.append("Топологія: reserved | K_res=1 | p_v=0.20")
    lines.append(f"M={M} прогонів на конфігурацію | 3 конфігурації × {M} = {3*M} прогонів")
    lines.append("c_backup=100: поточний стандартний режим")
    lines.append("c_backup=300: як ребра рівня 2→3 дерева")
    lines.append("c_backup=900: як ребра рівня 1→2 дерева (потужний резерв)")
    lines.append("")
    lines.append("Питання: чи стає ρ_CVaR помітно вищим, коли backup-ребра товщі?")
    lines.append("")

    # ── Таблиця ──────────────────────────────────────────────────────────────
    hdr = (f"{'c_backup':>9} | {'n':>4} | {'ρ_med':>7} | {'ρ_CVaR':>7} | "
           f"{'R_rb_med':>9} | {'ρ_IQR':>7} | {'excl%':>6}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    stats = {}
    for c_backup in C_BACKUP_VALS:
        sub   = df[df["c_backup_sweep"] == c_backup]
        valid = sub[sub["valid"] == True]
        n_tot = len(sub)
        n_val = len(valid)
        excl  = 100.0 * (n_tot - n_val) / max(n_tot, 1)

        if n_val == 0:
            stats[c_backup] = None
            lines.append(f"{int(c_backup):>9} | {0:>4} | {'---':>7} | "
                         f"{'---':>7} | {'---':>9} | {'---':>7} | {excl:>5.1f}%")
            continue

        arr    = valid["rho"].values
        r_rb   = valid["R_rb"].values
        rho_med = float(np.median(arr))
        rho_cv  = _cvar(arr)
        r_rb_m  = float(np.median(r_rb))
        rho_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))

        stats[c_backup] = {"rho_med": rho_med, "rho_cvar": rho_cv,
                           "r_rb": r_rb_m, "iqr": rho_iqr, "excl": excl,
                           "n": n_val, "arr": arr}
        lines.append(f"{int(c_backup):>9} | {n_val:>4} | {rho_med:>7.4f} | "
                     f"{rho_cv:>7.4f} | {r_rb_m:>9.4f} | {rho_iqr:>7.4f} | {excl:>5.1f}%")

    # ── Gains vs c_backup=100 ────────────────────────────────────────────────
    lines.append("")
    lines.append("=== Приріст відносно c_backup=100 (baseline) ===")
    lines.append("")
    base = stats.get(100.0)
    ghdr = f"{'c_backup':>9} | {'Δρ_med':>8} | {'Δρ_CVaR':>9} | {'ΔR_rb':>7}"
    lines.append(ghdr)
    lines.append("-" * len(ghdr))
    for c_backup in C_BACKUP_VALS:
        s = stats.get(c_backup)
        if s is None or base is None:
            lines.append(f"{int(c_backup):>9} | {'---':>8} | {'---':>9} | {'---':>7}")
            continue
        d_med  = s["rho_med"]  - base["rho_med"]
        d_cvar = s["rho_cvar"] - base["rho_cvar"]
        d_rb   = s["r_rb"]     - base["r_rb"]
        lines.append(f"{int(c_backup):>9} | {d_med:>+8.4f} | {d_cvar:>+9.4f} | {d_rb:>+7.4f}")

    # ── Bootstrap CI ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=== Bootstrap 95% CI (B=1500) ===")
    lines.append("")
    for c_backup in C_BACKUP_VALS:
        s = stats.get(c_backup)
        if s is None:
            continue
        ml, mh = _bci(s["arr"], np.median, rng=RNG)
        cl, ch = _bci(s["arr"], lambda a: _cvar(a), rng=RNG)
        lines.append(f"  c_backup={int(c_backup):>4}: "
                     f"ρ_med=[{ml:.4f},{mh:.4f}]  "
                     f"ρ_CVaR=[{cl:.4f},{ch:.4f}]")

    # ── Висновок ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=== Висновок Діагностики B ===")
    lines.append("")

    s100 = stats.get(100.0)
    s900 = stats.get(900.0)
    if s100 and s900:
        gain_med  = s900["rho_med"]  - s100["rho_med"]
        gain_cvar = s900["rho_cvar"] - s100["rho_cvar"]
        gain_rb   = s900["r_rb"]     - s100["r_rb"]

        lines.append(f"Gain ρ_med  (c_backup 100→900): {gain_med:+.4f}")
        lines.append(f"Gain ρ_CVaR (c_backup 100→900): {gain_cvar:+.4f}")
        lines.append(f"Gain R_rb   (c_backup 100→900): {gain_rb:+.4f}")
        lines.append("")

        threshold = 0.01
        if abs(gain_cvar) > threshold:
            lines.append("→ Висновок (3) ВІРОГІДНИЙ: backup capacity суттєво впливає на ρ_CVaR.")
            lines.append("  Товщі backup-ребра значно підвищують resilience у гірших сценаріях.")
        else:
            lines.append("→ Висновок (1) ВІРОГІДНИЙ: навіть c_backup=900 не дає суттєвого виграшу.")
            lines.append("  Random recovery нівелює виграш capacity — це справжня знахідка.")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ФУНКЦІЯ
# ════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path("results/aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Діагностика A ────────────────────────────────────────────────────────
    print("\n=== ДІАГНОСТИКА A: p_v sweep (K_res={0,3}) ===")
    t_start = time.time()
    df_A = _run_diag_A()
    print(f"A завершена за {time.time()-t_start:.1f}s\n")

    report_A = _report_diag_A(df_A)
    (out_dir / "S2_diag_A_intensity.txt").write_text(report_A, encoding="utf-8")
    print(report_A)

    # ── Діагностика B ────────────────────────────────────────────────────────
    print("\n\n=== ДІАГНОСТИКА B: c_backup sweep (K_res=1, p_v=0.20) ===")
    t_start = time.time()
    df_B = _run_diag_B()
    print(f"B завершена за {time.time()-t_start:.1f}s\n")

    report_B = _report_diag_B(df_B)
    (out_dir / "S2_diag_B_capacity.txt").write_text(report_B, encoding="utf-8")
    print(report_B)

    # ── Фінальний висновок ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ФІНАЛЬНИЙ ВИСНОВОК: збережено у звітах")
    print("results/aggregated/S2_diag_A_intensity.txt")
    print("results/aggregated/S2_diag_B_capacity.txt")


if __name__ == "__main__":
    main()

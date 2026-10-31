"""
Завершення розрахункового блоку: задачі R1-3 і R2-3.

R1-3 (рецензент 1, пункт 3): повернути у вибірку прогони, виключені як
«знищено всі джерела або всіх споживачів». За F(0) > 0 це не невизначена
нормалізація, а результат важкого пошкодження, тому траєкторію можна і
треба симулювати. Виключення за F(0) = 0 зберігається — там нормувати
справді немає на що.

R2-3 (рецензент 2, пункт 3): визначити одиницю повторення Monte Carlo і
окремо оцінити мінливість між реалізаціями топології та між пошкодженнями.
Раннер виводить net_seed і dis_seed з одного master seed, тому у штатному
дизайні обидва чинники змінюються разом. Тут будується схрещений дизайн:
G seed-ів графа × D seed-ів збою.

Скрипт не змінює src/experiments/runner.py — він повторно використовує його
внутрішні функції, щоб числа лишались зіставними зі штатними серіями.

Пише: results/aggregated/closeout_r1_3_inclusive.csv
      results/aggregated/closeout_r2_3_variance.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.experiments.runner import (
    _apply_disruption,
    _create_network,
    _simulate_recovery,
    compute_base_bc_sum,
)
from src.metrics.integral import compute_integral_metrics
from src.network.flow import compute_max_flow

ROOT = Path(__file__).parent
AGG = ROOT / "results" / "aggregated"
AGG.mkdir(parents=True, exist_ok=True)
TOPOS = ["hierarchical", "hub", "reserved", "dense"]


def load_cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


def one_run(cfg: dict, net_seed: int, dis_seed: int, rec_seed: int,
            include_destroyed: bool) -> dict | None:
    """
    Один прогін із явно заданими трьома seed.

    include_destroyed=True — не виключати прогони, де знищено всі джерела
    або всіх споживачів: симулювати їх і дозволити Q(t)=0 на цих кроках.
    Повертає None лише для F(0)=0.
    """
    sim = cfg["simulation"]
    T_max = int(sim.get("T_max", 100))
    omega = float(sim.get("omega", 1.0))
    Q_target_frac = float(sim.get("Q_target_frac", 0.95))
    mu = int(cfg["recovery"].get("mu", 1))

    net0 = _create_network(cfg, seed=net_seed)
    F0 = compute_max_flow(net0)
    if F0 <= 0.0:
        return None

    base_bc = compute_base_bc_sum(net0) if omega < 1.0 else 0.0
    net_d = _apply_disruption(cfg, net0, seed=dis_seed)

    active_s = net_d.suppliers - net_d.failed_nodes
    active_c = net_d.consumers - net_d.failed_nodes
    destroyed = (not active_s) or (not active_c)
    if destroyed and not include_destroyed:
        return None

    Q = _simulate_recovery(net_d, F0, T_max, omega, base_bc,
                           np.random.default_rng(rec_seed),
                           strategy_name=str(cfg["recovery"].get("strategy", "random")),
                           mu=mu,
                           c_base=float(cfg["network"].get("c_base", 100.0)))
    m = compute_integral_metrics(Q, t_e=1, Q_target=Q_target_frac * Q[0])
    return dict(F0=F0, rho=m["rho"], R_rb=m["R_rb"], R_inf=m["R_inf"],
                destroyed=bool(destroyed),
                non_recovered=bool(Q[-1] < Q_target_frac * Q[0]))


def seeds_from_master(master: int) -> tuple[int, int, int]:
    """Відтворює схему трьох під-seed з runner.py."""
    g = np.random.default_rng(master)
    return (int(g.integers(0, 2**31)), int(g.integers(0, 2**31)),
            int(g.integers(0, 2**31)))


# ════════════════════ R1-3: повернення виключених прогонів ════════════════════
def task_r1_3() -> pd.DataFrame:
    cfg0 = load_cfg("series_S1")
    s1 = pd.read_csv(ROOT / "results" / "raw" / "S1" / "raw_results.csv")
    recs = []
    for topo in TOPOS:
        cfg = {**cfg0, "network": {**cfg0["network"], "topology": topo}}
        for master in sorted(s1[s1.topology == topo]["seed"].unique()):
            ns, ds, rs = seeds_from_master(int(master))
            r = one_run(cfg, ns, ds, rs, include_destroyed=True)
            if r is None:
                recs.append(dict(topology=topo, seed=int(master), F0_zero=True))
                continue
            recs.append(dict(topology=topo, seed=int(master), F0_zero=False, **r))
    df = pd.DataFrame(recs)
    df.to_csv(AGG / "closeout_r1_3_inclusive.csv", index=False)
    return df


# ════════════════════ R2-3: декомпозиція дисперсії ════════════════════
def task_r2_3(n_graph: int = 20, n_dis: int = 20) -> pd.DataFrame:
    """Схрещений дизайн: n_graph seed-ів графа × n_dis seed-ів збою."""
    cfg0 = load_cfg("series_S1")
    recs = []
    for topo in TOPOS:
        cfg = {**cfg0, "network": {**cfg0["network"], "topology": topo}}
        for gi in range(n_graph):
            net_seed = 100_000 + gi
            for di in range(n_dis):
                dis_seed = 900_000 + di          # та сама сітка збоїв для всіх графів
                rec_seed = 500_000 + gi * 1000 + di
                r = one_run(cfg, net_seed, dis_seed, rec_seed, include_destroyed=False)
                if r is None:
                    continue
                recs.append(dict(topology=topo, graph_id=gi, dis_id=di, **r))
    df = pd.DataFrame(recs)
    df.to_csv(AGG / "closeout_r2_3_variance.csv", index=False)
    return df


if __name__ == "__main__":
    print("R1-3: повторний прогін S1 з поверненням виключених...")
    d1 = task_r1_3()
    print(f"  рядків: {len(d1)}, знищених джерел/споживачів: "
          f"{int(d1.get('destroyed', pd.Series(dtype=bool)).sum())}")
    print("R2-3: схрещений дизайн 20 графів × 20 збоїв × 4 топології...")
    d2 = task_r2_3()
    print(f"  рядків: {len(d2)}")
    print("Готово.")


# ════════════════════ R1-5: ремонтний ресурс проти масштабу ════════════════════
def task_r1_5(n_seeds: int = 100) -> pd.DataFrame:
    """
    S5-подібний дизайн із двома режимами ремонту на тих самих seed:
      mu = 1          — як у штатній серії S5 (сталий абсолютний ресурс);
      mu = round(N/30) — ресурс пропорційний масштабу (сталий відносний).
    Якщо рейтинг топологій однаковий в обох режимах, ефект масштабу не є
    артефактом сталого ремонтного ресурсу (зауваження рецензента 1, пункт 5).
    """
    cfg0 = load_cfg("series_S5")
    recs = []
    for N in [30, 50, 80, 120]:
        for mu in [1, max(1, round(N / 30))]:
            for topo in TOPOS:
                cfg = {**cfg0,
                       "network": {**cfg0["network"], "topology": topo, "N": N},
                       "recovery": {**cfg0["recovery"], "mu": mu}}
                for s in range(n_seeds):
                    ns, ds, rs = seeds_from_master(700_000 + s)
                    r = one_run(cfg, ns, ds, rs, include_destroyed=False)
                    if r is None:
                        continue
                    recs.append(dict(topology=topo, N=N, mu=mu, seed=s, **r))
    df = pd.DataFrame(recs)
    df.to_csv(AGG / "closeout_r1_5_mu_scaling.csv", index=False)
    return df

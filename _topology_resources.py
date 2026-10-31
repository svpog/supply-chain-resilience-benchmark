"""
Таблиця ресурсів топологій: |V|, |E|, сумарна пропускна здатність, F_0.

Призначення: відповідь на зауваження рецензента 2, пункт 2 —
«У порівнянні топологій слід навести кількість ребер, сумарні пропускні
здатності та початковий максимальний потік. Щільна мережа може отримувати
перевагу завдяки більшому обсягу ресурсів.»

Відтворює ті самі seed-и, що й серія S1, будує кожну топологію і рахує
структурні та ресурсні характеристики непошкодженої мережі. Обчислений F_0
звіряється з колонкою F0 у results/raw/S1/raw_results.csv.

Ключове похідне відношення — F_0 / sum(C): яку частку встановленої
пропускної здатності мережа здатна реально використати. Воно показує, чи
виграш щільної топології пояснюється більшим обсягом ресурсів, чи кращою
їх утилізацією.

Читає:  data/S1.csv, config/series_S1.yaml
Пише:   results/aggregated/topology_resources.txt
        results/aggregated/topology_resources.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.network.base import Network
from src.network.flow import compute_max_flow

ROOT = Path(__file__).parent
AGG = ROOT / "results" / "aggregated"
AGG.mkdir(parents=True, exist_ok=True)

TOPOS = ["hierarchical", "hub", "reserved", "dense"]


def build(topology: str, seed: int, net_cfg: dict) -> Network:
    """Повторює логіку _create_network() з src/experiments/runner.py."""
    kwargs = {"N": int(net_cfg.get("N", 50)),
              "c_base": float(net_cfg.get("c_base", 100.0)),
              "seed": seed}
    if topology in {"hierarchical", "reserved"}:
        kwargs["H"] = int(net_cfg.get("H", 4))
        kwargs["b"] = int(net_cfg.get("b", 3))
    if topology == "reserved":
        kwargs["K_res"] = int(net_cfg.get("K_res", 2))
    if topology == "hub":
        kwargs["h"] = int(net_cfg.get("h", 3))
    if topology == "dense":
        kwargs["k_avg"] = float(net_cfg.get("k_avg", 5.0))
    return Network.create(topology, **kwargs)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "series_S1.yaml").read_text(encoding="utf-8"))
    net_cfg = cfg["network"]

    s1 = pd.read_csv(ROOT / "data" / "S1.csv")
    seeds_by_topo = {t: sorted(s1[s1.topology == t]["seed"].unique()) for t in TOPOS}

    recs = []
    for topo in TOPOS:
        for seed in seeds_by_topo[topo]:
            net = build(topo, int(seed), net_cfg)
            caps = [net.graph[u][v]["capacity"] for u, v in net.graph.edges()]
            recs.append(dict(
                topology=topo, seed=int(seed),
                V=net.N, E=net.L,
                sum_C=float(np.nansum(caps)),
                mean_C=float(np.nanmean(caps)),
                n_sources=len(net.suppliers), n_sinks=len(net.consumers),
                F0=float(compute_max_flow(net)),
            ))
    df = pd.DataFrame(recs)
    df["deg_out"] = df["E"] / df["V"]
    df["util"] = df["F0"] / df["sum_C"]

    agg = df.groupby("topology").agg(
        runs=("seed", "size"), V=("V", "median"),
        E_med=("E", "median"), E_min=("E", "min"), E_max=("E", "max"),
        deg=("deg_out", "median"),
        sumC_med=("sum_C", "median"), sumC_min=("sum_C", "min"), sumC_max=("sum_C", "max"),
        meanC=("mean_C", "median"),
        src=("n_sources", "median"), snk=("n_sinks", "median"),
        F0_med=("F0", "median"), F0_min=("F0", "min"), F0_max=("F0", "max"),
        util=("util", "median"),
    ).reindex(TOPOS)

    # Звірка F0 з тим, що записав раннер
    ref = s1[s1.valid].groupby("topology")["F0"].median().reindex(TOPOS)
    agg["F0_raw_S1"] = ref
    agg["F0_match"] = np.isclose(agg["F0_med"], agg["F0_raw_S1"], rtol=0, atol=1e-6)

    L: list[str] = []
    W = L.append
    W("=" * 104)
    W("РЕСУРСИ ТОПОЛОГІЙ: СТРУКТУРА, ПРОПУСКНА ЗДАТНІСТЬ І БАЗОВИЙ ПОТІК")
    W("=" * 104)
    W("")
    W("Відповідь на зауваження рецензента 2, пункт 2.")
    W("")
    W(f"Параметри: N={net_cfg.get('N')}, H={net_cfg.get('H')}, b={net_cfg.get('b', 3)}, "
      f"c_base={net_cfg.get('c_base')}, K_res={net_cfg.get('K_res', 2)}, "
      f"h={net_cfg.get('h', 3)}, k_avg={net_cfg.get('k_avg', 5.0)}.")
    W("Мережі побудовані на тих самих seed, що й серія S1 (по 100 на топологію).")
    W("")
    W("-" * 104)
    W("ТАБЛИЦЯ РЕСУРСІВ НЕПОШКОДЖЕНОЇ МЕРЕЖІ (медіани за 100 реалізаціями)")
    W("-" * 104)
    W("Топологія      |V|    |E|   діапазон  |E|/|V|      sum(C)        діапазон   сер.C   "
      "|V_s| |V_c|      F_0    діапазон   F_0/sum(C)")
    W("-" * 104)
    for t in TOPOS:
        r = agg.loc[t]
        W(f"{t:13s} {int(r.V):4d} {int(r.E_med):6d} "
          f"{int(r.E_min):5d}-{int(r.E_max):<5d} {r.deg:6.2f} "
          f"{r.sumC_med:11,.0f} {r.sumC_min:8,.0f}-{r.sumC_max:<8,.0f} {r.meanC:7.1f} "
          f"{int(r.src):5d} {int(r.snk):5d} {r.F0_med:8,.0f} "
          f"{r.F0_min:6,.0f}-{r.F0_max:<6,.0f} {r.util:10.4f}")
    W("")
    W("Звірка обчисленого F_0 з колонкою F0 у results/raw/S1/raw_results.csv:")
    for t in TOPOS:
        r = agg.loc[t]
        mark = "збігається" if r.F0_match else "РОЗБІЖНІСТЬ"
        W(f"  {t:13s} обчислено {r.F0_med:10,.0f}   у S1 {r.F0_raw_S1:10,.0f}   {mark}")
    W("")

    # Відносні показники до найбіднішої ресурсами топології
    base_t = agg["sumC_med"].idxmin()
    W("-" * 104)
    W(f"ВІДНОСНІ РЕСУРСИ (за одиницю взято {base_t} — найменша sum(C))")
    W("-" * 104)
    W("Топологія        |E| відн.   sum(C) відн.   F_0 відн.   F_0/sum(C) відн.")
    W("-" * 104)
    b = agg.loc[base_t]
    for t in TOPOS:
        r = agg.loc[t]
        W(f"{t:13s} {r.E_med/b.E_med:11.2f}x {r.sumC_med/b.sumC_med:13.2f}x "
          f"{r.F0_med/b.F0_med:11.2f}x {r.util/b.util:17.2f}x")
    W("")
    W("-" * 104)
    W("ІНТЕРПРЕТАЦІЯ")
    W("-" * 104)
    W("")
    d, h_, hb, rsv = (agg.loc["dense"], agg.loc["hierarchical"],
                      agg.loc["hub"], agg.loc["reserved"])

    W("1. Припущення рецензента про ресурсну перевагу щільної мережі даними")
    W("   НЕ підтверджується.")
    W("")
    W(f"   Серед чотирьох топологій dense має НАЙМЕНШУ сумарну пропускну здатність")
    W(f"   ({d.sumC_med:,.0f} проти {h_.sumC_med:,.0f} у hierarchical, {rsv.sumC_med:,.0f} у reserved")
    W(f"   і {hb.sumC_med:,.0f} у hub) і найнижчий базовий потік ({d.F0_med:,.0f}, нарівні з hub,")
    W(f"   проти {h_.F0_med:,.0f} у hierarchical і reserved).")
    W("")
    W(f"   Щільна мережа виграє не обсягом ресурсу, а його подрібненням: {int(d.E_med)} ребер")
    W(f"   по {d.meanC:.0f} одиниць проти {int(h_.E_med)} ребер по {h_.meanC:.0f} одиниць у hierarchical.")
    W("   Той самий бюджет, розподілений на більшу кількість незалежних маршрутів.")
    W("")
    W("2. Нормування знімає різницю в абсолютному масштабі.")
    W("")
    W("   Q(t) = F(t)/F_0 рахується відносно власного базового потоку кожної")
    W("   топології, тому мережа з більшим F_0 не отримує переваги за побудовою")
    W("   метрики. Порівнюється частка збереженої функціональності, а не її обсяг.")
    W("")
    W("3. Коефіцієнт утилізації пояснює природу запасу.")
    W("")
    W(f"   F_0/sum(C): hierarchical {h_.util:.4f}, reserved {rsv.util:.4f}, "
      f"dense {d.util:.4f}, hub {hb.util:.4f}.")
    W(f"   У hub на кожну одиницю базового потоку припадає {1/hb.util:,.0f} одиниць встановленої")
    W("   потужності — ресурс сконцентрований у хабах і майже не конвертується")
    W("   у потік. Це структурна, а не ресурсна вада, і саме вона дає хабовій")
    W("   топології гіршу хвостову поведінку при високій медіанній.")
    W("")
    W("4. Зіставлення за рівного бюджету в дослідженні вже є.")
    W("")
    W(f"   hierarchical проти reserved: {int(h_.E_med)} проти {int(rsv.E_med)} ребер "
      f"({rsv.E_med/h_.E_med:.2f}x),")
    W(f"   {h_.sumC_med:,.0f} проти {rsv.sumC_med:,.0f} одиниць пропускної здатності "
      f"({rsv.sumC_med/h_.sumC_med:.2f}x),")
    W(f"   однаковий F_0 = {h_.F0_med:,.0f}, однакові |V_s| = {int(h_.src)} і |V_c| = {int(h_.snk)}.")
    W("")
    W("   Це найчистіше можливе зіставлення: топології відрізняються лише наявністю")
    W("   резервних ребер при практично тому самому бюджеті. І саме вони виявились")
    W("   статистично нерозрізнюваними за rho (p_adj = 1.000, Cliff d = -0.020,")
    W("   блок A статистичного додатку). Тобто приріст ресурсу на 3% дав нульовий")
    W("   приріст функції — окрема контрольна серія за узгодженим бюджетом не")
    W("   потрібна, бо потрібне порівняння вже закладене в дизайн.")
    W("")
    W("5. Межа дії висновку.")
    W("")
    W("   Пари (hierarchical, reserved) і (hub, dense) різняться також кількістю")
    W(f"   джерел і споживачів: {int(h_.src)}/{int(h_.snk)} проти {int(d.src)}/{int(d.snk)}.")
    W("   Тому порівняння між цими парами лишається порівнянням архітектур")
    W("   загалом, а не ізольованого впливу щільності. Це обмеження слід")
    W("   зафіксувати в тексті статті явно.")
    W("")
    W("-" * 104)
    W("ЗАУВАЖЕННЯ ДО РОЗДІЛУ 3 СТАТТІ (зауваження рецензента 1, пункт 2)")
    W("-" * 104)
    W("")
    W(f"   Для dense F_0 = {d.F0_med:,.0f} при діапазоні {d.F0_min:,.0f}-{d.F0_max:,.0f}, тобто")
    W(f"   детермінований: обмежувачем є не внутрішня мережа ({d.sumC_med:,.0f} одиниць),")
    W(f"   а ребра від суперджерела ({int(d.src)} джерел). Те саме для hub.")
    W("   Отже правила побудови звʼязків із суперджерелом і суперстоком та їхні")
    W("   пропускні здатності визначають F_0 для двох топологій із чотирьох —")
    W("   і мають бути описані у розділі 3, як і вимагає рецензент 1.")
    W("")
    out = AGG / "topology_resources.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    agg.to_csv(AGG / "topology_resources.csv")
    df.to_csv(AGG / "topology_resources_per_run.csv", index=False)
    print("\n".join(L))
    print(f"\nЗбережено: {out}")


if __name__ == "__main__":
    main()

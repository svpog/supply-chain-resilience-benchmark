"""
Точка входу для запуску серій експериментів.

Реалізує: CLI-інтерфейс для запуску серій S1–S6
(див. docs/Math_Spec_Стаття_1.docx, розділ 10 та протоколи серій).

Приклад використання:
    python main.py --series S1 --runs 5
    python main.py --series S2 --runs 100
"""

import argparse
import sys


VALID_SERIES = {
    "S1", "S2", "S3", "S4", "S5", "S6",
    "R1", "R2", "D1", "R2_prime", "R3", "R4", "D2a", "D2b", "R5",
    "M0", "M1", "M2", "M3", "I1", "M4", "M4ext", "M5ceiling",
}


def parse_args() -> argparse.Namespace:
    """Парсить аргументи командного рядка."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Запуск серій експериментів з resilience мереж поставок.\n"
            "Результати зберігаються у папці results/ за серіями."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Приклади:\n"
            "  python main.py --series S1 --runs 5     # smoke-тест\n"
            "  python main.py --series S1 --runs 100   # повна серія\n"
            "  python main.py --series S4 --runs 100   # вплив типу збоїв\n"
            "  python main.py --series R1 --runs 100   # стратегії recovery, стаття 3\n"
            "  python main.py --series R2 --runs 100   # K_res × strategy, стаття 3\n"
            "  python main.py --series D1              # D1 node-level recovery diagnostic\n"
            "  python main.py --series R2_prime        # c_backup sensitivity, стаття 3\n"
            "  python main.py --series R3              # recovery speed μ sensitivity, стаття 3\n"
            "  python main.py --series R4              # strategy × disruption type, стаття 3\n"
            "  python main.py --series D2a             # canonical edge degradation diagnostic\n"
            "  python main.py --series D2b             # enhanced edge degradation diagnostic\n"
            "  python main.py --series R5              # scale robustness across N, стаття 3\n"
            "  python main.py --series M0              # Article 4 bridge: local visibility\n"
            "  python main.py --series M1              # Article 4 multi-agent baseline\n"
            "  python main.py --series M2              # Article 4 K-scaling\n"
            "  python main.py --series M3              # Article 4 coordination mechanisms\n"
            "  python main.py --series I1              # Article 4 information value paired design\n"
            "  python main.py --series M4              # Article 4 scaling, best multi-agent\n"
            "  python main.py --series M5ceiling       # Article 4 RL ceiling check\n"
            "  python main.py --series M4ext           # Article 4 M4 extension N=160,200\n"
        ),
    )
    parser.add_argument(
        "--series",
        type=str,
        required=True,
        choices=sorted(VALID_SERIES),
        metavar="SERIES",
        help="Назва серії: S1–S6, R1–R5, D1/D2 або M0–M3/I1",
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=False,
        metavar="N",
        help="Кількість прогонів Monte Carlo. Якщо не задано — береться з config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        metavar="PATH",
        help="Папка для результатів (дефолт: results). Потрібно для regression-тестів.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=False,
        metavar="TOPOLOGY",
        help="Опційний фільтр topology для smoke-тестів, напр. --topology hierarchical.",
    )
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Smoke-check режим: запускає реальні прогони, але друкує додаткову позначку.",
    )
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="Alias для M1 smoke test; запускає реальні прогони з додатковою позначкою.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        required=False,
        metavar="LIST",
        help="Кома-розділений фільтр стратегій для M-серій smoke-тестів.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        required=False,
        metavar="LIST",
        help="Кома-розділений фільтр coordination modes для M-серій, напр. --modes token,auction.",
    )
    parser.add_argument(
        "--K",
        type=str,
        required=False,
        metavar="LIST",
        help="Кома-розділений фільтр K_agents для M-серій, напр. --K 1,3.",
    )
    parser.add_argument(
        "--N",
        type=str,
        required=False,
        metavar="LIST",
        help="Кома-розділений фільтр N для M4 smoke-тестів, напр. --N 30.",
    )
    return parser.parse_args()


def main() -> None:
    """Головна функція: парсить аргументи та запускає серію."""
    args = parse_args()

    multi_agent_series = {"M1", "M2", "M3", "I1", "M4", "M4ext", "M5ceiling"}
    if args.series in multi_agent_series:
        from src.experiments.multi_agent_runner import load_config as load_m_config
        from src.experiments.multi_agent_runner import run_paired_series as run_m_paired_series
        from src.experiments.multi_agent_runner import run_series as run_m_series
    else:
        from src.experiments.runner import load_config, run_series

    runs = args.runs
    if runs is None:
        cfg = load_m_config(args.series) if args.series in multi_agent_series else load_config(args.series)
        if args.series in multi_agent_series and "M_schedule" in cfg:
            runs = None
        else:
            constants = cfg.get("constants", {})
            runs = cfg.get(
                "n_runs",
                constants.get("M_test", constants.get("M", cfg.get("simulation", {}).get("n_runs"))),
            )
        if runs is None:
            if not (args.series in multi_agent_series and "M_schedule" in cfg):
                print("Помилка: --runs не задано, і config не містить n_runs.")
                sys.exit(1)
        else:
            runs = int(runs)

    if runs is not None and runs <= 0:
        print(f"Помилка: --runs має бути додатнім числом, отримано {runs}.")
        sys.exit(1)

    if args.dry_run_check or args.smoke_check:
        print("Smoke-check режим увімкнено: результати все одно будуть записані.")

    runs_label = str(runs) if runs is not None else "M_schedule"
    print(f"Запуск серії {args.series} з {runs_label} прогонів на конфігурацію...")
    if args.series in multi_agent_series:
        cfg = load_m_config(args.series)
        strategies_filter = (
            [item.strip() for item in args.strategies.split(",") if item.strip()]
            if args.strategies else None
        )
        K_filter = (
            [int(item.strip()) for item in args.K.split(",") if item.strip()]
            if args.K else None
        )
        coordination_filter = (
            [item.strip() for item in args.modes.split(",") if item.strip()]
            if args.modes else None
        )
        N_filter = (
            [int(item.strip()) for item in args.N.split(",") if item.strip()]
            if args.N else None
        )
        if cfg.get("paired_design", False):
            run_m_paired_series(
                cfg,
                n_runs=runs,
                output_dir=args.output_dir,
                topology_filter=args.topology,
            )
        else:
            run_m_series(
                cfg,
                n_runs=runs,
                output_dir=args.output_dir,
                topology_filter=args.topology,
                strategies_filter=strategies_filter,
                K_filter=K_filter,
                coordination_filter=coordination_filter,
                N_filter=N_filter,
            )
    else:
        run_series(
            series=args.series,
            n_runs=runs,
            output_dir=args.output_dir,
            topology_filter=args.topology,
        )
    if args.series in multi_agent_series:
        print(
            f"Готово. Результати у папках "
            f"{args.output_dir}/raw/article4/{args.series}/ та {args.output_dir}/aggregated/article4/"
        )
    else:
        print(
            f"Готово. Результати у папках "
            f"{args.output_dir}/raw/{args.series}/ та {args.output_dir}/aggregated/{args.series}/"
        )


if __name__ == "__main__":
    main()

"""
Unit-тести M1 MultiAgentRunner.

Реалізує: Протокол M1, §5
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.experiments.multi_agent_runner import (
    load_config,
    load_m1_config,
    run_m1_series,
    run_paired_series,
    run_series,
)
from src.network.base import Network
from src.recovery.agents.local_greedy_agent import LocalGreedyAgent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m1_config_loads() -> None:
    """series_M1.yaml має очікувані ключові параметри."""
    cfg = load_m1_config()

    assert cfg["series"] == "M1"
    assert cfg["constants"]["K_agents"] == 3
    assert cfg["constants"]["r_visibility"] == 2
    assert cfg["random_seed_base"] == 4400


def test_multi_agent_runner_writes_files(tmp_path) -> None:
    """M1 runner пише файли у results/raw/article4/M1/."""
    raw = run_m1_series(
        n_runs=1,
        output_dir=tmp_path,
        topology_filter="hierarchical",
        strategies_filter=["random_agent"],
    )

    assert len(raw) == 1
    assert (tmp_path / "raw" / "article4" / "M1" / "raw_results.csv").exists()
    assert (tmp_path / "raw" / "article4" / "M1" / "Q_curves.parquet").exists()
    assert (tmp_path / "raw" / "article4" / "M1" / "run_metadata.json").exists()
    assert (tmp_path / "aggregated" / "article4" / "M1_aggregated_results.csv").exists()


def test_multi_agent_runner_isolates_M0(tmp_path) -> None:
    """Запуск M1 у tmp output не змінює M0 raw_results.csv."""
    m0_raw = Path("results/raw/article4/M0/raw_results.csv")
    before = _sha256(m0_raw)

    run_m1_series(
        n_runs=1,
        output_dir=tmp_path,
        topology_filter="dense",
        strategies_filter=["coord_greedy"],
    )

    assert _sha256(m0_raw) == before


def test_i1_paired_config_loads() -> None:
    """series_I1.yaml оголошує paired design r=2 vs r=∞."""
    cfg = load_config("I1")

    assert cfg["series"] == "I1"
    assert cfg["paired_design"] is True
    assert cfg["constants"]["coordination_mode"] == "token"
    assert cfg["varying"]["r_visibility"] == [2, None]


def test_i1_paired_run_id_consistency(tmp_path) -> None:
    """Кожен paired_run_id має рівно два рядки: r2 і r_inf."""
    cfg = load_config("I1")
    raw = run_paired_series(cfg, n_runs=2, output_dir=tmp_path, topology_filter="hierarchical")

    counts = raw.groupby("paired_run_id").size()
    labels = raw.groupby("paired_run_id")["r_label"].apply(set)

    assert len(raw) == 4
    assert set(counts) == {2}
    assert all(value == {"r2", "r_inf"} for value in labels)
    assert (tmp_path / "raw" / "article4" / "I1" / "raw_results.csv").exists()
    assert (tmp_path / "aggregated" / "article4" / "I1_aggregated_results.csv").exists()


def test_i1_paired_same_network_and_disruption(tmp_path) -> None:
    """Обидва r у парі мають однаковий network_hash і disruption_hash."""
    cfg = load_config("I1")
    raw = run_paired_series(cfg, n_runs=2, output_dir=tmp_path, topology_filter="dense")

    for _, group in raw.groupby("paired_run_id"):
        assert group["network_hash"].nunique() == 1
        assert group["disruption_hash"].nunique() == 1


def test_i1_paired_net_copy_isolation(tmp_path) -> None:
    """r=∞ не залежить від порядку запуску r-гілок у paired episode."""
    cfg_forward = load_config("I1")
    cfg_reverse = load_config("I1")
    cfg_reverse["varying"] = dict(cfg_reverse["varying"])
    cfg_reverse["varying"]["r_visibility"] = [None, 2]

    forward = run_paired_series(cfg_forward, n_runs=1, output_dir=tmp_path / "forward", topology_filter="hub")
    reverse = run_paired_series(cfg_reverse, n_runs=1, output_dir=tmp_path / "reverse", topology_filter="hub")

    f_inf = forward[forward["r_label"] == "r_inf"].iloc[0]
    r_inf = reverse[reverse["r_label"] == "r_inf"].iloc[0]
    assert f_inf["network_hash"] == r_inf["network_hash"]
    assert f_inf["disruption_hash"] == r_inf["disruption_hash"]
    assert f_inf["rho"] == r_inf["rho"]


def test_i1_r_inf_observation_is_full_visibility() -> None:
    """r=None означає повну видимість для агента."""
    net = Network.create("hierarchical", N=30, seed=42)
    failed = next(iter(set(net.graph.nodes()) - set(net.suppliers) - set(net.consumers)))
    net.set_node_state(failed, "failed")
    agent = LocalGreedyAgent(agent_id=0, r=None, network=net)

    obs = agent.observe(net, failed_nodes={failed})

    assert obs["visible_nodes"] == set(net.graph.nodes())
    assert obs["visible_failed"] == {failed}


def test_i1_tmp_run_does_not_modify_M3_results(tmp_path) -> None:
    """I1 у tmp output не торкається вже прийнятих M3 raw results."""
    m3_raw = Path("results/raw/article4/M3/raw_results.csv")
    before = _sha256(m3_raw)

    cfg = load_config("I1")
    run_paired_series(cfg, n_runs=1, output_dir=tmp_path, topology_filter="reserved")

    assert _sha256(m3_raw) == before


def test_m4_config_locked() -> None:
    """M4 фіксує best-agent конфігурацію з M0-M3/I1."""
    cfg = load_config("M4")

    assert cfg["series"] == "M4"
    assert cfg["constants"]["agent_strategy"] == "coord_greedy"
    assert cfg["constants"]["coordination_mode"] == "token"
    assert cfg["constants"]["K_agents"] == 3
    assert cfg["constants"]["r_visibility"] == 2
    assert cfg["disruption"]["type"] == "node_failure"
    assert cfg["disruption"]["p_v"] == 0.20


def test_m4_M_schedule() -> None:
    """M4 бере M=200 для N<=50 і M=100 для N>=80."""
    cfg = load_config("M4")
    schedule = {int(key): int(value) for key, value in cfg["M_schedule"].items()}

    assert schedule == {30: 200, 50: 200, 80: 100, 120: 100}


def test_m4_N_scaling_runs(tmp_path) -> None:
    """M4 runner запускає окремі scaling cells для різних N."""
    cfg = load_config("M4")
    raw = run_series(
        cfg,
        n_runs=1,
        output_dir=tmp_path,
        topology_filter="hierarchical",
        N_filter=[30, 80],
    )

    assert len(raw) == 2
    assert set(raw["N"]) == {30, 80}
    assert set(raw["agent_strategy"]) == {"coord_greedy"}
    assert set(raw["coordination_mode"]) == {"token"}
    assert (tmp_path / "raw" / "article4" / "M4" / "raw_results.csv").exists()
    assert (tmp_path / "aggregated" / "article4" / "M4_aggregated_results.csv").exists()


def test_m4_benchmark_compat() -> None:
    """R5 benchmark сумісний з M4 за N, topology, strategy і disruption params."""
    desc = pd.read_csv("results/aggregated/article3/R5_descriptive_statistics.csv")
    raw = pd.read_csv("results/raw/R5/raw_results.csv")

    assert set(desc["N"]) == {30, 50, 80, 120}
    assert {"hierarchical", "dense"} <= set(desc["topology"])
    assert {"flow_greedy", "random"} <= set(desc["recovery_strategy"])

    sub = raw[
        raw["topology"].isin(["hierarchical", "dense"])
        & raw["N"].isin([30, 50, 80, 120])
        & raw["recovery_strategy"].isin(["flow_greedy", "random"])
    ]
    assert set(sub["disruption_type"].dropna()) == {"node_failure"}
    assert set(sub["p_v"].dropna()) == {0.20}
    assert set(sub["T_max"].dropna()) == {100}
    assert set(sub["mu"].dropna()) == {1}


def test_m4_N50_reproduces_I1() -> None:
    """N=50 best-agent baseline узгоджений між I1 r=2 і M2 K=3."""
    i1 = pd.read_csv("results/aggregated/article4/I1_descriptive_statistics.csv")
    m2 = pd.read_csv("results/aggregated/article4/M2_descriptive_statistics.csv")

    for topology in ["hierarchical", "dense"]:
        i1_value = float(
            i1[(i1["topology"] == topology) & (i1["r_label"] == "r2")]["rho_CVaR_0.10"].iloc[0]
        )
        m2_value = float(
            m2[(m2["topology"] == topology) & (m2["K_agents"] == 3)]["rho_CVaR_0.10"].iloc[0]
        )
        assert abs(i1_value - m2_value) <= 0.010


def test_m5ceiling_config_loads_and_locks_design() -> None:
    """M5-ceiling config фіксує hier N=50, K=3, r=2, token і seed split."""
    cfg = load_config("M5ceiling")

    assert cfg["series"] == "M5ceiling"
    assert cfg["network"]["N"] == 50
    assert cfg["constants"]["K_agents"] == 3
    assert cfg["constants"]["r_visibility"] == 2
    assert cfg["constants"]["coordination_mode"] == "token"
    assert cfg["constants"]["train_seed_base"] == 7000
    assert cfg["constants"]["test_seed_base"] == 8000
    assert "offline_oracle" in cfg["varying"]["agent_strategy"]
    assert "rl_policy" in cfg["varying"]["agent_strategy"]

"""
Тести runner-а серій: метрики, схема виводу, відтворюваність.

Реалізує: Math Spec 11.5 (integration test) та 11.6 (reproducibility test).

Тест test_metrics_match_Q_curves є ключовим: він перевіряє, що метрики у
raw_results відповідають Q-кривій у Q_curves.parquet для кожного прогону.
"""

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from src.experiments.runner import (
    _apply_combo,
    _apply_invariant_edge_degradation,
    _apply_r2_invariant_disruption,
    _build_sweep_combinations,
    _config_id,
    _create_network,
    _d2_seed_and_paired_id,
    _effective_combo,
    _make_node_recovery_log_entry,
    _m0_seed_and_paired_id,
    _r3_seed_and_paired_id,
    _r5_seed_and_paired_id,
    _run_single,
    load_config,
    run_series,
)
from src.metrics.integral import compute_integral_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. Юніт-тести метрик: перевіряємо, що runner передає t_e=1 і Q_target=0.95
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegralMetricsPassedCorrectly:
    """
    Синтетичні Q-криві з відомими відповідями.

    runner.py викликає compute_integral_metrics(Q_curve, t_e=1, Q_target=0.95*Q[0]).
    Тут ми тестуємо саму функцію напряму, щоб переконатись у правильності
    передачі параметрів.
    """

    KNOWN_Q = [1.0, 0.30, 0.50, 0.70, 0.95, 1.0]

    def test_rho_equals_mean(self):
        """ρ = mean(Q_curve) для всього вікна t=0..T_max."""
        Q = self.KNOWN_Q
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        expected = sum(Q) / len(Q)
        assert math.isclose(m["rho"], expected, rel_tol=1e-9)

    def test_R_rb_equals_min(self):
        """R_rb = min Q(t)."""
        Q = self.KNOWN_Q
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        assert math.isclose(m["R_rb"], 0.30, rel_tol=1e-9)

    def test_R_rp_speed_of_recovery(self):
        """R_rp = (Q[t_r] - Q[t_e]) / (t_r - t_e).

        Q[4]=0.95 >= 0.95 → t_r=4, t_e=1.
        R_rp = (0.95 - 0.30) / (4 - 1) = 0.65 / 3.
        """
        Q = self.KNOWN_Q
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        expected = (0.95 - 0.30) / 3
        assert math.isclose(m["R_rp"], expected, rel_tol=1e-9)

    def test_R_inf_equals_Q_tr_over_Q0(self):
        """R_∞ = Q[t_r] / Q[0] = 0.95 / 1.0 = 0.95."""
        Q = self.KNOWN_Q
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        assert math.isclose(m["R_inf"], 0.95, rel_tol=1e-9)

    def test_undisrupted_R_rp_is_nan(self):
        """Нерушена мережа: Q≡1 → Q(t_e)=1 ≥ 0.95 → t_r=t_e → R_rp=NaN."""
        Q = [1.0, 1.0, 1.0, 1.0, 1.0]
        m = compute_integral_metrics(Q, t_e=1)
        assert math.isnan(m["R_rp"])

    def test_undisrupted_R_inf_is_one(self):
        """Нерушена мережа: R_∞ = Q[t_r]/Q[0] = 1.0."""
        Q = [1.0, 1.0, 1.0, 1.0, 1.0]
        m = compute_integral_metrics(Q, t_e=1)
        assert math.isclose(m["R_inf"], 1.0)

    def test_no_recovery_R_rp_zero(self):
        """Мережа застрягла на Q=0.3 (не відновилась): t_r=T_max → R_rp≈0."""
        Q = [1.0, 0.3, 0.3, 0.3, 0.3]
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        # t_r=4 (T_max), Q[t_r]=Q[t_e]=0.3 → R_rp=(0.3-0.3)/(4-1)=0
        assert math.isclose(m["R_rp"], 0.0, abs_tol=1e-9)

    def test_no_recovery_R_inf_below_one(self):
        """Мережа не відновилась: R_∞ = Q[T_max] / Q[0] < 1."""
        Q = [1.0, 0.3, 0.3, 0.3, 0.3]
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        assert math.isclose(m["R_inf"], 0.3, rel_tol=1e-9)

    def test_t_e_respected(self):
        """t_r шукається починаючи з t_e, не з 0."""
        # Q[0]=1.0 >= 0.95, але t_e=1, тому t_r шукається з t=1
        Q = [1.0, 0.5, 0.95, 1.0]
        m = compute_integral_metrics(Q, t_e=1, Q_target=0.95)
        # t_r=2 (Q[2]=0.95 >= 0.95)
        # R_rp = (0.95 - 0.5) / (2 - 1) = 0.45
        assert math.isclose(m["R_rp"], 0.45, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Smoke test: S1 з 3 прогонами (Math Spec 11.5)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_RAW_COLUMNS = {
    "run_id", "series", "config_id", "seed",
    "topology", "N", "H", "T_max",
    "F0", "rho", "R_rb", "R_rp", "R_inf",
    "N_irreversible_nodes", "N_irreversible_edges",
    "valid", "exclusion_reason", "non_recovered",
}

REQUIRED_AGG_COLUMNS = {
    "config_id", "series", "topology",
    "rho_mean", "R_rb_mean", "R_inf_mean",
    "rho_med", "rho_CVaR", "rho_IQR",
    "ci_lower", "ci_upper",
    "n_valid", "n_invalid", "n_non_recovered",
}


def test_smoke_S1_schema(tmp_path):
    """S1 з 3 прогонами: схема raw_results і aggregated_results коректна."""
    raw_df = run_series("S1", n_runs=3, output_dir=str(tmp_path))

    missing = REQUIRED_RAW_COLUMNS - set(raw_df.columns)
    assert not missing, f"Відсутні колонки у raw_results: {missing}"

    # 4 топології × 3 прогони = 12 рядків
    assert len(raw_df) == 12
    assert raw_df["series"].eq("S1").all()

    agg_df = pd.read_csv(tmp_path / "aggregated" / "S1" / "aggregated_results.csv")
    missing_agg = REQUIRED_AGG_COLUMNS - set(agg_df.columns)
    assert not missing_agg, f"Відсутні колонки у aggregated_results: {missing_agg}"
    assert len(agg_df) == 4  # 4 топології


def test_smoke_S1_rho_range(tmp_path):
    """ρ у valid прогонах має бути у [0, 1]."""
    raw_df = run_series("S1", n_runs=3, output_dir=str(tmp_path))
    valid = raw_df[raw_df["valid"].astype(bool)]
    assert (valid["rho"] >= 0.0).all()
    assert (valid["rho"] <= 1.0 + 1e-9).all()


def test_smoke_S1_files_exist(tmp_path):
    """Перевіряє, що всі 4 вихідних файли створені."""
    run_series("S1", n_runs=2, output_dir=str(tmp_path))
    assert (tmp_path / "raw" / "S1" / "raw_results.csv").exists()
    assert (tmp_path / "raw" / "S1" / "Q_curves.parquet").exists()
    assert (tmp_path / "raw" / "S1" / "run_metadata.json").exists()
    assert (tmp_path / "aggregated" / "S1" / "aggregated_results.csv").exists()


def test_series_outputs_are_isolated(tmp_path):
    """Різні серії пишуться у власні папки і не перезаписують одна одну."""
    run_series("S1", n_runs=1, output_dir=str(tmp_path))
    run_series("S2", n_runs=1, output_dir=str(tmp_path))

    s1_raw = tmp_path / "raw" / "S1" / "raw_results.csv"
    s2_raw = tmp_path / "raw" / "S2" / "raw_results.csv"
    assert s1_raw.exists()
    assert s2_raw.exists()

    s1_df = pd.read_csv(s1_raw)
    s2_df = pd.read_csv(s2_raw)
    assert s1_df["series"].eq("S1").all()
    assert s2_df["series"].eq("S2").all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Перевірка метрик у raw_results відповідають Q_curves.parquet
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_match_Q_curves(tmp_path):
    """
    Ключовий тест: метрики у raw_results обчислені правильно з Q-кривої.

    Для кожного valid прогону обчислюємо метрики незалежно з Q_curves.parquet
    і порівнюємо з raw_results.csv.
    """
    run_series("S1", n_runs=2, output_dir=str(tmp_path))

    raw_df = pd.read_csv(tmp_path / "raw" / "S1" / "raw_results.csv")
    q_df = pd.read_parquet(tmp_path / "raw" / "S1" / "Q_curves.parquet")

    valid_rows = raw_df[raw_df["valid"].astype(bool)]
    assert len(valid_rows) > 0, "Немає валідних прогонів для перевірки"

    for _, row in valid_rows.iterrows():
        run_id = row["run_id"]
        run_q = q_df[q_df["run_id"] == run_id].sort_values("t")["Q"].tolist()

        assert len(run_q) > 0, f"Порожня Q-крива для run_id={run_id}"

        Q_target = 0.95 * run_q[0]
        metrics = compute_integral_metrics(run_q, t_e=1, Q_target=Q_target)

        assert math.isclose(row["rho"], metrics["rho"], rel_tol=1e-6), \
            f"rho mismatch для {run_id}"
        assert math.isclose(row["R_rb"], metrics["R_rb"], rel_tol=1e-6), \
            f"R_rb mismatch для {run_id}"

        # R_rp може бути NaN (нерушена мережа або миттєве відновлення)
        if math.isnan(metrics["R_rp"]):
            assert math.isnan(row["R_rp"]), f"R_rp має бути NaN для {run_id}"
        else:
            assert math.isclose(row["R_rp"], metrics["R_rp"], rel_tol=1e-6), \
                f"R_rp mismatch для {run_id}"

        assert math.isclose(row["R_inf"], metrics["R_inf"], rel_tol=1e-6), \
            f"R_inf mismatch для {run_id}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Q_curves.parquet: схема і Q[0]=1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_Q_curves_parquet_schema(tmp_path):
    """Q_curves.parquet має правильну схему і Q[0]=1.0 для всіх прогонів."""
    run_series("S1", n_runs=2, output_dir=str(tmp_path))
    q_df = pd.read_parquet(tmp_path / "raw" / "S1" / "Q_curves.parquet")

    assert {"run_id", "config_id", "t", "Q"}.issubset(set(q_df.columns))

    # Q[0] = 1.0 для всіх прогонів
    q0 = q_df[q_df["t"] == 0]["Q"]
    assert len(q0) == 4 * 2  # 4 конфіги × 2 прогони
    assert (q0 == 1.0).all()

    # Кожен прогін має рівно T_max+1 = 101 точок
    run_counts = q_df.groupby("run_id")["t"].count()
    assert (run_counts == 101).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Колонки N_irreversible_nodes / N_irreversible_edges
# ─────────────────────────────────────────────────────────────────────────────

def test_irreversible_columns_present(tmp_path):
    """N_irreversible_nodes і N_irreversible_edges присутні і >= 0."""
    raw_df = run_series("S1", n_runs=2, output_dir=str(tmp_path))
    assert "N_irreversible_nodes" in raw_df.columns
    assert "N_irreversible_edges" in raw_df.columns
    # S1 має w=0.0 → жодних незворотних елементів
    assert (raw_df["N_irreversible_nodes"] == 0).all()
    assert (raw_df["N_irreversible_edges"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Відтворюваність (Math Spec 11.6)
# ─────────────────────────────────────────────────────────────────────────────

def test_reproducibility(tmp_path):
    """Той самий series + n_runs → побайтово однаковий raw_results.csv."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    run_series("S1", n_runs=3, output_dir=str(out1))
    run_series("S1", n_runs=3, output_dir=str(out2))

    hash1 = hashlib.md5((out1 / "raw" / "S1" / "raw_results.csv").read_bytes()).hexdigest()
    hash2 = hashlib.md5((out2 / "raw" / "S1" / "raw_results.csv").read_bytes()).hexdigest()

    assert hash1 == hash2, "Reproducibility failed: raw_results.csv відрізняються"


def test_reproducibility_Q_curves(tmp_path):
    """Parquet Q_curves також відтворюваний (ті самі байти)."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    run_series("S1", n_runs=2, output_dir=str(out1))
    run_series("S1", n_runs=2, output_dir=str(out2))

    hash1 = hashlib.md5((out1 / "raw" / "S1" / "Q_curves.parquet").read_bytes()).hexdigest()
    hash2 = hashlib.md5((out2 / "raw" / "S1" / "Q_curves.parquet").read_bytes()).hexdigest()

    assert hash1 == hash2, "Reproducibility failed: Q_curves.parquet відрізняються"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Exclusion: valid / exclusion_reason / non_recovered
# ─────────────────────────────────────────────────────────────────────────────

def test_exclusion_columns_present(tmp_path):
    """Колонки valid, exclusion_reason, non_recovered присутні у CSV."""
    raw_df = run_series("S1", n_runs=2, output_dir=str(tmp_path))
    assert "valid" in raw_df.columns
    assert "exclusion_reason" in raw_df.columns
    assert "non_recovered" in raw_df.columns


def test_valid_runs_have_empty_exclusion_reason(tmp_path):
    """valid=True прогони мають exclusion_reason=''."""
    raw_df = run_series("S1", n_runs=2, output_dir=str(tmp_path))
    valid = raw_df[raw_df["valid"].astype(bool)]
    assert (valid["exclusion_reason"] == "").all()


def test_F0_zero_exclusion(tmp_path):
    """Прогін з F0=0 позначається invalid з exclusion_reason='F0_zero'."""
    import src.experiments.runner as runner_module

    cfg = load_config("S1", "config")
    cfg["network"]["topology"] = "hierarchical"

    original = runner_module.compute_max_flow

    # Monkey-patch у просторі імен runner-модуля
    runner_module.compute_max_flow = lambda net: 0.0
    try:
        row, Q_curve = _run_single(cfg, seed=1, run_id="test_001", config_id="test")
        assert row["valid"] is False
        assert row["exclusion_reason"] == "F0_zero"
        assert Q_curve == [1.0]
    finally:
        runner_module.compute_max_flow = original


# ─────────────────────────────────────────────────────────────────────────────
# 8. Config loading і sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_load_config_S1():
    """Завантаження S1: є секції network, disruption, recovery, simulation, sweep."""
    cfg = load_config("S1", "config")
    assert "network" in cfg
    assert "disruption" in cfg
    assert "recovery" in cfg
    assert "simulation" in cfg
    assert "sweep" in cfg


def test_S1_sweep_produces_4_combos():
    """S1 sweep: 4 топології → 4 combo після дедуплікації."""
    cfg = load_config("S1", "config")
    combos = _build_sweep_combinations(cfg["sweep"])
    assert len(combos) == 4
    topologies = {c["topology"] for c in combos}
    assert topologies == {"hierarchical", "hub", "reserved", "dense"}


def test_S3_sweep_fixed_budget_design():
    """S3 має 5 h_hubs × 4 attack_fraction × 2 modes = 40 конфігурацій."""
    cfg = load_config("S3", "config")
    combos = _build_sweep_combinations(cfg["sweep"])
    effective_ids = {_config_id(_effective_combo(combo)) for combo in combos}
    assert len(effective_ids) == 40
    assert cfg["disruption"]["type"] == "node_failure_fixed_budget"
    assert "h_hubs" in cfg["sweep"]
    assert "attack_fraction" in cfg["sweep"]
    assert set(cfg["sweep"]["disruption_mode"]) == {"random_fixed_k", "targeted"}


def test_effective_combo_deduplication_S4():
    """Дедуплікація для S4: різні p_v/p_e/r_clust при тому самому disruption_type
    дають один ефективний config_id."""
    combo1 = {"topology": "hierarchical", "disruption_type": "node_failure",
               "p_v": 0.05, "p_e": 0.10, "gamma": 0.25, "r_clust": 1}
    combo2 = {"topology": "hierarchical", "disruption_type": "node_failure",
               "p_v": 0.05, "p_e": 0.20, "gamma": 0.40, "r_clust": 3}

    eff1 = _effective_combo(combo1)
    eff2 = _effective_combo(combo2)

    # Обидва зводяться до {topology, disruption_type, p_v}
    assert eff1 == eff2
    assert _config_id(eff1) == _config_id(eff2)


def test_effective_combo_edge_degradation():
    """edge_degradation: залишаються лише p_e і gamma."""
    combo = {"topology": "hierarchical", "disruption_type": "edge_degradation",
             "p_v": 0.20, "p_e": 0.15, "gamma": 0.30, "r_clust": 2}
    eff = _effective_combo(combo)
    assert "p_v" not in eff
    assert "r_clust" not in eff
    assert "p_e" in eff
    assert "gamma" in eff


def test_run_id_format(tmp_path):
    """run_id формат: {series}_{config_id}_{run_idx:05d}."""
    raw_df = run_series("S1", n_runs=1, output_dir=str(tmp_path))
    for run_id in raw_df["run_id"]:
        parts = run_id.split("_")
        assert parts[0] == "S1"
        assert parts[-1] == "00001"  # run_idx=1, formatted as 5 digits


def test_metadata_json_schema(tmp_path):
    """run_metadata.json має необхідні поля."""
    import json as _json
    run_series("S1", n_runs=2, output_dir=str(tmp_path))
    with open(tmp_path / "raw" / "S1" / "run_metadata.json", encoding="utf-8") as f:
        meta = _json.load(f)
    assert meta["series"] == "S1"
    assert meta["n_configs"] == 4
    assert meta["n_runs_per_config"] == 2
    assert meta["total_runs"] == 8
    assert "valid_runs" in meta
    assert "elapsed_seconds" in meta
    assert "timestamp" in meta


def test_S3_raw_contains_fixed_budget_fields(tmp_path):
    """S3 raw_results має поля paired/fixed-budget/centralization з протоколу."""
    raw_df = run_series("S3", n_runs=1, output_dir=str(tmp_path))
    required = {
        "paired_id", "attack_fraction", "k_attack", "attack_rule",
        "candidate_set_size", "failed_nodes_count", "failed_hubs_count",
        "targeted_nodes", "degree_centralization",
        "betweenness_centralization", "flow_concentration_gini",
    }
    assert required.issubset(raw_df.columns)
    assert len(raw_df) == 40
    assert raw_df["topology"].eq("hub").all()
    assert raw_df["h_hubs"].isin([1, 2, 3, 5, 8]).all()
    assert raw_df["k_attack"].ge(1).all()


def test_S3_paired_seed_design(tmp_path):
    """random_fixed_k і targeted мають однаковий paired_id для одного h/q/run."""
    raw_df = run_series("S3", n_runs=1, output_dir=str(tmp_path))
    counts = raw_df.groupby(["h_hubs", "attack_fraction", "paired_id"]).size()
    assert (counts == 2).all()
    for _, group in raw_df.groupby(["h_hubs", "attack_fraction", "paired_id"]):
        assert set(group["disruption_mode"]) == {"random_fixed_k", "targeted"}


def test_S3_aggregated_contains_vulnerability_indices(tmp_path):
    """S3 aggregated_results містить Delta/VI індекси для конференційного аналізу."""
    run_series("S3", n_runs=2, output_dir=str(tmp_path))
    agg_df = pd.read_csv(tmp_path / "aggregated" / "S3" / "aggregated_results.csv")
    for col in ["Delta_rho_med", "Delta_CVaR", "VI_target", "VI_CVaR"]:
        assert col in agg_df.columns
    targeted = agg_df[agg_df["disruption_mode"] == "targeted"]
    assert targeted[["Delta_rho_med", "Delta_CVaR", "VI_target", "VI_CVaR"]].notna().all().all()


def test_R2_K_res_invariant_disruption_same_failed_nodes():
    """R2: однаковий seed → однаковий failed_nodes для K_res=0 і K_res=3."""
    cfg0 = load_config("R2", "config")
    cfg3 = load_config("R2", "config")
    cfg0["network"]["K_res"] = 0
    cfg3["network"]["K_res"] = 3

    rng_master = np.random.default_rng(12345)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))

    net0 = _create_network(cfg0, seed=net_seed)
    net3 = _create_network(cfg3, seed=net_seed)

    disrupted0 = _apply_r2_invariant_disruption(cfg0, net0, net_seed, dis_seed)
    disrupted3 = _apply_r2_invariant_disruption(cfg3, net3, net_seed, dis_seed)

    assert disrupted0.failed_nodes == disrupted3.failed_nodes
    assert disrupted0.disruption_metadata["disruption_invariant"] is True
    assert disrupted3.disruption_metadata["reference_K_res"] == 0


def test_D1_K_res0_nodes_have_no_backup_neighbors():
    """D1 instrumentation: у K_res=0 reserved немає incident backup-ребер."""
    cfg = load_config("R2", "config")
    cfg["network"]["K_res"] = 0
    net = _create_network(cfg, seed=42)

    for node in net.graph.nodes():
        entry = _make_node_recovery_log_entry(
            net=net,
            t=1,
            element=("node", node),
            action_context={},
            c_base=float(cfg["network"]["c_base"]),
        )
        assert entry["has_backup_neighbor"] is False
        assert entry["incident_backup_count"] == 0


def test_R2_prime_recovery_log_uses_backup_edge_type():
    """R2': backup detection не має залежати від capacity == c_base."""
    cfg = load_config("R2_prime", "config")
    cfg["network"]["K_res"] = 3
    cfg["network"]["c_backup"] = 500.0
    net = _create_network(cfg, seed=42)

    backup_edge = next(
        (u, v)
        for u, v, data in net.graph.edges(data=True)
        if data.get("edge_type") == "backup"
    )
    node = backup_edge[1]

    entry = _make_node_recovery_log_entry(
        net=net,
        t=1,
        element=("node", node),
        action_context={},
        c_base=float(cfg["network"]["c_base"]),
    )

    assert entry["has_backup_neighbor"] is True
    assert entry["incident_backup_count"] >= 1
    assert entry["newly_unlocked_backup_capacity"] >= 500.0


def test_R2_prime_reference_K_res_metadata():
    """R2': invariant disruption генерується на reference_K_res=3."""
    cfg = load_config("R2_prime", "config")

    rng_master = np.random.default_rng(12345)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))

    net = _create_network(cfg, seed=net_seed)
    disrupted = _apply_r2_invariant_disruption(cfg, net, net_seed, dis_seed)

    assert disrupted.disruption_metadata["disruption_invariant"] is True
    assert disrupted.disruption_metadata["reference_K_res"] == 3


def test_R3_sweep_and_paired_mu_design(tmp_path):
    """R3: 2 topology × 4 μ × 3 strategies; paired_id спільний у межах topology/seed."""
    raw_df = run_series("R3", n_runs=1, output_dir=str(tmp_path))

    assert len(raw_df) == 24
    assert set(raw_df["topology"]) == {"hierarchical", "dense"}
    assert set(raw_df["mu"]) == {1, 2, 5, 10}
    assert set(raw_df["recovery_strategy"]) == {
        "random",
        "centrality_betweenness",
        "flow_greedy",
    }

    counts = raw_df.groupby(["topology", "paired_id"]).size()
    assert (counts == 12).all()


def test_R3_invariant_disruption_same_failed_nodes_for_mu():
    """R3: однаковий seed/topology → однаковий failed_nodes для μ=1 і μ=10."""
    cfg1 = load_config("R3", "config")
    cfg10 = load_config("R3", "config")
    cfg1["network"]["topology"] = "hierarchical"
    cfg10["network"]["topology"] = "hierarchical"
    cfg1["recovery"]["mu"] = 1
    cfg10["recovery"]["mu"] = 10

    seed, paired_id = _r3_seed_and_paired_id(cfg1, run_idx=0, seed_offset=0)
    assert paired_id == "R3_hierarchical_00001"

    rng_master = np.random.default_rng(seed)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))

    net1 = _create_network(cfg1, seed=net_seed)
    net10 = _create_network(cfg10, seed=net_seed)
    disrupted1 = _apply_r2_invariant_disruption(cfg1, net1, net_seed, dis_seed)
    disrupted10 = _apply_r2_invariant_disruption(cfg10, net10, net_seed, dis_seed)

    assert disrupted1.failed_nodes == disrupted10.failed_nodes
    assert disrupted1.disruption_metadata["disruption_invariant"] is True


def test_recovery_log_disabled_by_default():
    """R1/R2 default: без enable_recovery_log row не містить diagnostic actions."""
    cfg = load_config("R2", "config")
    row, _ = _run_single(cfg, seed=123, run_id="r2_no_log", config_id="test")
    assert "_recovery_actions" not in row


def test_D2_sweep_and_paired_design(tmp_path):
    """D2a: 4 K_res × 3 strategies, один paired_id на seed."""
    raw_df = run_series("D2a", n_runs=1, output_dir=str(tmp_path))

    assert len(raw_df) == 12
    assert set(raw_df["K_res"]) == {0, 1, 2, 3}
    assert set(raw_df["recovery_strategy"]) == {
        "random",
        "centrality_betweenness",
        "flow_greedy",
    }
    assert raw_df["disruption_type"].eq("edge_degradation").all()
    assert raw_df["p_e"].eq(0.20).all()
    assert raw_df["gamma"].eq(0.50).all()

    counts = raw_df.groupby("paired_id").size()
    assert len(counts) == 1
    assert counts.iloc[0] == 12


def test_D2_invariant_trunk_degradation_same_edges_for_Kres():
    """D2: однаковий seed → той самий список degraded trunk-ребер для K_res."""
    cfg0 = load_config("D2b", "config")
    cfg3 = load_config("D2b", "config")
    cfg0["network"]["K_res"] = 0
    cfg3["network"]["K_res"] = 3

    seed, paired_id = _d2_seed_and_paired_id("D2b", cfg0, run_idx=0, seed_offset=0)
    assert paired_id == "D2b_reserved_00001"

    rng_master = np.random.default_rng(seed)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))

    net0 = _create_network(cfg0, seed=net_seed)
    net3 = _create_network(cfg3, seed=net_seed)
    disrupted0 = _apply_invariant_edge_degradation(cfg0, net0, net_seed, dis_seed)
    disrupted3 = _apply_invariant_edge_degradation(cfg3, net3, net_seed, dis_seed)

    degraded0 = {
        edge for edge, cap in disrupted0.edge_capacities.items()
        if cap < disrupted0._base_capacities[edge]
    }
    degraded3 = {
        edge for edge, cap in disrupted3.edge_capacities.items()
        if cap < disrupted3._base_capacities[edge]
    }

    assert degraded0
    assert degraded0 == degraded3
    for edge in degraded3:
        assert disrupted3.graph.edges[edge].get("edge_type") != "backup"


def test_R5_sweep_design():
    """R5: 2 topology × 4 N × 3 strategies = 24 конфігурації."""
    cfg = load_config("R5", "config")
    combos = _build_sweep_combinations(cfg["sweep"])
    effective_ids = {_config_id(_effective_combo(combo)) for combo in combos}

    assert len(effective_ids) == 24
    assert set(cfg["sweep"]["topology"]) == {"hierarchical", "dense"}
    assert set(cfg["sweep"]["N"]) == {30, 50, 80, 120}
    assert set(cfg["sweep"]["recovery_strategy"]) == {
        "random",
        "centrality_betweenness",
        "flow_greedy",
    }
    assert cfg["paired_design"]["disruption_invariant"] is True


def test_R5_seed_same_across_strategies_for_topology_N():
    """R5: strategy не входить у seed, тож paired_id спільний у межах topology/N/run."""
    cfg_random = load_config("R5", "config")
    cfg_flow = load_config("R5", "config")
    cfg_random["network"]["topology"] = "hierarchical"
    cfg_flow["network"]["topology"] = "hierarchical"
    cfg_random["network"]["N"] = 80
    cfg_flow["network"]["N"] = 80
    cfg_random["recovery"]["strategy"] = "random"
    cfg_flow["recovery"]["strategy"] = "flow_greedy"

    seed_random, paired_random = _r5_seed_and_paired_id(cfg_random, run_idx=0, seed_offset=0)
    seed_flow, paired_flow = _r5_seed_and_paired_id(cfg_flow, run_idx=0, seed_offset=0)

    assert seed_random == seed_flow
    assert paired_random == paired_flow
    assert paired_random == "R5_hierarchical_N80_00001"


def test_R5_invariant_disruption_same_failed_nodes_for_strategies():
    """R5: однаковий topology/N/seed → однаковий failed_nodes для різних strategies."""
    cfg_random = load_config("R5", "config")
    cfg_flow = load_config("R5", "config")
    cfg_random["network"]["topology"] = "dense"
    cfg_flow["network"]["topology"] = "dense"
    cfg_random["network"]["N"] = 30
    cfg_flow["network"]["N"] = 30
    cfg_random["recovery"]["strategy"] = "random"
    cfg_flow["recovery"]["strategy"] = "flow_greedy"

    seed, paired_id = _r5_seed_and_paired_id(cfg_random, run_idx=0, seed_offset=0)
    assert paired_id == "R5_dense_N30_00001"

    rng_master = np.random.default_rng(seed)
    net_seed = int(rng_master.integers(0, 2**31))
    dis_seed = int(rng_master.integers(0, 2**31))

    net_random = _create_network(cfg_random, seed=net_seed)
    net_flow = _create_network(cfg_flow, seed=net_seed)
    disrupted_random = _apply_r2_invariant_disruption(cfg_random, net_random, net_seed, dis_seed)
    disrupted_flow = _apply_r2_invariant_disruption(cfg_flow, net_flow, net_seed, dis_seed)

    assert disrupted_random.failed_nodes == disrupted_flow.failed_nodes
    assert disrupted_random.disruption_metadata["disruption_invariant"] is True


def test_M0_sweep_design():
    """M0: 4 topology × 5 visibility strategies = 20 конфігурацій."""
    cfg = load_config("M0", "config")
    combos = _build_sweep_combinations(cfg["sweep"])
    effective_ids = {_config_id(_effective_combo(combo)) for combo in combos}

    assert len(effective_ids) == 20
    assert set(cfg["sweep"]["topology"]) == {"hierarchical", "dense", "hub", "reserved"}
    assert set(cfg["sweep"]["recovery_strategy"]) == {
        "flow_greedy_local_r0",
        "flow_greedy_local_r1",
        "flow_greedy_local_r2",
        "flow_greedy_local_r3",
        "flow_greedy_local_inf",
    }
    assert cfg["paired_design"]["disruption_invariant"] is True
    assert cfg["random_seed_base"] == 4200


def test_M0_seed_same_across_visibility_strategies():
    """M0: r-стратегія не входить у seed, paired_id спільний у межах topology/run."""
    cfg0 = load_config("M0", "config")
    cfginf = load_config("M0", "config")
    cfg0["network"]["topology"] = "hub"
    cfginf["network"]["topology"] = "hub"
    cfg0["recovery"]["strategy"] = "flow_greedy_local_r0"
    cfginf["recovery"]["strategy"] = "flow_greedy_local_inf"

    seed0, paired0 = _m0_seed_and_paired_id(cfg0, run_idx=0, seed_offset=0)
    seedinf, pairedinf = _m0_seed_and_paired_id(cfginf, run_idx=0, seed_offset=0)

    assert seed0 == seedinf
    assert paired0 == pairedinf
    assert paired0 == "M0_hub_00001"


def test_M0_outputs_use_article4_directories(tmp_path):
    """M0 пише у results/raw/article4/M0 та results/aggregated/article4."""
    raw_df = run_series("M0", n_runs=1, output_dir=str(tmp_path), topology_filter="hierarchical")

    assert len(raw_df) == 5
    assert (tmp_path / "raw" / "article4" / "M0" / "raw_results.csv").exists()
    assert (tmp_path / "raw" / "article4" / "M0" / "Q_curves.parquet").exists()
    assert (tmp_path / "raw" / "article4" / "M0" / "run_metadata.json").exists()
    assert (tmp_path / "aggregated" / "article4" / "M0_aggregated_results.csv").exists()
    assert not (tmp_path / "raw" / "M0").exists()

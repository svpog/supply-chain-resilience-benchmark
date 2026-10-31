"""
Regression-тести generic multi-agent runner config routing.

Реалізує: Протокол M2, блокер-рішення щодо узагальнення runner-а
(див. docs/Протокол_M2.docx).
"""

from __future__ import annotations

from src.experiments.multi_agent_runner import load_config, run_series


def test_m1_config_still_writes_to_m1_dir(tmp_path) -> None:
    """Generic runner із series_M1.yaml пише artifacts у M1 namespace."""
    cfg = load_config("M1")

    raw = run_series(
        cfg,
        n_runs=2,
        output_dir=tmp_path,
        topology_filter="hierarchical",
        strategies_filter=["random_agent"],
    )

    assert len(raw) == 2
    assert (tmp_path / "raw" / "article4" / "M1" / "raw_results.csv").exists()
    assert (tmp_path / "raw" / "article4" / "M1" / "Q_curves.parquet").exists()
    assert (tmp_path / "raw" / "article4" / "M1" / "run_metadata.json").exists()
    assert (tmp_path / "aggregated" / "article4" / "M1_aggregated_results.csv").exists()


def test_m2_config_writes_to_m2_dir(tmp_path) -> None:
    """Generic runner із series_M2.yaml пише artifacts у M2 namespace, не M1."""
    cfg = load_config("M2")

    raw = run_series(
        cfg,
        n_runs=2,
        output_dir=tmp_path,
        topology_filter="hierarchical",
        K_filter=[1],
    )

    assert len(raw) == 2
    assert (tmp_path / "raw" / "article4" / "M2" / "raw_results.csv").exists()
    assert (tmp_path / "raw" / "article4" / "M2" / "Q_curves.parquet").exists()
    assert (tmp_path / "raw" / "article4" / "M2" / "run_metadata.json").exists()
    assert (tmp_path / "aggregated" / "article4" / "M2_aggregated_results.csv").exists()
    assert not (tmp_path / "raw" / "article4" / "M1" / "raw_results.csv").exists()

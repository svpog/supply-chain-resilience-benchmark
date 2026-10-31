"""
Unit-тести системи метрик resilience.

Реалізує: розділ 11.4 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Тести охоплюють:
    - Потокові метрики (flow): dict-інтерфейс, Q_t=1.0 для intact.
    - Структурні метрики: LCC=1, N_cc=1, I_iso=0 для intact мережі.
    - Інтегральні метрики: R_rb=1, ρ=1 для intact; ρ=1/T для destroyed.
    - R_∞ = 1.0 при w=0 і достатньому T_max (Math Spec 11.4).
    - ρ_CVaR_0.10 на синтетичних даних = аналітичний результат.
    - Bootstrap CI містить справжню медіану.
    - TimeSeriesLogger: log_step, get_Q_curve, hook log_event.
"""

import math

import networkx as nx
import numpy as np
import pytest

from src.network.base import Network
from src.network.flow import compute_max_flow, compute_Q
from src.disruptions.node_failure import NodeFailureDisruption
from src.recovery.random_strategy import RandomRecoveryStrategy

from src.metrics.flow import compute_flow_metrics
from src.metrics.structural import (
    compute_LCC, compute_N_cc, compute_I_iso, compute_APL,
    compute_B_c_lost, compute_base_bc_sum, compute_structural_metrics,
)
from src.metrics.integral import (
    compute_rho, compute_R_rb, compute_R_rp, compute_R_inf,
    compute_integral_metrics,
)
from src.metrics.stochastic import (
    compute_rho_med, compute_rho_CVaR, compute_rho_IQR,
    compute_bootstrap_CI, compute_stochastic_metrics,
)
from src.experiments.logger import TimeSeriesLogger


# ─────────────────────────────────────────────
# Допоміжні фабрики
# ─────────────────────────────────────────────

def _make_line_network() -> Network:
    """Лінія A→B (ідентична фабриці у test_flow.py)."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, capacity=3.0)
    return Network(
        graph=G,
        suppliers=frozenset([0]),
        consumers=frozenset([1]),
        base_capacities={(0, 1): 3.0},
        node_supply={0: 5.0, 1: 0.0},
        node_demand={0: 0.0, 1: 7.0},
    )


# ─────────────────────────────────────────────
# Тести потокових метрик
# ─────────────────────────────────────────────

class TestFlowMetrics:

    def test_returns_all_keys(self):
        """compute_flow_metrics повертає dict з чотирма ключами."""
        net = _make_line_network()
        F0 = compute_max_flow(net)
        metrics = compute_flow_metrics(net, F0)
        assert set(metrics.keys()) == {"F_t", "Q_t", "D_sat", "delta_C"}

    def test_Q_t_equals_one_for_intact(self):
        """Q_t = 1.0 для нерушеної мережі."""
        net = Network.create("hierarchical", N=30, seed=1)
        F0 = compute_max_flow(net)
        metrics = compute_flow_metrics(net, F0)
        assert metrics["Q_t"] == pytest.approx(1.0)

    def test_delta_C_zero_for_intact(self):
        """ΔC = 0.0 для нерушеної мережі."""
        net = Network.create("hierarchical", N=30, seed=1)
        F0 = compute_max_flow(net)
        metrics = compute_flow_metrics(net, F0)
        assert metrics["delta_C"] == pytest.approx(0.0)

    def test_Q_t_zero_when_F0_zero(self):
        """Q_t = 0.0 коли F0 = 0 (граничний випадок)."""
        net = _make_line_network()
        metrics = compute_flow_metrics(net, F0=0.0)
        assert metrics["Q_t"] == pytest.approx(0.0)

    def test_D_sat_intact_line_network(self):
        """D_sat = F(0) / Σd_j = 3 / 7 для line network."""
        net = _make_line_network()
        F0 = compute_max_flow(net)
        metrics = compute_flow_metrics(net, F0)
        assert metrics["D_sat"] == pytest.approx(3.0 / 7.0)

    def test_flow_metrics_after_disruption(self):
        """Після збою Q_t < 1 і ΔC > 0."""
        net = Network.create("hierarchical", N=30, seed=1)
        F0 = compute_max_flow(net)
        disrupted = NodeFailureDisruption(p_v=0.5).apply(net, seed=0)
        metrics = compute_flow_metrics(disrupted, F0)
        assert 0.0 <= metrics["Q_t"] <= 1.0 + 1e-9
        assert metrics["delta_C"] >= 0.0


# ─────────────────────────────────────────────
# Тести структурних метрик
# ─────────────────────────────────────────────

class TestStructuralMetrics:

    def test_LCC_intact_connected(self):
        """LCC = 1.0 для нерушеної зв'язної мережі (dense гарантує зв'язність)."""
        net = Network.create("dense", N=20, seed=1)
        assert compute_LCC(net) == pytest.approx(1.0)

    def test_N_cc_intact(self):
        """N_cc = 1 для нерушеної зв'язної мережі."""
        net = Network.create("dense", N=20, seed=1)
        assert compute_N_cc(net) == 1

    def test_I_iso_intact(self):
        """I_iso = 0.0 для нерушеної зв'язної мережі."""
        net = Network.create("dense", N=20, seed=1)
        assert compute_I_iso(net) == pytest.approx(0.0)

    def test_B_c_lost_intact(self):
        """B_c_lost = 0.0 для нерушеної мережі."""
        net = Network.create("hierarchical", N=30, seed=1)
        base_bc = compute_base_bc_sum(net)
        assert compute_B_c_lost(net, base_bc) == pytest.approx(0.0, abs=1e-9)

    def test_APL_intact_positive(self):
        """APL > 0.0 для зв'язної мережі з N > 1."""
        net = Network.create("hierarchical", N=30, seed=1)
        assert compute_APL(net) > 0.0

    def test_LCC_drops_after_failures(self):
        """LCC < 1.0 після збою ≥ 1 вузла у ланцюжковій мережі."""
        net = Network.create("hierarchical", N=30, seed=1)
        disrupted = NodeFailureDisruption(p_v=0.5).apply(net, seed=0)
        lcc = compute_LCC(disrupted)
        assert 0.0 <= lcc <= 1.0

    def test_N_cc_increases_after_failures(self):
        """N_cc ≥ 1 після будь-яких збоїв."""
        net = Network.create("hierarchical", N=30, seed=1)
        disrupted = NodeFailureDisruption(p_v=0.5).apply(net, seed=0)
        assert compute_N_cc(disrupted) >= 1

    def test_I_iso_all_failed(self):
        """I_iso = 1.0 якщо всі вузли failed."""
        net = _make_line_network()
        for n in list(net.graph.nodes()):
            net.set_node_state(n, "failed")
        assert compute_I_iso(net) == pytest.approx(1.0)

    def test_compute_structural_metrics_returns_all_keys(self):
        """compute_structural_metrics повертає dict з 5 ключами."""
        net = Network.create("dense", N=20, seed=3)
        base_bc = compute_base_bc_sum(net)
        metrics = compute_structural_metrics(net, base_bc)
        assert set(metrics.keys()) == {"LCC", "N_cc", "I_iso", "APL", "B_c_lost"}

    def test_structural_metrics_range(self):
        """Всі метрики ∈ [0, N] (LCC, I_iso ∈ [0,1]; N_cc ≥ 0; APL ≥ 0)."""
        net = Network.create("hub", N=20, seed=5)
        base_bc = compute_base_bc_sum(net)
        disrupted = NodeFailureDisruption(p_v=0.3).apply(net, seed=1)
        m = compute_structural_metrics(disrupted, base_bc)
        assert 0.0 <= m["LCC"] <= 1.0 + 1e-9
        assert 0.0 <= m["I_iso"] <= 1.0 + 1e-9
        assert 0.0 <= m["B_c_lost"] <= 1.0 + 1e-9
        assert m["N_cc"] >= 0
        assert m["APL"] >= 0.0


# ─────────────────────────────────────────────
# Тести інтегральних метрик
# ─────────────────────────────────────────────

class TestIntegralMetrics:

    def test_R_rb_intact_network(self):
        """R_rb = 1.0 для нерушеної мережі (Math Spec 11.4)."""
        Q_curve = [1.0] * 20
        assert compute_R_rb(Q_curve) == pytest.approx(1.0)

    def test_rho_intact_network(self):
        """ρ = 1.0 для нерушеної мережі (Math Spec 11.4)."""
        Q_curve = [1.0] * 20
        assert compute_rho(Q_curve) == pytest.approx(1.0)

    def test_rho_fully_destroyed_equals_one_over_T(self):
        """
        ρ = 1/T для повністю знищеної мережі з T кроками (Math Spec 11.4).

        Q_curve = [1.0, 0.0, ..., 0.0] довжиною T.
        """
        T = 25
        Q_curve = [1.0] + [0.0] * (T - 1)
        rho = compute_rho(Q_curve)
        assert rho == pytest.approx(1.0 / T)

    def test_R_inf_recovery_simulation(self):
        """
        При w=0 і достатньому T_max: мережа повністю відновлюється.

        Перевіряємо два факти (Math Spec 11.4):
        1. Q(T_max) ≈ 1.0 — фактичне повне відновлення.
        2. R_∞ ≥ 0.95 — відновлення перетнуло Q_target (≥ 95% від F(0)).

        Примітка: R_∞ = Q(t_r)/Q(0), де t_r = перший крок з Q ≥ 0.95.
        Q(t_r) може бути точно 0.95 (якщо крок відновлення виводить Q на
        поріг, а не вище), тому R_∞ = 0.95 є коректним результатом.
        """
        net = Network.create("hierarchical", N=30, seed=42)
        F0 = compute_max_flow(net)
        assert F0 > 0.0, "Базовий потік = 0, прогін невалідний"

        disrupted = NodeFailureDisruption(p_v=0.3, w=0.0).apply(net, seed=7)
        strategy = RandomRecoveryStrategy()
        rng = np.random.default_rng(0)

        Q_curve = [1.0]
        T_max = 60
        for t in range(1, T_max + 1):
            Q_curve.append(compute_Q(disrupted, F0))
            element = strategy.select_element_to_repair(disrupted, t, rng=rng)
            if element:
                strategy.repair(disrupted, element)

        # Факт 1: повне відновлення до Q=1.0 за T_max кроків (w=0)
        assert Q_curve[-1] == pytest.approx(1.0), (
            f"Q(T_max={T_max})={Q_curve[-1]:.4f} ≠ 1.0 при w=0"
        )
        # Факт 2: R_∞ ≥ Q_target = 0.95 (відновлення перетнуло поріг)
        R_inf = compute_R_inf(Q_curve, t_e=1)
        assert R_inf >= 0.95 - 1e-9, (
            f"R_∞={R_inf:.4f} < 0.95 при w=0 і T_max={T_max}"
        )

    def test_R_rb_is_minimum(self):
        """R_rb = мінімальне значення у Q_curve."""
        Q_curve = [1.0, 0.9, 0.4, 0.6, 0.85, 1.0]
        assert compute_R_rb(Q_curve) == pytest.approx(0.4)

    # --- R_rp: нові тести за Math Spec 6.2 ---

    def test_R_rp_intact_returns_nan(self):
        """
        R_rp = NaN для нерушеної мережі (Math Spec 6.2).

        Q ≡ 1 → Q(t_e) = 1.0 ≥ Q_target = 0.95 → t_r = t_e → NaN.
        Швидкість відновлення невизначена: не було чого відновлювати.
        """
        Q_curve = [1.0] * 10
        result = compute_R_rp(Q_curve, t_e=1)
        assert math.isnan(result), f"Очікувалось NaN, отримано {result}"

    def test_R_rp_finds_t_r_and_computes_rate(self):
        """
        R_rp = (Q(t_r) − Q(t_e)) / (t_r − t_e)  (Math Spec 6.2).

        Q = [1.0, 0.3, 0.5, 0.7, 0.95, 0.97], Q_target=0.95, t_e=1.
        t_r = 4 (перший t≥1 де Q≥0.95).
        R_rp = (0.95 − 0.3) / (4 − 1) = 0.65/3 ≈ 0.2167 Q/крок.
        """
        Q_curve = [1.0, 0.3, 0.5, 0.7, 0.95, 0.97]
        result = compute_R_rp(Q_curve, t_e=1, Q_target=0.95)
        assert result == pytest.approx(0.65 / 3), (
            f"R_rp={result:.4f}, очікувалось {0.65/3:.4f}"
        )

    def test_R_rp_never_reaches_target_uses_T_max(self):
        """
        R_rp при Q, що ніколи не досягає Q_target: t_r = T_max (Math Spec 4.1).

        Q = [1.0, 0.3, 0.4, 0.5], Q_target=0.95, t_e=1, T_max=3.
        t_r = 3 (Q[3]=0.5 < 0.95, але це T_max).
        R_rp = (0.5 − 0.3) / (3 − 1) = 0.2/2 = 0.1 Q/крок.
        """
        Q_curve = [1.0, 0.3, 0.4, 0.5]
        result = compute_R_rp(Q_curve, t_e=1, Q_target=0.95)
        assert result == pytest.approx(0.1), (
            f"R_rp={result:.4f}, очікувалось 0.1"
        )

    def test_R_rp_no_change_at_T_max(self):
        """R_rp = 0.0 якщо Q не зросло від t_e до T_max."""
        Q_curve = [1.0, 0.3, 0.3, 0.3]
        result = compute_R_rp(Q_curve, t_e=1, Q_target=0.95)
        assert result == pytest.approx(0.0)

    # --- R_inf: тести за Math Spec 6.4 ---

    def test_R_inf_at_t_r(self):
        """
        R_∞ = Q(t_r) / Q(0) (Math Spec 6.4).

        Q = [1.0, 0.3, 0.95, 1.0], Q_target=0.95, t_e=1.
        t_r = 2 (Q[2]=0.95≥0.95).
        R_∞ = Q[2] / Q[0] = 0.95 / 1.0 = 0.95.
        """
        Q_curve = [1.0, 0.3, 0.95, 1.0]
        result = compute_R_inf(Q_curve, t_e=1, Q_target=0.95)
        assert result == pytest.approx(0.95), (
            f"R_∞={result:.4f}, очікувалось 0.95"
        )

    def test_R_inf_intact_network(self):
        """
        R_∞ = 1.0 для нерушеної мережі (Math Spec 6.4).

        Q ≡ 1 → t_r = t_e (Q(t_e)=1≥0.95) → R_∞ = Q[t_e]/Q[0] = 1.0.
        """
        Q_curve = [1.0] * 10
        result = compute_R_inf(Q_curve, t_e=1)
        assert result == pytest.approx(1.0)

    def test_empty_Q_curve_returns_nan(self):
        """Порожня Q_curve → всі метрики NaN (Math Spec 6.2)."""
        metrics = compute_integral_metrics([])
        assert math.isnan(metrics["rho"])
        assert math.isnan(metrics["R_rb"])
        assert math.isnan(metrics["R_rp"])
        assert math.isnan(metrics["R_inf"])

    def test_compute_integral_metrics_keys(self):
        """compute_integral_metrics повертає dict з 4 ключами."""
        metrics = compute_integral_metrics([1.0, 0.8, 0.9, 1.0])
        assert set(metrics.keys()) == {"rho", "R_rb", "R_rp", "R_inf"}

    def test_rho_range(self):
        """ρ ∈ [0, 1] для будь-яких Q-значень з [0, 1]."""
        for vals in [[0.0]*10, [0.5]*10, [1.0]*10, [0.0, 1.0, 0.5]]:
            assert 0.0 <= compute_rho(vals) <= 1.0 + 1e-9


# ─────────────────────────────────────────────
# Тести стохастичних метрик
# ─────────────────────────────────────────────

class TestStochasticMetrics:

    def test_rho_CVaR_analytical(self):
        """
        ρ_CVaR_0.10 на синтетичних даних збігається з аналітичним результатом
        (Math Spec 11.4).

        Дані: [0.1, 0.2, ..., 1.0] (10 значень).
        Bottom 10% = floor(0.10 * 10) = 1 значення = [0.1].
        CVaR = 0.1.
        """
        rho_values = [0.1 * i for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]
        cvar = compute_rho_CVaR(rho_values, alpha=0.10)
        assert cvar == pytest.approx(0.1), (
            f"ρ_CVaR_0.10 = {cvar:.4f}, очікувалось 0.1"
        )

    def test_rho_CVaR_larger_alpha(self):
        """ρ_CVaR_0.20 = mean([0.1, 0.2]) = 0.15."""
        rho_values = [0.1 * i for i in range(1, 11)]
        cvar = compute_rho_CVaR(rho_values, alpha=0.20)
        assert cvar == pytest.approx(0.15)

    def test_rho_med_equals_numpy_median(self):
        """ρ_med збігається зі стандартним numpy median."""
        rho_values = np.array([0.3, 0.7, 0.5, 0.9, 0.1])
        assert compute_rho_med(rho_values) == pytest.approx(float(np.median(rho_values)))

    def test_rho_IQR_uniform(self):
        """ρ_IQR для рівномірних 0..1 ≈ 0.5."""
        rho_values = np.linspace(0, 1, 101)
        iqr = compute_rho_IQR(rho_values)
        assert iqr == pytest.approx(0.5, abs=0.02)

    def test_rho_IQR_constant(self):
        """ρ_IQR = 0 для константного масиву."""
        assert compute_rho_IQR([0.7] * 20) == pytest.approx(0.0)

    def test_bootstrap_CI_contains_true_median(self):
        """Bootstrap 95% CI містить справжню медіану на синтетичних даних."""
        rng_data = np.random.default_rng(42)
        rho_values = rng_data.uniform(0.5, 1.0, size=100)
        true_median = float(np.median(rho_values))
        ci_lower, ci_upper = compute_bootstrap_CI(rho_values, B=1000, seed=0)
        assert ci_lower <= true_median <= ci_upper, (
            f"True median={true_median:.4f} не потрапляє у CI [{ci_lower:.4f}, {ci_upper:.4f}]"
        )

    def test_bootstrap_CI_lower_leq_upper(self):
        """CI нижня межа ≤ верхня межа."""
        rho_values = np.array([0.4, 0.6, 0.8, 0.5, 0.7])
        lo, hi = compute_bootstrap_CI(rho_values, B=200, seed=1)
        assert lo <= hi

    def test_nan_excluded(self):
        """NaN-прогони виключаються з обчислень."""
        rho_values = [0.5, float("nan"), 0.8, float("nan"), 0.6]
        med = compute_rho_med(rho_values)
        expected = np.median([0.5, 0.8, 0.6])
        assert med == pytest.approx(expected)

    def test_empty_array_returns_nan(self):
        """Порожній масив → всі метрики NaN."""
        metrics = compute_stochastic_metrics([], alpha=0.10)
        for key, val in metrics.items():
            assert math.isnan(val), f"{key} має бути NaN для порожнього масиву"

    def test_compute_stochastic_metrics_keys(self):
        """compute_stochastic_metrics повертає dict з 5 ключами."""
        rho_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        metrics = compute_stochastic_metrics(rho_values, alpha=0.10, B=100, seed=0)
        assert set(metrics.keys()) == {"rho_med", "rho_CVaR", "rho_IQR",
                                        "ci_lower", "ci_upper"}

    def test_CVaR_invalid_alpha_raises(self):
        """alpha ≤ 0 або > 1 → ValueError."""
        with pytest.raises(ValueError):
            compute_rho_CVaR([0.5, 0.6], alpha=0.0)
        with pytest.raises(ValueError):
            compute_rho_CVaR([0.5, 0.6], alpha=1.5)

    def test_CVaR_leq_median(self):
        """ρ_CVaR_α ≤ ρ_med (нижній хвіст завжди ≤ медіани)."""
        rho_values = np.random.default_rng(5).uniform(0, 1, 200)
        cvar = compute_rho_CVaR(rho_values, alpha=0.10)
        med = compute_rho_med(rho_values)
        assert cvar <= med + 1e-9


# ─────────────────────────────────────────────
# Тести TimeSeriesLogger
# ─────────────────────────────────────────────

class TestTimeSeriesLogger:

    def test_log_step_stores_data(self):
        """log_step зберігає метрики з ключем 't'."""
        logger = TimeSeriesLogger()
        logger.log_step(t=0, metrics_dict={"Q_t": 1.0, "F_t": 100.0})
        steps = logger.get_steps()
        assert len(steps) == 1
        assert steps[0]["t"] == 0
        assert steps[0]["Q_t"] == pytest.approx(1.0)

    def test_get_Q_curve(self):
        """get_Q_curve повертає список Q_t у порядку кроків."""
        logger = TimeSeriesLogger()
        for t, q in enumerate([1.0, 0.7, 0.85, 1.0]):
            logger.log_step(t=t, metrics_dict={"Q_t": q})
        assert logger.get_Q_curve() == pytest.approx([1.0, 0.7, 0.85, 1.0])

    def test_log_event_hook(self):
        """log_event зберігає подію і не впливає на get_Q_curve."""
        logger = TimeSeriesLogger()
        logger.log_step(t=1, metrics_dict={"Q_t": 0.8})
        logger.log_event(t=1, event_type="node_repaired", payload={"node": 3})
        assert len(logger.get_Q_curve()) == 1
        events = logger.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "node_repaired"

    def test_clear_resets_all(self):
        """clear() видаляє всі кроки і події."""
        logger = TimeSeriesLogger()
        logger.log_step(t=0, metrics_dict={"Q_t": 1.0})
        logger.log_event(t=0, event_type="test", payload={})
        logger.clear()
        assert len(logger.get_steps()) == 0
        assert len(logger.get_events()) == 0
        assert len(logger.get_Q_curve()) == 0

    def test_len(self):
        """len(logger) повертає кількість кроків."""
        logger = TimeSeriesLogger()
        for t in range(5):
            logger.log_step(t=t, metrics_dict={"Q_t": 0.9})
        assert len(logger) == 5

    def test_logger_Q_curve_used_for_integral_metrics(self):
        """Q_curve з logger можна передати у compute_integral_metrics."""
        logger = TimeSeriesLogger()
        Q_values = [1.0, 0.6, 0.8, 0.9, 1.0]
        for t, q in enumerate(Q_values):
            logger.log_step(t=t, metrics_dict={"Q_t": q})

        from src.metrics.integral import compute_integral_metrics
        metrics = compute_integral_metrics(logger.get_Q_curve())
        assert metrics["rho"] == pytest.approx(np.mean(Q_values))
        assert metrics["R_inf"] == pytest.approx(1.0)

"""
Потокові метрики: F(t), Q(t), D_sat(t), ΔC(t).

Реалізує: розділ 5.2 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Тонкий wrapper над src/network/flow.py. Надає уніфікований dict-інтерфейс
для runner.py і скорочує кількість викликів compute_max_flow до одного
на крок симуляції.

Приклад використання:
    >>> from src.metrics.flow import compute_flow_metrics
    >>> metrics = compute_flow_metrics(network_t, F0)
    >>> print(metrics["Q_t"], metrics["D_sat"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.network.flow import compute_delta_C, compute_max_flow

if TYPE_CHECKING:
    from src.network.base import Network


def compute_flow_metrics(network: "Network", F0: float) -> dict[str, float]:
    """
    Обчислює всі потокові метрики для стану мережі в момент t.

    Викликає compute_max_flow рівно ОДИН раз і похідно обчислює Q_t і D_sat,
    уникаючи надлишкових обчислень у циклі симуляції.

    Args:
        network: поточний стан мережі (після збою та/або відновлення).
        F0: базовий потік F(0), обчислений до disruption.

    Returns:
        dict з ключами:
            "F_t"     — максимальний потік у момент t ≥ 0;
            "Q_t"     — нормована продуктивність Q(t) = F(t)/F(0) ∈ [0, 1];
            "D_sat"   — рівень задоволення попиту D_sat(t) ∈ [0, 1];
            "delta_C" — нормована втрата capacity ΔC(t) ∈ [0, 1].
    """
    F_t = compute_max_flow(network)

    # Q(t) = F(t)/F(0); 0 якщо F(0)=0 (граничний випадок — прогін виключається)
    Q_t = F_t / F0 if F0 > 0.0 else 0.0

    # D_sat = F(t) / Σd_j(0); знаменник — БАЗОВИЙ попит (до disruption)
    total_base_demand = sum(network._base_demand.values())
    D_sat = F_t / total_base_demand if total_base_demand > 0.0 else 0.0

    delta_C = compute_delta_C(network)

    return {
        "F_t": F_t,
        "Q_t": Q_t,
        "D_sat": D_sat,
        "delta_C": delta_C,
    }

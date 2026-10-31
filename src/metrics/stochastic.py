"""
Авторські стохастичні метрики resilience: ρ_med, ρ_CVaR_α, ρ_IQR.

Реалізує: розділ 7 Math Spec (АВТОРСЬКА НОВИЗНА 1-ї статті)
(див. docs/Math_Spec_Стаття_1.docx).

Bootstrap довірчі інтервали — розділ 9.2 Math Spec (B=1000).

Вхід: масив значень ρ зі M прогонів одного протоколу (одна серія, одна конфігурація).
NaN-прогони (F(0)=0) автоматично виключаються.

Формули (Math Spec 7):
    ρ_med     = median(ρ₁, ..., ρ_M)
    ρ_CVaR_α  = E[ρ | ρ ≤ VaR_α] = mean(найменших ⌊α·M⌋ значень)
    ρ_IQR     = Q75 − Q25

Bootstrap CI (Math Spec 9.2):
    B=1000 resample → CI для ρ_med (95% за замовчуванням).

Приклад використання:
    >>> from src.metrics.stochastic import compute_stochastic_metrics
    >>> metrics = compute_stochastic_metrics(rho_values=[0.7, 0.8, 0.9], alpha=0.10)
"""

from __future__ import annotations

import numpy as np


def compute_rho_med(rho_values: np.ndarray | list[float]) -> float:
    """
    ρ_med = median(ρ) — медіана розподілу resilience.

    Стійка до викидів альтернатива середньому. Основна описова статистика
    для порівняння топологій у серіях S1–S5.
    """
    arr = _clean(rho_values)
    if len(arr) == 0:
        return float("nan")
    return float(np.median(arr))


def compute_rho_CVaR(
    rho_values: np.ndarray | list[float],
    alpha: float = 0.10,
) -> float:
    """
    ρ_CVaR_α = E[ρ | ρ ≤ VaR_α] — очікуване значення resilience у найгірших α·100%.

    АВТОРСЬКА МЕТРИКА (Math Spec 7): вимірює «хвостовий ризик» — середню
    продуктивність у α·M найгірших прогонах. Дозволяє виявляти топології,
    стійкі до найбільш несприятливих збоїв.

    Формула: відсортувати ρ за зростанням → взяти перші ⌊α·M⌋ значень → mean.

    Args:
        rho_values: масив значень ρ (NaN автоматично виключаються).
        alpha: рівень ∈ (0, 1]. За замовчуванням 0.10 (нижні 10%).

    Returns:
        ρ_CVaR_α ∈ [0, 1]. NaN якщо масив порожній.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha={alpha} має бути у (0, 1].")
    arr = _clean(rho_values)
    if len(arr) == 0:
        return float("nan")
    sorted_arr = np.sort(arr)
    n_tail = max(1, int(np.floor(alpha * len(sorted_arr))))
    return float(np.mean(sorted_arr[:n_tail]))


def compute_rho_IQR(rho_values: np.ndarray | list[float]) -> float:
    """
    ρ_IQR = Q75 − Q25 — міжквартильний розмах розподілу resilience.

    АВТОРСЬКА МЕТРИКА (Math Spec 7): вимірює варіабельність resilience
    між прогонами. Висока IQR → непередбачувана поведінка мережі.

    Returns:
        ρ_IQR ≥ 0. NaN якщо масив порожній.
    """
    arr = _clean(rho_values)
    if len(arr) == 0:
        return float("nan")
    q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
    return q75 - q25


def compute_bootstrap_CI(
    rho_values: np.ndarray | list[float],
    B: int = 1000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Bootstrap довірчий інтервал для ρ_med (Math Spec 9.2).

    Алгоритм:
        1. B разів вибираємо n значень з rho_values з поверненням.
        2. Обчислюємо median для кожної вибірки.
        3. Повертаємо (α/2, 1−α/2) квантилі медіан.

    Args:
        rho_values: масив значень ρ.
        B: кількість bootstrap вибірок (Math Spec 9.2: B=1000).
        ci_level: рівень довіри (0.95 → 95% CI).
        seed: random seed для відтворюваності.

    Returns:
        (ci_lower, ci_upper) — межі CI.
    """
    arr = _clean(rho_values)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_medians = np.zeros(B)
    for i in range(B):
        sample = rng.choice(arr, size=n, replace=True)
        boot_medians[i] = np.median(sample)
    alpha_tail = (1.0 - ci_level) / 2.0
    ci_lower = float(np.quantile(boot_medians, alpha_tail))
    ci_upper = float(np.quantile(boot_medians, 1.0 - alpha_tail))
    return (ci_lower, ci_upper)


def compute_stochastic_metrics(
    rho_values: np.ndarray | list[float],
    alpha: float = 0.10,
    B: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Обчислює всі стохастичні метрики з масиву ρ-значень M прогонів.

    NaN-прогони (F(0)=0) автоматично виключаються.

    Args:
        rho_values: список/масив ρ зі M прогонів.
        alpha: рівень CVaR (0.10 → нижні 10%). Math Spec 7.
        B: кількість bootstrap вибірок (Math Spec 9.2).
        seed: seed для bootstrap.

    Returns:
        dict з ключами:
            "rho_med"   — медіана;
            "rho_CVaR"  — CVaR_α;
            "rho_IQR"   — IQR;
            "ci_lower"  — нижня межа 95% CI для rho_med;
            "ci_upper"  — верхня межа 95% CI для rho_med.
    """
    arr = _clean(rho_values)
    if len(arr) == 0:
        return {
            "rho_med": float("nan"),
            "rho_CVaR": float("nan"),
            "rho_IQR": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    ci_lower, ci_upper = compute_bootstrap_CI(arr, B=B, seed=seed)
    return {
        "rho_med": compute_rho_med(arr),
        "rho_CVaR": compute_rho_CVaR(arr, alpha=alpha),
        "rho_IQR": compute_rho_IQR(arr),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _clean(rho_values: np.ndarray | list[float]) -> np.ndarray:
    """Перетворює на масив float, видаляє NaN."""
    arr = np.array(rho_values, dtype=float)
    return arr[~np.isnan(arr)]

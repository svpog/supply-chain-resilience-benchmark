"""
Інтегральні метрики resilience: R_rb, R_rp, ρ, R_∞.

Реалізує: розділ 6 Math Spec
(див. docs/Math_Spec_Стаття_1.docx).

Метрики обчислюються з Q-кривої одного прогону (Q_curve — список Q(t)
від t=0 до t=T_max). Q_curve[0] = 1.0 завжди (pre-disruption baseline).

Формули (Math Spec 6):
    ρ    = (1/T) · Σ Q(t)                              [середня продуктивність]
    R_rb = min Q(t)                                    [найгірша продуктивність]
    R_rp = (Q(t_r) − Q(t_e)) / (t_r − t_e)           [Q/крок, швидкість відновлення]
    R_∞  = Q(t_r) / Q(0)                              [рівень відновлення в t_r]

де t_r = min{t ≥ t_e : Q(t) ≥ Q_target} або T_max  (Math Spec 4.1)
    Q_target = 0.95 · Q(0)  (поріг «практично відновлено»)

Обробка NaN (Math Spec 6.2):
    - Порожня Q_curve → всі метрики NaN.
    - t_r == t_e для R_rp (мережа вже на target у момент збою) → NaN.

Примітка для runner (Етап 5):
    Виклик: compute_integral_metrics(Q_curve, t_e=1)
    де t_e=1 — перший крок після збою (збій накладається після t=0).
    Q_target за замовчуванням = 0.95 · Q_curve[0].

Приклад використання:
    >>> from src.metrics.integral import compute_integral_metrics
    >>> metrics = compute_integral_metrics([1.0, 0.3, 0.7, 0.95, 1.0], t_e=1)
    >>> print(metrics["rho"], metrics["R_rp"], metrics["R_inf"])
"""

from __future__ import annotations

import math

import numpy as np

_NAN_DICT = {"rho": float("nan"), "R_rb": float("nan"),
             "R_rp": float("nan"), "R_inf": float("nan")}


# ─────────────────────────────────────────────
# Внутрішній помічник: знаходить t_r
# ─────────────────────────────────────────────

def _find_t_r(Q_curve: list[float], t_e: int, Q_target: float) -> int:
    """
    t_r = min{t ≥ t_e : Q(t) ≥ Q_target}  або  T_max  (Math Spec 4.1).

    Пошук ВКЛЮЧАЄ t_e: якщо Q(t_e) ≥ Q_target → t_r = t_e (мережа вже
    відновлена на момент збою; R_rp стає NaN, R_∞ = Q(t_e)/Q(0)).
    """
    T_max = len(Q_curve) - 1
    for t in range(t_e, T_max + 1):
        if Q_curve[t] >= Q_target:
            return t
    return T_max


# ─────────────────────────────────────────────
# Основні метрики
# ─────────────────────────────────────────────

def compute_rho(Q_curve: list[float]) -> float:
    """
    ρ = (1/T) · Σₜ Q(t) — інтегральна resilience (середня продуктивність).

    Еквівалентна нормованій площі під Q(t)-кривою.
    Для нерушеної мережі (Q≡1): ρ = 1.0.
    Для повністю знищеної Q=[1,0,...,0] з T точок: ρ = 1/T.
    """
    if not Q_curve:
        return float("nan")
    return float(np.nanmean(Q_curve))


def compute_R_rb(Q_curve: list[float]) -> float:
    """
    R_rb = min Q(t) — robustness (найгірша продуктивність за весь прогін).

    Вимірює глибину провалу мережі. R_rb = 1.0 для нерушеної.
    """
    if not Q_curve:
        return float("nan")
    return float(np.nanmin(Q_curve))


def compute_R_rp(
    Q_curve: list[float],
    t_e: int = 1,
    Q_target: float | None = None,
) -> float:
    """
    R_rp = (Q(t_r) − Q(t_e)) / (t_r − t_e)  [Q/крок] — швидкість відновлення.

    Вимірює середній приріст Q за один крок у фазі відновлення від t_e до t_r.
    Одиниці: Q-одиниць за крок (НЕ безрозмірна частка).

    t_r визначається через _find_t_r() за Math Spec 4.1.
    Q_target = 0.95 · Q(0) за замовчуванням.

    Спеціальні випадки (Math Spec 6.2):
        - Q_curve порожня або t_e поза межами → NaN.
        - t_r == t_e (Q(t_e) ≥ Q_target, мережа вже на target) → NaN.
          Наприклад: нерушена мережа з Q≡1 → Q(t_e)=1≥0.95 → NaN.

    Args:
        Q_curve: список Q(t) для t=0..T_max; Q_curve[0] = Q(0) = 1.0.
        t_e: крок збою (disruption event time). За замовчуванням 1.
        Q_target: абсолютний поріг відновлення. За замовчуванням 0.95·Q(0).

    Returns:
        R_rp [Q/крок] або NaN.
    """
    if not Q_curve or t_e >= len(Q_curve):
        return float("nan")
    if Q_target is None:
        Q_target = 0.95 * Q_curve[0]
    t_r = _find_t_r(Q_curve, t_e, Q_target)
    if t_r == t_e:
        # Мережа вже на порозі у момент збою → швидкість відновлення невизначена
        return float("nan")
    return (Q_curve[t_r] - Q_curve[t_e]) / (t_r - t_e)


def compute_R_inf(
    Q_curve: list[float],
    t_e: int = 1,
    Q_target: float | None = None,
) -> float:
    """
    R_∞ = Q(t_r) / Q(0) — рівень відновлення в момент t_r (Math Spec 6.4).

    Показує, наскільки мережа відновилась відносно початкового стану.
    R_∞ = 1.0: повне відновлення (або нерушена мережа).
    R_∞ < 1.0: часткове або нульове відновлення (w > 0 або малий T_max).

    t_r визначається так само як у compute_R_rp.
    Для нерушеної мережі: Q(t_e)=1≥Q_target → t_r=t_e → R_∞=Q(t_e)/Q(0)=1.0.

    Args:
        Q_curve: список Q(t) для t=0..T_max.
        t_e: крок збою. За замовчуванням 1.
        Q_target: абсолютний поріг. За замовчуванням 0.95·Q(0).

    Returns:
        R_∞ ∈ [0, 1] або NaN якщо Q(0) = 0.
    """
    if not Q_curve:
        return float("nan")
    Q_0 = Q_curve[0]
    if Q_0 <= 0.0:
        return float("nan")
    if Q_target is None:
        Q_target = 0.95 * Q_0
    t_r = _find_t_r(Q_curve, t_e, Q_target)
    return Q_curve[t_r] / Q_0


# ─────────────────────────────────────────────
# Збережена для довідки: попередня формула R_rp
# ─────────────────────────────────────────────

def _compute_recovery_completeness(Q_curve: list[float]) -> float:
    """
    Повнота відновлення = (Q_final − Q_min) / (1 − Q_min).

    Збережена для довідки (використовувалась як compute_R_rp до виправлення).
    Не відповідає Math Spec 6.2, але є осмисленою безрозмірною метрикою
    «наскільки мережа повернулась від мінімуму до максимуму».
    НЕ використовується у виробничому коді.
    """
    if not Q_curve:
        return float("nan")
    Q_min = float(np.nanmin(Q_curve))
    Q_final = float(Q_curve[-1])
    if math.isnan(Q_min) or math.isnan(Q_final):
        return float("nan")
    if Q_min >= 1.0 - 1e-9:
        return 1.0
    return float(np.clip((Q_final - Q_min) / (1.0 - Q_min), 0.0, 1.0))


# ─────────────────────────────────────────────
# Уніфікована функція для runner.py (Етап 5)
# ─────────────────────────────────────────────

def compute_integral_metrics(
    Q_curve: list[float],
    t_e: int = 1,
    Q_target: float | None = None,
) -> dict[str, float]:
    """
    Обчислює всі інтегральні метрики з Q-кривої одного прогону.

    Runner (Етап 5) викликає:
        metrics = compute_integral_metrics(Q_curve, t_e=1)
    де t_e=1 завжди, бо збій накладається після вимірювання Q(0).
    Q_target за замовчуванням = 0.95 · Q_curve[0].

    Args:
        Q_curve: список Q(t) для t=0..T_max; Q_curve[0]=1.0.
        t_e: крок збою. За замовчуванням 1.
        Q_target: абсолютний поріг відновлення. За замовчуванням 0.95·Q(0).

    Returns:
        dict з ключами: "rho", "R_rb", "R_rp", "R_inf".
        Всі NaN якщо Q_curve порожній.
    """
    if not Q_curve:
        return dict(_NAN_DICT)

    _Q_target = (0.95 * Q_curve[0]) if Q_target is None else Q_target

    return {
        "rho": compute_rho(Q_curve),
        "R_rb": compute_R_rb(Q_curve),
        "R_rp": compute_R_rp(Q_curve, t_e=t_e, Q_target=_Q_target),
        "R_inf": compute_R_inf(Q_curve, t_e=t_e, Q_target=_Q_target),
    }

"""
TimeSeriesLogger — логер часових рядів і подій для одного прогону.

Реалізує: hook для 2-ї статті (архітектурна точка розширення)
(див. CLAUDE.md розділ 2.3 та docs/Скелети_статей_2_3_4.docx).

У 1-й статті:
    - log_step() викликається на кожному кроці t і зберігає метрики.
    - get_Q_curve() повертає список Q(t) для integral metrics.

У 2-й статті (hook):
    - log_event() реєструватиме рішення агентів, зміни ролей тощо.
    - payload може містити agent_id, action, local_state тощо.

Приклад використання:
    >>> from src.experiments.logger import TimeSeriesLogger
    >>> logger = TimeSeriesLogger()
    >>> logger.log_step(t=0, metrics_dict={"Q_t": 1.0, "F_t": 300.0})
    >>> logger.log_step(t=1, metrics_dict={"Q_t": 0.6, "F_t": 180.0})
    >>> print(logger.get_Q_curve())   # [1.0, 0.6]
"""

from __future__ import annotations


class TimeSeriesLogger:
    """
    Зберігає метрики та події одного прогону симуляції.

    Дані зберігаються у пам'яті до завершення прогону, потім
    runner.py зчитує їх через get_steps() і get_Q_curve().
    """

    def __init__(self) -> None:
        self._steps: list[dict] = []
        self._events: list[dict] = []

    # ------------------------------------------------------------------
    # Основний інтерфейс (1-а стаття)
    # ------------------------------------------------------------------

    def log_step(self, t: int, metrics_dict: dict) -> None:
        """
        Записує метрики одного кроку симуляції.

        Args:
            t: номер кроку (0 = pre-disruption baseline).
            metrics_dict: словник метрик, наприклад:
                {"Q_t": 0.85, "F_t": 255.0, "D_sat": 0.81, "delta_C": 0.04}.
        """
        self._steps.append({"t": t, **metrics_dict})

    def log_event(self, t: int, event_type: str, payload: dict) -> None:
        """
        Записує подію (hook для 2-ї статті).

        У 1-й статті метод існує, але не викликається у production-коді.
        У 2-й статті агенти логуватимуть свої рішення тут.

        Args:
            t: крок, на якому сталась подія.
            event_type: рядковий тип події, наприклад 'node_repaired',
                        'agent_decision', 'role_change'.
            payload: довільний словник з деталями події.
        """
        self._events.append({"t": t, "event_type": event_type, "payload": payload})

    # ------------------------------------------------------------------
    # Зчитування даних
    # ------------------------------------------------------------------

    def get_steps(self) -> list[dict]:
        """Повертає всі записані кроки у порядку логування."""
        return list(self._steps)

    def get_Q_curve(self) -> list[float]:
        """
        Витягує Q_t з кожного кроку у порядку t.

        Потрібно для compute_integral_metrics(Q_curve).
        Кроки без ключа "Q_t" ігноруються.
        """
        return [step["Q_t"] for step in self._steps if "Q_t" in step]

    def get_events(self) -> list[dict]:
        """Повертає всі записані події (для 2-ї статті)."""
        return list(self._events)

    def clear(self) -> None:
        """Очищає всі дані (для повторного використання об'єкту між прогонами)."""
        self._steps.clear()
        self._events.clear()

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"TimeSeriesLogger(steps={len(self._steps)}, events={len(self._events)})"

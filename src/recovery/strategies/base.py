"""
Базовий інтерфейс recovery-стратегій статті 3.

Реалізує: розділ 3.1 Math Spec Article 3
(див. docs/Math_Spec_Article_3.docx).

У поточному коді проєкту базовий клас уже існує як
src.recovery.base.RecoveryStrategy. Цей модуль re-export-ить його для
сумісності з протоколом R1 і майбутньою структурою src/recovery/strategies/.
"""

from __future__ import annotations

from src.recovery.base import RecoveryStrategy


__all__ = ["RecoveryStrategy"]

"""
Random recovery strategy для статті 3.

Реалізує: розділ 3.3 Math Spec Article 3
(див. docs/Math_Spec_Article_3.docx).

Цей клас навмисно наслідує існуючу RandomRecoveryStrategy без зміни
алгоритму, щоб default='random' залишав поведінку статті 1 незмінною.
"""

from __future__ import annotations

from src.recovery.random_strategy import RandomRecoveryStrategy


class RandomStrategy(RandomRecoveryStrategy):
    """Alias-клас для registry статті 3 без зміни random behavior."""

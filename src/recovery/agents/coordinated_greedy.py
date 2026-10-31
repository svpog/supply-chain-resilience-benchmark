"""
Coordinated greedy-agent для M1.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from __future__ import annotations

from src.recovery.agents.local_greedy_agent import LocalGreedyAgent


class CoordinatedGreedyAgent(LocalGreedyAgent):
    """
    LocalGreedyAgent, який у M1 використовується з TokenCoordinator.

    Окремий клас потрібен для чіткої назви стратегії у конфігах і raw output.
    """

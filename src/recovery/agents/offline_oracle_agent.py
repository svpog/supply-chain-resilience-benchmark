"""
Informed-greedy upper reference для M5-ceiling.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §4.2
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    >>> from src.recovery.agents import create_agent
    >>> agent = create_agent("offline_oracle", agent_id=0, r=None, network=network)
"""

from __future__ import annotations

from src.network.flow import compute_max_flow
from src.recovery.agents.base_agent import AgentBase


class OfflineOracleAgent(AgentBase):
    """
    Обирає failed-вузол з максимальним глобальним Δmax_flow.

    Це НЕ оптимальний планувальник і НЕ "стеля задачі". Агент є
    informed-greedy upper reference: повна видимість, той самий бюджет K=3
    під token, і жадібна одношагова оцінка на повному графі.
    """

    def choose_action(self, observation: dict, locked: set | None = None):
        """Повертає node_id з максимальним глобальним gain або None."""
        locked = locked or set()
        network = observation["network"]
        candidates = sorted(set(network.failed_nodes) - set(locked))
        if not candidates:
            return None

        F_before = compute_max_flow(network)
        best_node = None
        best_gain = float("-inf")
        for node in candidates:
            trial = network.copy()
            trial.set_node_state(node, "active")
            gain = float(compute_max_flow(trial) - F_before)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        return best_node

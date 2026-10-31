"""
Broadcast coordinator для multi-agent recovery M3.

Реалізує: Math Spec Article 4 §3.5 та Протокол M3, §4.2
(див. docs/Math_Spec_Article_4_v4_1.docx, docs/Протокол_M3.docx).
"""

from __future__ import annotations

from src.recovery.coordination.base_coordinator import CoordinatorBase


def _node_sort_key(node) -> tuple:
    """Стабільний tie-break для node_id різних типів."""
    if isinstance(node, (int, float, str)):
        return (0, node)
    return (1, repr(node))


class BroadcastCoordinator(CoordinatorBase):
    """
    Broadcast-координація через об'єднання локальних спостережень агентів.

    Агенти не отримують повний граф: shared pool дорівнює union видимих
    failed-вузлів і union visible_nodes з їхніх O_k(t). Далі top-K вузлів
    за локальним flow-impact призначаються агентам у стабільному порядку.
    """

    def allocate(self, agents: list, network, failed_nodes: set) -> dict[int, object | None]:
        observations = {
            agent.agent_id: agent.observe(network, failed_nodes)
            for agent in agents
        }
        shared_nodes: set = set()
        shared_failed: set = set()
        for obs in observations.values():
            shared_nodes.update(obs["visible_nodes"])
            shared_failed.update(obs["visible_failed"])

        actions: dict[int, object | None] = {agent.agent_id: None for agent in agents}
        if not shared_failed:
            return actions

        shared_obs = {
            "network": network,
            "visible_nodes": shared_nodes,
            "visible_failed": shared_failed,
            "G_sub": network.graph.subgraph(shared_nodes).copy(),
        }
        score_agent = sorted(agents, key=lambda agent: agent.agent_id)[0]
        gains = {
            node: float(score_agent.estimate_gain(node, shared_obs))
            for node in shared_failed
        }
        ranked_nodes = sorted(
            shared_failed,
            key=lambda node: (-gains[node], _node_sort_key(node)),
        )

        for agent, node in zip(sorted(agents, key=lambda a: a.agent_id), ranked_nodes):
            actions[agent.agent_id] = node
        return actions

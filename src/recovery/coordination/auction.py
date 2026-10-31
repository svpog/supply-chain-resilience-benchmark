"""
Auction coordinator для multi-agent recovery M3.

Реалізує: Math Spec Article 4 §3.5 та Протокол M3, §4.1
(див. docs/Math_Spec_Article_4_v4_1.docx, docs/Протокол_M3.docx).
"""

from __future__ import annotations

from src.recovery.coordination.base_coordinator import CoordinatorBase


def _node_sort_key(node) -> tuple:
    """Стабільний tie-break для node_id різних типів."""
    if isinstance(node, (int, float, str)):
        return (0, node)
    return (1, repr(node))


class AuctionCoordinator(CoordinatorBase):
    """
    Аукціон repair-дій: вузол отримує агент з найвищим локальним bid.

    Bid = agent.estimate_gain(v, observation). Після сортування за спаданням
    bid діє mutex і на агентів, і на вузли: один агент ремонтує не більше
    одного вузла, а один вузол не може бути призначений двічі.
    """

    def allocate(self, agents: list, network, failed_nodes: set) -> dict[int, object | None]:
        actions: dict[int, object | None] = {agent.agent_id: None for agent in agents}
        bids: list[tuple[int, object, float]] = []

        for agent in agents:
            obs = agent.observe(network, failed_nodes)
            for node in sorted(obs["visible_failed"], key=_node_sort_key):
                bids.append((agent.agent_id, node, float(agent.estimate_gain(node, obs))))

        assigned_agents: set[int] = set()
        assigned_nodes: set = set()
        for agent_id, node, _bid in sorted(
            bids,
            key=lambda item: (-item[2], _node_sort_key(item[1]), item[0]),
        ):
            if agent_id in assigned_agents or node in assigned_nodes:
                continue
            actions[agent_id] = node
            assigned_agents.add(agent_id)
            assigned_nodes.add(node)
        return actions

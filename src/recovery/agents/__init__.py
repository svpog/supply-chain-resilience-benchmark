"""
Агенти multi-agent recovery для статті 4.

Реалізує: Протокол M1, §4.2
(див. docs/Протокол_M1.docx).
"""

from src.recovery.agents.base_agent import AgentBase
from src.recovery.agents.coordinated_greedy import CoordinatedGreedyAgent
from src.recovery.agents.local_centrality_agent import LocalCentralityAgent
from src.recovery.agents.local_greedy_agent import LocalGreedyAgent
from src.recovery.agents.offline_oracle_agent import OfflineOracleAgent
from src.recovery.agents.random_agent import RandomAgent
from src.recovery.agents.rl_policy_agent import RLPolicyAgent


AGENT_REGISTRY = {
    "random_agent": RandomAgent,
    "local_greedy": LocalGreedyAgent,
    "local_centrality": LocalCentralityAgent,
    "coord_greedy": CoordinatedGreedyAgent,
    "offline_oracle": OfflineOracleAgent,
    "rl_policy": RLPolicyAgent,
}


def create_agent(
    strategy: str,
    agent_id: int,
    r: int | None,
    network,
    rng=None,
    **kwargs,
) -> AgentBase:
    """Створює агента M1 за назвою стратегії."""
    try:
        cls = AGENT_REGISTRY[strategy]
    except KeyError as exc:
        allowed = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Невідомий agent strategy '{strategy}'. Дозволені: {allowed}.") from exc
    return cls(agent_id=agent_id, r=r, network=network, rng=rng, **kwargs)


__all__ = [
    "AGENT_REGISTRY",
    "AgentBase",
    "CoordinatedGreedyAgent",
    "LocalCentralityAgent",
    "LocalGreedyAgent",
    "OfflineOracleAgent",
    "RandomAgent",
    "RLPolicyAgent",
    "create_agent",
]

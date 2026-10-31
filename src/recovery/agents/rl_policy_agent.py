"""
RL policy-agent для M5-ceiling.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §4.3
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    >>> agent = RLPolicyAgent(agent_id=0, r=2, network=network)
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from src.network.flow import compute_max_flow
from src.recovery.agents.base_agent import AgentBase
from src.rl.observation import SetObservationEncoder


DEFAULT_MODEL_PATH = Path("models/article4/rl_seed0.zip")


class RLPolicyAgent(AgentBase):
    """
    Агент, що делегує вибір вузла навченій SB3 PPO політиці.

    Модель приймає той самий set-encoded observation, що використовується
    RecoveryEnv. Action index мапиться на sorted visible candidates; остання
    дія action_space відповідає wait.
    """

    def __init__(
        self,
        agent_id: int,
        r: int | None,
        network,
        rng=None,
        model_path: str | Path | None = None,
        encoder: SetObservationEncoder | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, r=r, network=network, rng=rng)
        self.encoder = encoder if encoder is not None else SetObservationEncoder()
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Не знайдено RL model: {self.model_path}. "
                "Спершу запустіть src/rl/train.py або передайте model_path."
            )
        self.model = PPO.load(self.model_path)

    def choose_action(self, observation: dict, locked: set | None = None):
        """Повертає node_id з PPO policy або None для wait/invalid action."""
        locked = locked or set()
        F_norm = max(1.0, compute_max_flow(observation["network"]))
        encoded, candidates = self.encoder.encode(self, observation, locked=set(locked), F0=F_norm)
        if not candidates:
            return None
        action, _ = self.model.predict(encoded, deterministic=True)
        action_idx = int(action)
        if action_idx >= len(candidates):
            return None
        return candidates[action_idx]

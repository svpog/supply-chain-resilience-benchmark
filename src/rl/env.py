"""
Gymnasium-обгортка recovery episode для M5-ceiling.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §4.3
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    >>> env = RecoveryEnv(cfg, seed_range=range(7000, 7010))
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.disruptions.node_failure import NodeFailureDisruption
from src.experiments.multi_agent_runner import _create_network, _has_active_supply_and_demand, _stable_seed
from src.metrics.integral import compute_integral_metrics
from src.network.flow import compute_max_flow
from src.recovery.agents.local_greedy_agent import LocalGreedyAgent
from src.rl.observation import SetObservationEncoder


class RecoveryEnv(gym.Env):
    """
    Shared-policy IPPO-style середовище для одного агента під token.

    Кожен виклик step(action) відповідає дії одного з K агентів. Після K
    дій token timestep завершується, і середовище переходить до наступного
    recovery-кроку. Збій статичний: застосовується один раз у reset().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict,
        topology: str = "hierarchical",
        N: int = 50,
        seed_range: Iterable[int] = range(7000, 8000),
        encoder: SetObservationEncoder | None = None,
    ) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.topology = topology
        self.N = int(N)
        self.seed_values = list(seed_range)
        if not self.seed_values:
            raise ValueError("RecoveryEnv потребує непорожній seed_range.")
        self.encoder = encoder if encoder is not None else SetObservationEncoder()
        self.action_space = spaces.Discrete(self.encoder.max_candidates + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.encoder.obs_dim,),
            dtype=np.float32,
        )
        self._episode_idx = 0
        self.network = None
        self.F0 = 0.0
        self.t = 0
        self.agent_index = 0
        self.locked: set = set()
        self.Q_curve: list[float] = []
        self.current_candidates: list = []
        self.agent = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Створює нову мережу+збій із train/test seed namespace."""
        super().reset(seed=seed)
        episode_seed = int(seed) if seed is not None else self.seed_values[self._episode_idx % len(self.seed_values)]
        self._episode_idx += 1
        self.network, self.F0 = self._make_valid_episode(episode_seed)
        self.t = 1
        self.agent_index = 0
        self.locked = set()
        self.agent = LocalGreedyAgent(agent_id=0, r=self._r_visibility(), network=self.network)
        self.Q_curve = [1.0]
        self._append_current_Q()
        obs = self._encode_current_observation()
        return obs, {"seed": episode_seed, "F0": self.F0, "t": self.t, "action_mask": self._action_mask()}

    def step(self, action: int):
        """Виконує repair action для поточного agent slot."""
        if self.network is None:
            raise RuntimeError("Викличте reset() перед step().")
        reward = 0.0
        rho_final = None
        terminated = False
        truncated = False
        node = self._action_to_node(int(action))
        if node is not None and node in self.network.failed_nodes:
            obs_before = self.agent.observe(self.network, set(self.network.failed_nodes))
            reward = float(self.agent.estimate_gain(node, obs_before)) / max(self.F0, 1.0)
            self.network.set_node_state(node, "active")
            self.locked.add(node)

        self.agent_index += 1
        if self.agent_index >= self._K_agents() or not self.network.failed_nodes:
            self.agent_index = 0
            self.locked = set()
            if not self.network.failed_nodes or self.t >= self._T_max():
                terminated = not self.network.failed_nodes
                truncated = self.t >= self._T_max() and bool(self.network.failed_nodes)
                rho_final = self._final_rho()
                reward += self._terminal_bonus(rho_final)
            else:
                self.t += 1
                self._append_current_Q()

        obs = self._encode_current_observation()
        info = {
            "t": self.t,
            "agent_index": self.agent_index,
            "Q_current": self.Q_curve[-1] if self.Q_curve else 0.0,
            "failed_nodes": len(self.network.failed_nodes),
            "rho_final": rho_final,
            "action_mask": self._action_mask(),
        }
        return obs, float(reward), terminated, truncated, info

    def _make_valid_episode(self, episode_seed: int):
        """Генерує валідний епізод; invalid seeds пропускає детерміновано."""
        for offset in range(len(self.seed_values) + 10):
            seed = episode_seed + offset
            cfg = deepcopy(self.cfg)
            cfg.setdefault("network", {})["N"] = self.N
            net0 = _create_network(cfg, topology=self.topology, seed=_stable_seed("M5", seed, "network"))
            F0 = compute_max_flow(net0)
            if F0 <= 0.0:
                continue
            disrupted = NodeFailureDisruption(p_v=float(cfg["disruption"].get("p_v", 0.20))).apply(
                net0,
                seed=_stable_seed("M5", seed, "disruption"),
            )
            if _has_active_supply_and_demand(disrupted):
                return disrupted, float(F0)
        raise RuntimeError("Не вдалося згенерувати валідний M5-ceiling епізод.")

    def _append_current_Q(self) -> None:
        """Додає Q(t) у стилі runner: вимір перед repair-діями timestep."""
        F_t = compute_max_flow(self.network)
        self.Q_curve.append(float(np.clip(F_t / self.F0 if self.F0 > 0.0 else 0.0, 0.0, 1.0)))

    def _encode_current_observation(self) -> np.ndarray:
        """Кодує observation поточного agent slot."""
        if self.network is None or self.agent is None:
            return np.zeros(self.encoder.obs_dim, dtype=np.float32)
        obs = self.agent.observe(self.network, set(self.network.failed_nodes))
        encoded, candidates = self.encoder.encode(self.agent, obs, locked=self.locked, F0=self.F0)
        self.current_candidates = candidates
        return encoded

    def _action_to_node(self, action: int):
        """Мапить discrete action на candidate node; остання дія = wait."""
        if action >= self.encoder.max_candidates:
            return None
        if action < 0 or action >= len(self.current_candidates):
            return None
        return self.current_candidates[action]

    def _action_mask(self) -> np.ndarray:
        """Повертає Gymnasium-compatible mask для видимих repair-кандидатів."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.current_candidates:
            mask[: len(self.current_candidates)] = 1
        else:
            mask[self.encoder.max_candidates] = 1
        return mask

    def _final_rho(self) -> float:
        """Повертає фінальну rho епізоду для terminal info та reward bonus."""
        q_curve = self._final_Q_curve()
        q_target = float(self.cfg.get("simulation", {}).get("Q_target_frac", 0.95)) * q_curve[0]
        metrics = compute_integral_metrics(q_curve, t_e=1, Q_target=q_target)
        return float(metrics["rho"])

    def _final_Q_curve(self) -> list[float]:
        """Добиває terminal Q_curve до повного горизонту [0, T_max]."""
        curve = list(self.Q_curve)
        T_max = self._T_max()
        if self.network is not None and not self.network.failed_nodes and len(curve) <= T_max:
            F_t = compute_max_flow(self.network)
            curve.append(float(np.clip(F_t / self.F0 if self.F0 > 0.0 else 0.0, 0.0, 1.0)))
        if curve:
            curve.extend([curve[-1]] * max(0, T_max + 1 - len(curve)))
        return curve[: T_max + 1]

    def _terminal_bonus(self, rho_final: float | None = None) -> float:
        """Малий β·ρ_final бонус, узгоджений з основною метрикою."""
        beta = float(self.cfg.get("constants", {}).get("reward_beta", 0.1))
        rho = self._final_rho() if rho_final is None else float(rho_final)
        return beta * rho

    def _K_agents(self) -> int:
        return int(self.cfg.get("constants", {}).get("K_agents", 3))

    def _T_max(self) -> int:
        return int(self.cfg.get("simulation", {}).get("T_max", self.cfg.get("constants", {}).get("T", 100)))

    def _r_visibility(self) -> int | None:
        return self.cfg.get("constants", {}).get("r_visibility", 2)

"""
Unit-тести M5-ceiling RL Gym-обгортки.

Реалізує: Протокол M5-ceiling, §4.3 smoke/sanity.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recovery.agents import AGENT_REGISTRY
from src.rl.env import RecoveryEnv
from src.rl.observation import SetObservationEncoder
from src.rl.train import assert_seed_split


def make_m5_cfg() -> dict:
    """Мінімальний config для RecoveryEnv tests."""
    return {
        "series": "M5ceiling",
        "network": {"N": 30, "H": 4, "b": 3, "c_base": 100, "K_res": 2, "h": 3, "k_avg": 5.0},
        "disruption": {"type": "node_failure", "p_v": 0.20},
        "simulation": {"T_max": 20, "Q_target_frac": 0.95},
        "constants": {"K_agents": 3, "r_visibility": 2, "reward_beta": 0.1},
    }


def test_observation_encoder_fixed_shape() -> None:
    """Encoder повертає фіксований shape незалежно від кількості кандидатів."""
    env = RecoveryEnv(make_m5_cfg(), N=30, seed_range=range(7000, 7003), encoder=SetObservationEncoder(max_candidates=32))
    obs, _ = env.reset(seed=7000)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.action_space.n == 33


def test_recovery_env_random_episode_runs() -> None:
    """Один random-policy episode проходить без винятків і накопичує Q_curve."""
    env = RecoveryEnv(make_m5_cfg(), N=30, seed_range=range(7000, 7003), encoder=SetObservationEncoder(max_candidates=32))
    obs, _ = env.reset(seed=7001)
    done = False
    steps = 0
    while not done and steps < 200:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1

    assert steps > 0
    assert len(env.Q_curve) >= 2
    assert all(0.0 <= q <= 1.0 for q in env.Q_curve)


def test_recovery_env_reset_is_deterministic_for_seed() -> None:
    """Однаковий explicit seed дає однаковий стартовий Q після disruption."""
    cfg = make_m5_cfg()
    env_a = RecoveryEnv(cfg, N=30, seed_range=range(7000, 7003), encoder=SetObservationEncoder(max_candidates=32))
    env_b = RecoveryEnv(cfg, N=30, seed_range=range(7000, 7003), encoder=SetObservationEncoder(max_candidates=32))
    env_a.reset(seed=7002)
    env_b.reset(seed=7002)

    assert env_a.Q_curve == env_b.Q_curve
    assert env_a.network.failed_nodes == env_b.network.failed_nodes


def test_rl_policy_registered() -> None:
    """rl_policy є доступною agent strategy, але модель вантажиться лише при init."""
    assert "rl_policy" in AGENT_REGISTRY


def test_train_test_seed_split_rejects_overlap() -> None:
    """Train/test seed namespaces M5-ceiling не мають перетинатися."""
    assert_seed_split(7000, 8000)
    with pytest.raises(ValueError):
        assert_seed_split(7000, 7500)

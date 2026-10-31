"""
Навчання PPO policy для M5-ceiling.

Реалізує: Patch v4.9 та Протокол M5-ceiling, §4.3
(див. docs/Math_Spec_Article_4_Patch_v4_9.docx).

Приклад використання:
    python -m src.rl.train --episodes 5000 --n-seeds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.experiments.multi_agent_runner import load_config
from src.rl.env import RecoveryEnv


MODEL_DIR = Path("models/article4")
LOG_DIR = Path("results/raw/article4/M5ceiling")


class RhoEpisodeCallback(BaseCallback):
    """Логує terminal rho_final і зупиняє PPO після заданої кількості епізодів."""

    def __init__(self, target_episodes: int, train_seed: int, seed_idx: int) -> None:
        super().__init__(verbose=0)
        self.target_episodes = int(target_episodes)
        self.train_seed = int(train_seed)
        self.seed_idx = int(seed_idx)
        self.rows: list[dict[str, object]] = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episode_count += 1
            self.rows.append(
                {
                    "seed_idx": self.seed_idx,
                    "train_seed": self.train_seed,
                    "episode": self.episode_count,
                    "timestep": int(self.num_timesteps),
                    "rho_final": info.get("rho_final"),
                    "Q_current": info.get("Q_current"),
                    "failed_nodes": info.get("failed_nodes"),
                }
            )
            if self.episode_count >= self.target_episodes:
                return False
        return True


def assert_seed_split(train_seed_base: int, test_seed_base: int) -> None:
    """Гарантує неперетин train/test namespaces M5-ceiling."""
    train_range = set(range(train_seed_base, train_seed_base + 1000))
    test_range = set(range(test_seed_base, test_seed_base + 500))
    if train_range.intersection(test_range):
        raise ValueError("Train/test seed namespaces перетинаються.")


def train_one_seed(
    cfg: dict,
    train_seed: int,
    seed_idx: int,
    episodes: int,
    model_dir: Path = MODEL_DIR,
    log_dir: Path = LOG_DIR,
) -> Path:
    """Навчає один PPO model seed і зберігає model + training log."""
    constants = cfg["constants"]
    train_seed_base = int(constants["train_seed_base"])
    test_seed_base = int(constants["test_seed_base"])
    assert_seed_split(train_seed_base, test_seed_base)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = RecoveryEnv(
        cfg=cfg,
        topology=str(constants.get("topology", "hierarchical")),
        N=int(constants.get("N", 50)),
        seed_range=range(train_seed_base, train_seed_base + 1000),
    )
    env.action_space.seed(train_seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=train_seed,
        verbose=0,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
    )
    callback = RhoEpisodeCallback(target_episodes=int(episodes), train_seed=train_seed, seed_idx=seed_idx)
    # Верхня межа у agent-slot timesteps; callback зупиняє навчання рівно на target_episodes.
    total_timesteps = int(episodes) * int(constants.get("K_agents", 3)) * int(constants.get("T", 100)) + 1
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model_path = model_dir / f"rl_seed{seed_idx}.zip"
    model.save(model_path)
    log_rows = callback.rows or [{
        "seed_idx": seed_idx,
        "train_seed": train_seed,
        "episode": 0,
        "timestep": 0,
        "rho_final": np.nan,
        "Q_current": np.nan,
        "failed_nodes": np.nan,
    }]
    log_df = pd.DataFrame(log_rows)
    log_df["target_episodes"] = int(episodes)
    log_df["total_timesteps_budget"] = total_timesteps
    log_df["model_path"] = str(model_path)
    log_df["train_seed_start"] = train_seed_base
    log_df["train_seed_end"] = train_seed_base + 999
    log_df["test_seed_start"] = test_seed_base
    log_df["test_seed_end"] = test_seed_base + 399
    log_df.to_parquet(log_dir / f"training_log_seed{seed_idx}.parquet", index=False)
    return model_path


def train_models(cfg: dict, episodes: int | None = None, n_seeds: int | None = None) -> list[Path]:
    """Навчає n незалежних PPO seeds для M5-ceiling."""
    constants = cfg["constants"]
    episodes = int(episodes if episodes is not None else constants.get("rl_episodes", 5000))
    n_seeds = int(n_seeds if n_seeds is not None else constants.get("n_train_seeds", 5))
    train_seed_base = int(constants["train_seed_base"])
    test_seed_base = int(constants["test_seed_base"])
    assert_seed_split(train_seed_base, test_seed_base)
    return [
        train_one_seed(cfg, train_seed=train_seed_base + seed_idx, seed_idx=seed_idx, episodes=episodes)
        for seed_idx in range(n_seeds)
    ]


def main() -> None:
    """CLI entrypoint для PPO training."""
    parser = argparse.ArgumentParser(description="Train M5-ceiling PPO policies.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated model seed indices, e.g. 0,1,2,3,4.")
    parser.add_argument("--save-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--r", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config("M5ceiling")
    if args.topology is not None:
        cfg.setdefault("constants", {})["topology"] = args.topology
        cfg.setdefault("network", {})["topology"] = args.topology
    if args.N is not None:
        cfg.setdefault("constants", {})["N"] = int(args.N)
        cfg.setdefault("network", {})["N"] = int(args.N)
    if args.K is not None:
        cfg.setdefault("constants", {})["K_agents"] = int(args.K)
    if args.r is not None:
        cfg.setdefault("constants", {})["r_visibility"] = None if args.r in {"inf", "None", "none"} else int(args.r)
    cfg.setdefault("constants", {})["rl_model_dir"] = args.save_dir

    if args.seeds is not None:
        seed_indices = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        episodes = int(args.episodes if args.episodes is not None else cfg["constants"].get("rl_episodes", 5000))
        train_seed_base = int(cfg["constants"]["train_seed_base"])
        assert_seed_split(train_seed_base, int(cfg["constants"]["test_seed_base"]))
        paths = [
            train_one_seed(
                cfg,
                train_seed=train_seed_base + seed_idx,
                seed_idx=seed_idx,
                episodes=episodes,
                model_dir=Path(args.save_dir),
                log_dir=Path(args.log_dir),
            )
            for seed_idx in seed_indices
        ]
    else:
        episodes = int(args.episodes if args.episodes is not None else cfg["constants"].get("rl_episodes", 5000))
        n_seeds = int(args.n_seeds if args.n_seeds is not None else cfg["constants"].get("n_train_seeds", 5))
        train_seed_base = int(cfg["constants"]["train_seed_base"])
        assert_seed_split(train_seed_base, int(cfg["constants"]["test_seed_base"]))
        paths = [
            train_one_seed(
                cfg,
                train_seed=train_seed_base + seed_idx,
                seed_idx=seed_idx,
                episodes=episodes,
                model_dir=Path(args.save_dir),
                log_dir=Path(args.log_dir),
            )
            for seed_idx in range(n_seeds)
        ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

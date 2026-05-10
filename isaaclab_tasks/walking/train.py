"""Training entry point.

Usage (inside Docker / Isaac Sim python):
    python isaaclab_tasks/walking/train.py task=stand_v1
    python isaaclab_tasks/walking/train.py task=walk_v1 num_envs=2048
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import torch

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = cfg.seed
    torch.manual_seed(seed)

    # ── Run dir ───────────────────────────────────────────────────────────────
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save full resolved config
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Save git commit hash
    _save_git_info(run_dir)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if cfg.wandb.enabled:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity") or os.environ.get("WANDB_ENTITY"),
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.tags),
            dir=str(run_dir),
        )

    # ── Isaac Lab env + runner ────────────────────────────────────────────────
    # Import Isaac Lab here so the script can be imported without Isaac installed
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.walking import StandV1EnvCfg, WalkV1EnvCfg

    task_map = {
        "stand_v1": StandV1EnvCfg,
        "walk_v1": WalkV1EnvCfg,
    }
    env_cfg_cls = task_map[cfg.env.name]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # RSL-RL runner (bundled with Isaac Lab 2.x)
    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    runner_cfg = RslRlOnPolicyRunnerCfg(**OmegaConf.to_container(cfg.train.runner))
    vec_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(vec_env, runner_cfg, log_dir=str(run_dir), device=cfg.device)
    runner.learn(num_learning_iterations=cfg.train.runner.max_iterations)

    # Save final checkpoint to run dir
    runner.save(str(run_dir / "model.pt"))

    env.close()
    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


def _save_git_info(run_dir: Path) -> None:
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "status", "--short"], text=True
        )
        (run_dir / "git_commit.txt").write_text(f"{commit}\n\n{diff}")
    except Exception:
        (run_dir / "git_commit.txt").write_text("git info unavailable\n")


if __name__ == "__main__":
    main()

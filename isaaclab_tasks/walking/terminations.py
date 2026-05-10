"""Termination and reset functions."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End episode when max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End episode when base drops below threshold height — robot has fallen."""
    robot = env.scene[asset_cfg.name]
    base_height = robot.data.root_pos_w[:, 2]
    return base_height < threshold


def reset_robot_state(env: ManagerBasedRLEnv) -> None:
    """Reset robot to default pose with small random perturbations."""
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Small random perturbation to avoid identical resets
    joint_pos += torch.randn_like(joint_pos) * 0.02

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins  # place in correct env slot

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

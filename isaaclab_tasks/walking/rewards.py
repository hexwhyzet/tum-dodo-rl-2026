"""Reward functions for locomotion tasks.

Each function receives an IsaacLab ManagerBasedRLEnv and returns a 1-D tensor
of shape (num_envs,).

All functions should be fast — they run every physics step for thousands of envs.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


# ── Stability ─────────────────────────────────────────────────────────────────

def upright_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for keeping the base close to upright.

    Uses projected gravity: when perfectly upright, gravity projects to (0, 0, -1)
    in body frame. We reward alignment of the z-component with -1.
    """
    gravity_b = env.scene["robot"].data.projected_gravity_b
    return torch.clip(-gravity_b[:, 2], 0.0, 1.0)


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant +1 per step as long as episode is running."""
    return torch.ones(env.num_envs, device=env.device)


# ── Velocity tracking ─────────────────────────────────────────────────────────

def lin_vel_tracking(env: ManagerBasedRLEnv, sigma: float = 0.25) -> torch.Tensor:
    """Exponential reward for matching commanded linear velocity."""
    cmd = env.command_manager.get_command("base_velocity")
    vel = env.scene["robot"].data.root_lin_vel_b
    error = torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1)
    return torch.exp(-error / sigma)


def ang_vel_tracking(env: ManagerBasedRLEnv, sigma: float = 0.25) -> torch.Tensor:
    """Exponential reward for matching commanded yaw rate."""
    cmd = env.command_manager.get_command("base_velocity")
    ang_vel_z = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error = torch.square(cmd[:, 2] - ang_vel_z)
    return torch.exp(-error / sigma)


# ── Penalties ─────────────────────────────────────────────────────────────────

def energy_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise joint torques squared — encourages efficient motion."""
    torques = env.scene["robot"].data.applied_torque
    return torch.sum(torch.square(torques), dim=1)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise large changes in action between consecutive steps."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def dof_pos_limits_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise joints that exceed soft limits."""
    robot = env.scene["robot"]
    out_of_limits = (
        torch.clip(robot.data.joint_pos - robot.data.soft_joint_pos_limits[:, :, 1], min=0.0)
        + torch.clip(robot.data.soft_joint_pos_limits[:, :, 0] - robot.data.joint_pos, min=0.0)
    )
    return torch.sum(out_of_limits, dim=1)


def collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise undesired body contacts (non-foot links touching ground)."""
    # TODO: configure body_ids for non-foot links after robot is imported
    net_forces = env.scene["contact_sensor"].data.net_forces_w
    return torch.any(torch.norm(net_forces, dim=-1) > 1.0, dim=1).float()


def foot_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward swing feet for being above the ground."""
    # TODO: implement after foot body names are known
    return torch.zeros(env.num_envs, device=env.device)

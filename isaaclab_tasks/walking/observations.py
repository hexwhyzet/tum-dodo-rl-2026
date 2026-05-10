"""Observation functions for locomotion tasks.

Each function receives an IsaacLab ManagerBasedRLEnv and returns a tensor
of shape (num_envs, obs_dim).
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint positions relative to default pose."""
    return env.scene["robot"].data.joint_pos - env.scene["robot"].data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities relative to default (usually 0)."""
    return env.scene["robot"].data.joint_vel


def base_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base angular velocity in body frame."""
    return env.scene["robot"].data.root_ang_vel_b


def projected_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gravity vector projected into body frame — encodes tilt."""
    return env.scene["robot"].data.projected_gravity_b


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action sent to actuators."""
    return env.action_manager.action


def base_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base linear velocity in body frame."""
    return env.scene["robot"].data.root_lin_vel_b


def velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Commanded velocities [vx, vy, yaw_rate] from the command manager."""
    return env.command_manager.get_command("base_velocity")

"""Environment configs for standing and walking tasks (Dodo Daimao biped).

Robot joints (from dodo_daimao.csv):
  hip_right       — roll,  effort 27 Nm, vel 6 rad/s, limits [-0.35, 0.35]
  upper_leg_right — pitch, effort 27 Nm, vel 6 rad/s, limits [-1.57, 1.57]
  lower_leg_right — knee,  effort  9 Nm, vel 6 rad/s, limits [-1.3963, 3.1416]
  foot_right      — ankle, effort  9 Nm, vel 6 rad/s, limits [-1.57, 1.05]
  hip_left        — roll,  effort 27 Nm, vel 6 rad/s, limits [-0.35, 0.35]
  upper_leg_left  — pitch, effort 27 Nm, vel 6 rad/s, limits [-1.57, 1.57]
  lower_leg_left  — knee,  effort  9 Nm, vel 6 rad/s, limits [-1.3963, 3.1416]
  foot_left       — ankle, effort  9 Nm, vel 6 rad/s, limits [-1.57, 1.05]
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import observations as obs_fns
from . import rewards as rew_fns
from . import terminations as term_fns


# ── Robot asset ───────────────────────────────────────────────────────────────

# Main USD is dodo_ROS.usd; it references sub-layers in usd/configuration/.
ROBOT_USD_PATH = "assets/robot/usd/dodo_ROS.usd"

# Approximate standing height: body CoM z ≈ 0.085 m, legs extend ~0.4 m below
ROBOT_SPAWN_HEIGHT = 0.55  # metres above ground — tweak after first visual check

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_SPAWN_HEIGHT),
        joint_pos={
            # Default pose: legs slightly bent, feet flat
            # Hip roll — symmetric, zero
            "hip_right": 0.0,
            "hip_left":  0.0,
            # Hip pitch — slight forward lean to load legs
            "upper_leg_right": 0.2,
            "upper_leg_left":  0.2,
            # Knee — bent to absorb load
            "lower_leg_right": -0.4,
            "lower_leg_left":  -0.4,
            # Ankle — compensate for knee bend so foot stays flat
            "foot_right": 0.2,
            "foot_left":  0.2,
        },
    ),
    actuators={
        # Hip joints: high-torque motors (27 Nm)
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["hip_right", "hip_left"],
            effort_limit=27.0,
            velocity_limit=6.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Thigh joints: high-torque motors (27 Nm)
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=["upper_leg_right", "upper_leg_left"],
            effort_limit=27.0,
            velocity_limit=6.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Knee + ankle: lower-torque motors (9 Nm)
        "knees_ankles": ImplicitActuatorCfg(
            joint_names_expr=["lower_leg_right", "lower_leg_left", "foot_right", "foot_left"],
            effort_limit=9.0,
            velocity_limit=6.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)


# ── Scene ─────────────────────────────────────────────────────────────────────

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


# ── Observations ─────────────────────────────────────────────────────────────

@configclass
class ProprioObsCfg(ObsGroup):
    """Proprioceptive observations only — no exteroception."""

    joint_pos = ObsTerm(func=obs_fns.joint_pos_rel)
    joint_vel = ObsTerm(func=obs_fns.joint_vel_rel)
    base_ang_vel = ObsTerm(func=obs_fns.base_ang_vel)
    projected_gravity = ObsTerm(func=obs_fns.projected_gravity)
    last_action = ObsTerm(func=obs_fns.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: ProprioObsCfg = ProprioObsCfg()


# ── Actions ───────────────────────────────────────────────────────────────────
# Defined inline in env cfg via isaac lab action managers.
# Using joint position targets with PD control.


# ── Rewards ───────────────────────────────────────────────────────────────────

@configclass
class StandRewardsCfg:
    upright = RewTerm(func=rew_fns.upright_bonus, weight=2.0)
    alive = RewTerm(func=rew_fns.alive_bonus, weight=0.5)
    energy = RewTerm(func=rew_fns.energy_penalty, weight=-0.001)
    action_rate = RewTerm(func=rew_fns.action_rate_penalty, weight=-0.01)
    dof_pos_limits = RewTerm(func=rew_fns.dof_pos_limits_penalty, weight=-1.0)


@configclass
class WalkRewardsCfg:
    lin_vel_tracking = RewTerm(
        func=rew_fns.lin_vel_tracking, weight=2.0, params={"sigma": 0.25}
    )
    ang_vel_tracking = RewTerm(
        func=rew_fns.ang_vel_tracking, weight=0.5, params={"sigma": 0.25}
    )
    upright = RewTerm(func=rew_fns.upright_bonus, weight=1.0)
    alive = RewTerm(func=rew_fns.alive_bonus, weight=0.5)
    energy = RewTerm(func=rew_fns.energy_penalty, weight=-0.001)
    action_rate = RewTerm(func=rew_fns.action_rate_penalty, weight=-0.01)
    foot_clearance = RewTerm(func=rew_fns.foot_clearance, weight=0.2)
    dof_pos_limits = RewTerm(func=rew_fns.dof_pos_limits_penalty, weight=-1.0)
    collision = RewTerm(func=rew_fns.collision_penalty, weight=-1.0)


# ── Terminations ──────────────────────────────────────────────────────────────

@configclass
class TerminationsCfg:
    timeout = DoneTerm(func=term_fns.time_out, time_out=True)
    falling = DoneTerm(
        func=term_fns.base_height_below_threshold,
        params={"threshold": 0.3},
    )


# ── Events (resets, pushes) ───────────────────────────────────────────────────

@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=term_fns.reset_robot_state,
        mode="reset",
    )


# ── Root env configs ──────────────────────────────────────────────────────────

@configclass
class StandV1EnvCfg(ManagerBasedRLEnvCfg):
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4                # policy at 50 Hz, physics at 200 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.005                # 200 Hz physics
        self.sim.render_interval = self.decimation


@configclass
class WalkV1EnvCfg(StandV1EnvCfg):
    rewards: WalkRewardsCfg = WalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0

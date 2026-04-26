"""
System 0 Skill: Release Gently (28D observations).

Standalone ManagerBasedRLEnv for the release specialist.
Uses BlockStackConfig for all constants (NOT System0Config).

Object starts grasped (fingers closed around block at place position).
The policy must open fingers gradually so the block lands on the table
without toppling.

Observations (28D):
  finger_pos(7) + finger_vel(7) + force_mag(3) + contact_binary(3) + phase_onehot(8)
  Phase onehot is always RELEASE_HOLD (index 6 of 8).

Actions (7D): right hand finger joint position deltas, scale=1.5.

Note: train_release.py uses BlockStackEnvCfg directly (not this file).
This env is provided for standalone evaluation / debugging of the release
specialist in isolation.
"""

import os
import sys
import math
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

from experiments.system0_skills.block_stack_config import BlockStackConfig

CFG = BlockStackConfig()

RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices

# Phase index for RELEASE_HOLD (from arm_trajectory.Phase)
RELEASE_HOLD_PHASE_ID = 6
NUM_PHASES = 8

# Finger open/closed reference positions for closure fraction computation
FINGER_OPEN = [0.08, 0.08, 0.0, 0.09, 0.09, 0.0, -0.09]
FINGER_CLOSED = [0.45, 0.45, -0.28, 0.50, 0.50, -0.29, -0.50]

project_root = os.environ.get("PROJECT_ROOT")


def _make_release_robot_cfg() -> ArticulationCfg:
    """Robot with arm at PLACE position (sp=0.0, sr=-0.5, sy=0.1, el=0.0).
    Fingers start CLOSED (grasping position). Legs/waist locked."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=CFG.robot_pos,
            rot=CFG.robot_rot,
            joint_pos={
                # Legs
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.05,
                "left_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.15,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.05,
                "right_knee_joint": 0.2,
                "right_ankle_pitch_joint": -0.15,
                "right_ankle_roll_joint": 0.0,
                # Waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                # Left arm -- zero
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                # Right arm -- PLACE position (same as arm_grasp_joints in config)
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.5,
                "right_shoulder_yaw_joint": 0.1,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Left hand -- zero
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand -- start CLOSED (grasping position)
                "right_hand_index_0_joint": 0.45,
                "right_hand_middle_0_joint": 0.45,
                "right_hand_thumb_0_joint": -0.28,
                "right_hand_index_1_joint": 0.50,
                "right_hand_middle_1_joint": 0.50,
                "right_hand_thumb_1_joint": -0.29,
                "right_hand_thumb_2_joint": -0.50,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                                  ".*_hip_pitch_joint", ".*_knee_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_hand_.*_joint"],
                effort_limit=300.0, velocity_limit=100.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_hand_.*_joint"],
                effort_limit=300.0, velocity_limit=100.0,
                stiffness={".*": CFG.hand_stiffness}, damping={".*": CFG.hand_damping},
                armature={".*": 0.1},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Observation functions (28D total)
# ---------------------------------------------------------------------------

def obs_finger_pos(env) -> torch.Tensor:
    """Right hand finger joint positions [num_envs, 7]."""
    return env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]


def obs_finger_vel(env) -> torch.Tensor:
    """Right hand finger joint velocities [num_envs, 7]."""
    return env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]


def obs_contact_force(env) -> torch.Tensor:
    """Fingertip contact force magnitudes [num_envs, 3]."""
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    force_magnitudes = right_forces.norm(dim=-1)
    return force_magnitudes.clamp(0, 10.0)


def obs_contact_binary(env) -> torch.Tensor:
    """Binary contact indicators [num_envs, 3]."""
    force_mags = obs_contact_force(env)
    return (force_mags > 0.1).float()


def obs_phase_onehot(env) -> torch.Tensor:
    """Phase one-hot [num_envs, 8]. Always RELEASE_HOLD for this env."""
    onehot = torch.zeros(env.num_envs, NUM_PHASES, device=env.device)
    onehot[:, RELEASE_HOLD_PHASE_ID] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_release(env) -> torch.Tensor:
    """Combined release reward: block on surface, upright, fingers open, stable."""
    block_state = env.scene["block"].data.root_state_w
    block_pos = block_state[:, :3]
    block_quat = block_state[:, 3:7]
    block_vel = block_state[:, 7:10]
    block_z = block_pos[:, 2]
    block_xy = block_pos[:, :2]
    finger_pos = env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]
    action = env.action_manager.action

    env_origins = env.scene.env_origins
    target_xy = env_origins[:, :2] + torch.tensor(
        [CFG.block_initial_pos[0], CFG.block_initial_pos[1]], device=env.device)
    target_z = CFG.block_initial_z

    # Orientation error
    w = block_quat[:, 0]
    x = block_quat[:, 1]
    y = block_quat[:, 2]
    up_dot = (1.0 - 2.0 * (x**2 + y**2)).clamp(-1, 1)
    orient_error_rad = torch.acos(up_dot)

    xy_error = (block_xy - target_xy).norm(dim=-1)
    block_speed = block_vel.norm(dim=-1)

    # Finger openness (closure fraction)
    open_t = torch.tensor(FINGER_OPEN, device=env.device).unsqueeze(0)
    closed_t = torch.tensor(FINGER_CLOSED, device=env.device).unsqueeze(0)
    denom = (closed_t - open_t).abs() + 1e-6
    closure_frac = ((finger_pos - open_t).abs() / denom).clamp(0, 1).mean(dim=-1)
    openness = 1.0 - closure_frac

    reward = torch.zeros(env.num_envs, device=env.device)

    # PRIMARY: Success bonus
    block_near_target = (xy_error < 0.05) & ((block_z - target_z).abs() < 0.02)
    block_upright = orient_error_rad < (15.0 * math.pi / 180.0)
    fingers_open = closure_frac < 0.4
    block_stable = block_speed < 0.05
    success = block_near_target & block_upright & fingers_open & block_stable
    reward += success.float() * 10.0

    # SHAPED: encourage opening when block is near surface and upright
    near_surface = block_z < (target_z + 0.05)
    upright_frac = (1.0 - orient_error_rad / (math.pi / 2)).clamp(0, 1)
    reward += near_surface.float() * openness * upright_frac * 3.0

    # PENALTY: toppled or flew away
    reward -= (orient_error_rad > (math.pi / 4)).float() * 5.0

    # PENALTY: action smoothness
    reward -= 0.005 * (action ** 2).sum(dim=-1)

    return reward


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def terminate_block_fallen(env) -> torch.Tensor:
    """Terminate if block falls off the table."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    return block_z < CFG.block_initial_z - 0.15


def terminate_timeout(env) -> torch.Tensor:
    """Terminate at max episode length."""
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class ReleaseSceneCfg(InteractiveSceneCfg):
    """Scene: robot + table + block + contact sensor + ground + light."""

    robot: ArticulationCfg = _make_release_robot_cfg()

    # Table: kinematic cuboid (same as block_stack_env.py)
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CFG.table_pos,
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=CFG.table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.01,
            ),
        ),
    )

    # Block: dynamic cuboid on table at target position
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CFG.block_initial_pos,
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CFG.block_size, CFG.block_size, CFG.block_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=CFG.block_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=CFG.block_friction,
                dynamic_friction=CFG.block_friction * 0.8,
                restitution=0.01,
            ),
        ),
    )

    fingertip_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_hand_.*_link",
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )


# ---------------------------------------------------------------------------
# MDP configs
# ---------------------------------------------------------------------------

@configclass
class ReleaseActionsCfg:
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_.*_joint"],
        scale=CFG.action_scale,  # 1.5
        use_default_offset=True,
    )


@configclass
class ReleaseObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_pos = ObsTerm(func=obs_finger_pos)
        finger_vel = ObsTerm(func=obs_finger_vel)
        contact_force = ObsTerm(func=obs_contact_force)
        contact_binary = ObsTerm(func=obs_contact_binary)
        phase_onehot = ObsTerm(func=obs_phase_onehot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ReleaseRewardsCfg:
    release = RewTerm(func=reward_release, weight=1.0)


@configclass
class ReleaseTerminationsCfg:
    block_fallen = DoneTerm(func=terminate_block_fallen)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class ReleaseEventCfg:
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (0.0, 0.002),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------------------------------------------------------------------------
# Full environment config
# ---------------------------------------------------------------------------

@configclass
class ReleaseEnvCfg(ManagerBasedRLEnvCfg):
    """ManagerBasedRLEnv configuration for release specialist.
    Episode length: 100 steps (~1s at dt=0.005, decimation=2)."""

    scene: ReleaseSceneCfg = ReleaseSceneCfg(
        num_envs=CFG.num_envs,
        env_spacing=CFG.env_spacing,
    )
    observations: ReleaseObservationsCfg = ReleaseObservationsCfg()
    actions: ReleaseActionsCfg = ReleaseActionsCfg()
    rewards: ReleaseRewardsCfg = ReleaseRewardsCfg()
    terminations: ReleaseTerminationsCfg = ReleaseTerminationsCfg()
    events: ReleaseEventCfg = ReleaseEventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        # 100 steps: dt=0.005, decimation=2 -> 100 * 0.005 * 2 = 1.0s
        self.episode_length_s = 1.0
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

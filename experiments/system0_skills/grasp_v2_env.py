"""
System 0 Grasp V2: Pick up block from platform.

The arm AUTOMATICALLY lifts during the episode. The finger policy must
learn to grip the block tightly enough that it comes with the hand.

Episode phases (managed by the training loop, not here):
  Steps 0-100:   Arm at reach pose, fingers learn to close
  Steps 100-200: Arm lifts up, block either comes (success) or stays (fail)

Reward: based on block height — if it lifts with the hand, big reward.
The arm lifting is handled in the training loop via direct joint control.
This env just provides the scene, obs, and rewards.

Key design: arm uses MEDIUM stiffness so it can be driven by the training
loop to reach/lift positions, while fingers are policy-controlled.
"""

import os
import sys
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

from experiments.system0_skills.config import System0Config

CFG = System0Config()
RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_ARM_INDICES = CFG.right_arm_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices

project_root = os.environ.get("PROJECT_ROOT")

# Block on a small platform, within hand's reach at the arm reach pose
# VERIFIED positions with fingers OPEN vs CLOSED:
# Open fingertips: x=0.21-0.28, z=0.65-0.70
# Closed fingertips: x=0.06-0.12, z=0.56-0.66
# Palm (always): x=0.06-0.16, z=0.68-0.74
# Block must be where CLOSING fingers sweep through: ~(0.10, -0.09, 0.64)
BLOCK_SIZE = 0.035  # slightly smaller for easier grasp
PLATFORM_HEIGHT = 0.61
PLATFORM_SIZE = (0.08, 0.08, PLATFORM_HEIGHT)
# Between thumb(0.06,-0.07,0.66) and index(0.11,-0.07,0.62)
PLATFORM_POS = (0.085, -0.08, PLATFORM_HEIGHT / 2)
BLOCK_ON_PLATFORM_Z = PLATFORM_HEIGHT + BLOCK_SIZE / 2 + 0.001
BLOCK_POS = (0.085, -0.08, BLOCK_ON_PLATFORM_Z)  # exactly between thumb and index
BLOCK_INITIAL_Z = BLOCK_ON_PLATFORM_Z

# Arm joint targets
ARM_REACH_JOINTS = {
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.6,
    "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Lift pose — shoulder back, elbow less bent → hand moves up
ARM_LIFT_JOINTS = {
    "right_shoulder_pitch_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.6,
    "right_elbow_joint": 0.4,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def _make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=True,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=CFG.robot_pos, rot=CFG.robot_rot,
            joint_pos={
                "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.05, "left_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.15, "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.05, "right_knee_joint": 0.2,
                "right_ankle_pitch_joint": -0.15, "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                # Right arm starts at REACH pose
                **ARM_REACH_JOINTS,
                # Left hand open
                "left_hand_index_0_joint": 0.0, "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand OPEN — policy must close
                "right_hand_index_0_joint": 0.1,
                "right_hand_middle_0_joint": 0.1,
                "right_hand_thumb_0_joint": 0.0,
                "right_hand_index_1_joint": 0.1,
                "right_hand_middle_1_joint": 0.1,
                "right_hand_thumb_1_joint": 0.0,
                "right_hand_thumb_2_joint": -0.1,
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
            # Right arm — medium stiffness: follows position targets from training loop
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*_joint", "right_elbow_joint",
                                  "right_wrist_.*_joint"],
                effort_limit=300.0, velocity_limit=10.0,
                stiffness={".*": 200.0}, damping={".*": 50.0},
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*_joint", "left_elbow_joint",
                                  "left_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                  ".*_hand_thumb_.*_joint"],
                effort_limit=300, velocity_limit=100.0,
                stiffness={".*": 80.0}, damping={".*": 8.0},
                armature={".*": 0.1},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Observations (21D)
# ---------------------------------------------------------------------------

def obs_finger_pos(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]

def obs_finger_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]

def obs_contact_force(env) -> torch.Tensor:
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    rf = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    return rf.norm(dim=-1).clamp(0, 5.0)

def obs_target_force(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 3, device=env.device) * CFG.target_force

def obs_block_lifted(env) -> torch.Tensor:
    """How much the block has lifted above its starting height."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    lift = (block_z - BLOCK_INITIAL_Z).clamp(-0.1, 0.3)
    return lift.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Rewards — THE KEY: reward block actually being in the air
# ---------------------------------------------------------------------------

def reward_block_height(env) -> torch.Tensor:
    """The main reward: how high is the block above its start?
    Only positive when block lifts above initial position."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    lift = (block_z - BLOCK_INITIAL_Z).clamp(0, 0.2)  # 0 to 0.2m lift
    return lift * 50.0  # scale up: 0.1m lift = +5.0 reward

def reward_contact(env) -> torch.Tensor:
    """Reward for making fingertip contact with block."""
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    rf = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    contacts = (rf.norm(dim=-1) > 0.1).float().sum(dim=-1)
    return contacts  # 0 to 3

def reward_grip_force(env) -> torch.Tensor:
    """Reward for maintaining moderate grip force (not too much, not too little)."""
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    rf = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    total = rf.norm(dim=-1).sum(dim=-1)
    # Sweet spot: 1-5N total → reward 1.0, below or above → less
    return (1.0 - ((total - 3.0) / 3.0).abs()).clamp(0, 1.0)

def reward_block_dropped(env) -> torch.Tensor:
    """Penalty if block falls below starting height."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    dropped = (block_z < BLOCK_INITIAL_Z - 0.05).float()
    return -dropped

def reward_action_smoothness(env) -> torch.Tensor:
    action = env.action_manager.action
    return -(action ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def terminate_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class GraspV2SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_robot_cfg()

    platform: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Platform",
        spawn=sim_utils.CuboidCfg(
            size=PLATFORM_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PLATFORM_POS),
    )

    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BLOCK_POS, rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.0, restitution=0.01),
        ),
    )

    fingertip_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_hand_.*_link",
        history_length=1, track_air_time=False, debug_vis=False,
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0))


# ---------------------------------------------------------------------------
# MDP configs
# ---------------------------------------------------------------------------

@configclass
class GraspV2ActionsCfg:
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["right_hand_.*_joint"],
        scale=0.3, use_default_offset=True,
    )

@configclass
class GraspV2ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_pos = ObsTerm(func=obs_finger_pos)
        finger_vel = ObsTerm(func=obs_finger_vel)
        contact_force = ObsTerm(func=obs_contact_force)
        target_force = ObsTerm(func=obs_target_force)
        block_lifted = ObsTerm(func=obs_block_lifted)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class GraspV2RewardsCfg:
    block_height = RewTerm(func=reward_block_height, weight=1.0)
    contact = RewTerm(func=reward_contact, weight=0.5)
    grip_force = RewTerm(func=reward_grip_force, weight=0.3)
    block_dropped = RewTerm(func=reward_block_dropped, weight=5.0)
    action_smoothness = RewTerm(func=reward_action_smoothness, weight=0.005)

@configclass
class GraspV2TerminationsCfg:
    time_out = DoneTerm(func=terminate_timeout, time_out=True)

@configclass
class GraspV2EventCfg:
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class GraspV2EnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspV2SceneCfg = GraspV2SceneCfg(num_envs=512, env_spacing=3.0)
    observations: GraspV2ObsCfg = GraspV2ObsCfg()
    actions: GraspV2ActionsCfg = GraspV2ActionsCfg()
    rewards: GraspV2RewardsCfg = GraspV2RewardsCfg()
    terminations: GraspV2TerminationsCfg = GraspV2TerminationsCfg()
    events: GraspV2EventCfg = GraspV2EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        self.episode_length_s = 10.0  # 200 steps (100 grasp + 100 lift)
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

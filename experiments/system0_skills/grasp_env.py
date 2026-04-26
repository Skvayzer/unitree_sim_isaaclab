"""
System 0 Skill 1: Grasp Environment.

Minimal ManagerBasedRLEnv with:
- Robot (G1 + Dex3) with arm locked via high stiffness
- Small block spawned between right hand fingers
- Contact sensor on hand links
- Ground plane + dome light (NO room, table, cameras)

Actions: 7 right hand finger joint position deltas
Observations: finger_pos(7) + finger_vel(7) + contact_force(3) + target_force(3) + grasped(1) = 21
"""

import os
import sys
import torch

# Ensure project root is on path
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

# Indices from diagnostic output
RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices
BLOCK_INITIAL_Z = CFG.block_initial_z
TARGET_FORCE = CFG.target_force

# ---------------------------------------------------------------------------
# Robot config: copy the DEX3 base-fix config but increase arm/waist stiffness
# to lock those joints in place. Only hand joints remain soft.
# ---------------------------------------------------------------------------
project_root = os.environ.get("PROJECT_ROOT")


def _make_locked_robot_cfg() -> ArticulationCfg:
    """Create robot ArticulationCfg with arm/waist/leg joints locked via high stiffness."""
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
                # Arms — all zero
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Fingers — start open (zero)
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand starts OPEN — policy must learn to close
                "right_hand_index_0_joint": 0.1,     # [+0.079, +1.492] open=near lower
                "right_hand_middle_0_joint": 0.1,    # [+0.079, +1.492] open=near lower
                "right_hand_thumb_0_joint": 0.0,     # [-0.943, +0.943] open=center
                "right_hand_index_1_joint": 0.1,     # [+0.087, +1.658] open=near lower
                "right_hand_middle_1_joint": 0.1,    # [+0.087, +1.658] open=near lower
                "right_hand_thumb_1_joint": 0.0,     # [-0.964, +0.528] open=center
                "right_hand_thumb_2_joint": -0.1,    # [-1.658, -0.087] open=near upper
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # Lock legs
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
                armature=None,
            ),
            # Lock waist
            "waist": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                ],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
                armature=None,
            ),
            # Lock feet
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            # Lock arms with very high stiffness
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
                armature=None,
            ),
            # Hands — soft, these are the actuated joints
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hand_index_.*_joint",
                    ".*_hand_middle_.*_joint",
                    ".*_hand_thumb_.*_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={".*": 100.0},
                damping={".*": 10.0},
                armature={".*": 0.1},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Observation functions
# ---------------------------------------------------------------------------

def obs_finger_pos(env) -> torch.Tensor:
    """Right hand finger joint positions [num_envs, 7]."""
    return env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]


def obs_finger_vel(env) -> torch.Tensor:
    """Right hand finger joint velocities [num_envs, 7]."""
    return env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]


def obs_contact_force(env) -> torch.Tensor:
    """Fingertip contact force magnitudes [num_envs, 3].

    One scalar per fingertip (thumb, middle, index).
    Clamped to [0, 5] for numerical stability.
    """
    forces = env.scene["fingertip_contacts"].data.net_forces_w  # [num_envs, N_bodies, 3]
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]  # [num_envs, 3, 3]
    force_magnitudes = right_forces.norm(dim=-1)  # [num_envs, 3]
    return force_magnitudes.clamp(0, 5.0)


def obs_target_force(env) -> torch.Tensor:
    """Target contact force per finger [num_envs, 3]. Constant during training."""
    return torch.ones(env.num_envs, 3, device=env.device) * TARGET_FORCE


def obs_object_grasped(env) -> torch.Tensor:
    """Binary: is the object still between fingers? [num_envs, 1].

    Checks: block z > initial_z - 0.05 AND at least 2 fingertips have contact.
    """
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    contact_count = (right_forces.norm(dim=-1) > 0.1).sum(dim=-1)
    grasped = ((block_z > BLOCK_INITIAL_Z - 0.05) & (contact_count >= 2)).float()
    return grasped.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_grasp_success(env) -> torch.Tensor:
    """+1.0 if object is grasped."""
    return obs_object_grasped(env).squeeze(-1)


def reward_force_regulation(env) -> torch.Tensor:
    """Penalize deviation from target contact force."""
    forces = obs_contact_force(env)  # [num_envs, 3]
    target = torch.ones_like(forces) * TARGET_FORCE
    error = (forces - target).abs().sum(dim=-1)
    return -error


def reward_object_dropped(env) -> torch.Tensor:
    """Large penalty if block falls below threshold."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    dropped = (block_z < BLOCK_INITIAL_Z - 0.1).float()
    return -dropped


def reward_action_smoothness(env) -> torch.Tensor:
    """Penalize large actions."""
    action = env.action_manager.action  # [num_envs, 7]
    return -(action ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def terminate_object_dropped(env) -> torch.Tensor:
    """Terminate if block falls too far."""
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    return block_z < BLOCK_INITIAL_Z - 0.1


def terminate_timeout(env) -> torch.Tensor:
    """Terminate at max episode length (handled by Isaac Lab via episode_length_s)."""
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Minimal scene: robot + block + contact sensor + ground + light."""

    robot: ArticulationCfg = _make_locked_robot_cfg()

    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CFG.block_pos,
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.0,
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
class ActionsCfg:
    """Only control right hand finger joints (7 DOF)."""
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_.*_joint"],
        scale=CFG.action_scale,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_pos = ObsTerm(func=obs_finger_pos)
        finger_vel = ObsTerm(func=obs_finger_vel)
        contact_force = ObsTerm(func=obs_contact_force)
        target_force = ObsTerm(func=obs_target_force)
        object_grasped = ObsTerm(func=obs_object_grasped)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    grasp_success = RewTerm(func=reward_grasp_success, weight=CFG.grasp_reward_weight)
    force_regulation = RewTerm(func=reward_force_regulation, weight=CFG.force_penalty_weight)
    object_dropped = RewTerm(func=reward_object_dropped, weight=CFG.drop_penalty_weight)
    action_smoothness = RewTerm(func=reward_action_smoothness, weight=CFG.smooth_penalty_weight)


@configclass
class TerminationsCfg:
    object_dropped = DoneTerm(func=terminate_object_dropped)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class EventCfg:
    """Reset block position with small random offset on episode reset."""
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.003, 0.003),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------------------------------------------------------------------------
# Full environment config
# ---------------------------------------------------------------------------

@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """ManagerBasedRLEnv configuration for System 0 Skill 1: Grasp."""

    scene: GraspSceneCfg = GraspSceneCfg(
        num_envs=CFG.num_envs,
        env_spacing=CFG.env_spacing,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        self.episode_length_s = CFG.episode_length_s
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

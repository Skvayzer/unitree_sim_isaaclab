"""
System 0 Multi-Task Environment for MoE Router Training.

Each environment is randomly assigned one of 3 tasks (grasp/hold/release)
on reset. The observation is padded to a unified 28D space. A task_id
one-hot (3D) is appended, making the full obs 31D — but we exclude the
task_id from policy input and only use it for the router's benefit via
the physical state differences.

Actually, the router should learn from physical context alone (contact
pattern, arm velocity, target forces), NOT from an explicit task label.
This way it generalizes to deployment where task labels don't exist.

Unified obs layout (28D):
    [0:7]   finger_pos
    [7:14]  finger_vel
    [14:17] contact_force
    [17:20] target_force (1.0 for grasp/hold, 0.0 for release)
    [20]    object_grasped
    [21]    height_above_ground (release only, 0 otherwise)
    [22:28] padding zeros / arm_vel for hold

This environment uses the GRASP scene (simplest, arm locked) but
varies the task context per reset:
- Grasp: block between fingers, target_force=1.0, fingers open
- Hold:  same as grasp but with simulated arm velocity noise in obs
- Release: block between fingers, target_force=0.0, fingers closed
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
BLOCK_INITIAL_Z = CFG.block_initial_z
GROUND_Z = 0.02

project_root = os.environ.get("PROJECT_ROOT")

# Task IDs
TASK_GRASP = 0
TASK_HOLD = 1
TASK_RELEASE = 2

# Module-level state — set by the training loop after env creation
_task_ids = None  # [num_envs] tensor of task IDs
_arm_vel_noise = None  # [num_envs, 7] simulated arm velocity for hold task


def set_task_state(task_ids: torch.Tensor, arm_vel_noise: torch.Tensor):
    """Called by training loop to set per-env task context."""
    global _task_ids, _arm_vel_noise
    _task_ids = task_ids
    _arm_vel_noise = arm_vel_noise


def _make_multitask_robot_cfg() -> ArticulationCfg:
    """Same as grasp env robot — arm locked."""
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
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
                "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "left_hand_index_0_joint": 0.0, "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand starts ~30% closed (grasp default)
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
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                  ".*_hand_thumb_.*_joint"],
                effort_limit=300, velocity_limit=100.0,
                stiffness={".*": 100.0}, damping={".*": 10.0},
                armature={".*": 0.1},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Unified observation function (28D)
# ---------------------------------------------------------------------------

def obs_unified(env) -> torch.Tensor:
    """Build unified 28D observation vector.

    Layout:
        [0:7]   finger_pos
        [7:14]  finger_vel
        [14:17] contact_force (clamped)
        [17:20] target_force (task-dependent: 1.0 for grasp/hold, 0.0 for release)
        [20]    object_grasped
        [21]    height_above_ground
        [22:28] arm_vel (real for hold, zero for grasp/release)
        Total: 28
    """
    global _task_ids, _arm_vel_noise
    device = env.device
    N = env.num_envs

    obs = torch.zeros(N, 28, device=device)

    # Finger pos/vel
    obs[:, 0:7] = env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]
    obs[:, 7:14] = env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]

    # Contact forces
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    obs[:, 14:17] = right_forces.norm(dim=-1).clamp(0, 5.0)

    # Target force: 1.0 for grasp/hold, 0.0 for release
    if _task_ids is not None:
        release_mask = (_task_ids == TASK_RELEASE)
        target = torch.ones(N, 3, device=device)
        target[release_mask] = 0.0
        obs[:, 17:20] = target
    else:
        obs[:, 17:20] = 1.0  # default: grasp

    # Object grasped — relaxed: just check block height (contact count unreliable across sim versions)
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    contact_count = (right_forces.norm(dim=-1) > 0.1).sum(dim=-1)
    grasped = (block_z > BLOCK_INITIAL_Z - 0.08).float()  # block hasn't fallen much
    obs[:, 20] = grasped

    # Height above ground
    height = (block_z - GROUND_Z).clamp(0, 1.0)
    obs[:, 21] = height

    # Arm velocity (for hold task, simulated noise; otherwise zero)
    if _arm_vel_noise is not None:
        obs[:, 22:28] = _arm_vel_noise[:, :6]  # only 6 dims fit in remaining space

    return obs


# ---------------------------------------------------------------------------
# Reward: task-adaptive
# ---------------------------------------------------------------------------

def reward_multitask(env) -> torch.Tensor:
    """Reward that adapts based on task_id per environment.

    Grasp/Hold: +1 if grasped, penalty for drops
    Release: +5 if released upright, penalty for toppling
    """
    global _task_ids
    device = env.device
    N = env.num_envs
    reward = torch.zeros(N, device=device)

    block_z = env.scene["block"].data.root_pos_w[:, 2]
    # Relaxed grasped: block hasn't fallen much (contact count unreliable across sim versions)
    grasped = (block_z > BLOCK_INITIAL_Z - 0.08).float()
    dropped = (block_z < BLOCK_INITIAL_Z - 0.15).float()

    if _task_ids is None:
        # Fallback: pure grasp reward
        reward = grasped - 5.0 * dropped
        return reward

    # Grasp & Hold tasks
    grasp_hold_mask = (_task_ids == TASK_GRASP) | (_task_ids == TASK_HOLD)
    reward[grasp_hold_mask] = grasped[grasp_hold_mask] - 5.0 * dropped[grasp_hold_mask]

    # Release task
    release_mask = (_task_ids == TASK_RELEASE)
    if release_mask.any():
        # For release: reward block descending and low contact force
        forces = env.scene["fingertip_contacts"].data.net_forces_w
        right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
        force_mags = right_forces.norm(dim=-1)
        total_force = force_mags.sum(dim=-1)

        descended = (BLOCK_INITIAL_Z - block_z).clamp(0, 1.0) * 2.0  # up to +2.0
        low_force = torch.exp(-total_force)  # 0 to 1.0
        holding_penalty = -0.1 * (total_force.clamp(0, 10.0) / 10.0)

        reward[release_mask] = (descended + low_force + holding_penalty)[release_mask]

    # Action smoothness for all tasks
    action = env.action_manager.action
    reward -= 0.01 * (action ** 2).sum(dim=-1)

    return reward


def reward_placeholder(env) -> torch.Tensor:
    """Placeholder — actual reward computed in reward_multitask."""
    return reward_multitask(env)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def terminate_multitask(env) -> torch.Tensor:
    global _task_ids
    block_z = env.scene["block"].data.root_pos_w[:, 2]

    # Grasp/Hold: terminate on drop
    dropped = block_z < BLOCK_INITIAL_Z - 0.1

    if _task_ids is not None:
        # Release: terminate when block on ground and stable
        release_mask = (_task_ids == TASK_RELEASE)
        if release_mask.any():
            block_vel = env.scene["block"].data.root_lin_vel_w
            speed = block_vel.norm(dim=-1)
            on_ground = block_z < GROUND_Z + 0.05
            stable = speed < 0.05
            released = on_ground & stable
            # Only terminate release tasks on release, not on drop
            dropped = dropped & ~release_mask
            return dropped | (released & release_mask)

    return dropped


def terminate_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Scene & env config
# ---------------------------------------------------------------------------

@configclass
class MultiTaskSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_multitask_robot_cfg()
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CFG.block_pos, rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
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


@configclass
class MultiTaskActionsCfg:
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["right_hand_.*_joint"],
        scale=CFG.action_scale, use_default_offset=True,
    )


@configclass
class MultiTaskObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        unified = ObsTerm(func=obs_unified)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class MultiTaskRewardsCfg:
    multitask = RewTerm(func=reward_placeholder, weight=1.0)


@configclass
class MultiTaskTerminationsCfg:
    task_done = DoneTerm(func=terminate_multitask)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class MultiTaskEventCfg:
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.003, 0.003)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={
            "position_range": (-0.0, 0.0), "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class MultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: MultiTaskSceneCfg = MultiTaskSceneCfg(
        num_envs=CFG.num_envs, env_spacing=CFG.env_spacing)
    observations: MultiTaskObsCfg = MultiTaskObsCfg()
    actions: MultiTaskActionsCfg = MultiTaskActionsCfg()
    rewards: MultiTaskRewardsCfg = MultiTaskRewardsCfg()
    terminations: MultiTaskTerminationsCfg = MultiTaskTerminationsCfg()
    events: MultiTaskEventCfg = MultiTaskEventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        self.episode_length_s = 10.0
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

"""
System 0 Skill 2: Hold While Moving.

Same as Skill 1 (grasp), but after initial grasp the arm joints follow
scripted sinusoidal perturbations. The policy must maintain grip under
disturbance.

Key differences from grasp_env.py:
- Arm joints are NOT locked — they follow sinusoidal trajectories
- Observation includes arm joint velocities (7 extra dims → 28 total)
- Higher drop penalty (-10.0 vs -5.0) since arm is moving
- Episode length 300 steps (15s at dt=0.005 * decimation=2)
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

from experiments.system0_skills.config import System0Config

CFG = System0Config()

RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_ARM_INDICES = CFG.right_arm_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices
BLOCK_INITIAL_Z = CFG.block_initial_z
TARGET_FORCE = CFG.target_force

# Sinusoidal perturbation parameters
PERTURB_AMPLITUDE = 0.1   # ±0.1 rad
PERTURB_FREQ_HZ = 0.5     # 0.5 Hz
PERTURB_DT = CFG.sim_dt * CFG.decimation  # effective dt per policy step

project_root = os.environ.get("PROJECT_ROOT")


def _make_hold_robot_cfg() -> ArticulationCfg:
    """Robot with arm joints soft enough to follow position targets (for perturbation),
    but everything else locked. Only hand joints are controlled by the policy."""
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
                # Arms — all zero (will be perturbed during episode)
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
                # Right hand starts ~30% closed (same as Skill 1)
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
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
            ),
            # Lock feet
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            # Right arm — soft enough to follow position targets (for perturbation)
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_.*_joint",
                    "right_elbow_joint",
                    "right_wrist_.*_joint",
                ],
                effort_limit=300.0,
                velocity_limit=10.0,
                stiffness={".*": 200.0},
                damping={".*": 50.0},
            ),
            # Lock left arm
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_shoulder_.*_joint",
                    "left_elbow_joint",
                    "left_wrist_.*_joint",
                ],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            # Hands — soft, controlled by policy
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
# Arm perturbation — applied in the training loop, not via manager
# ---------------------------------------------------------------------------

class ArmPerturbation:
    """Generates sinusoidal joint position targets for right arm joints.

    Each env gets random phase offsets so perturbations are decorrelated.
    Only shoulder_pitch and elbow are perturbed (the joints that create
    the most wrist displacement).
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        # Random phase per env per perturbed joint (2 joints: shoulder_pitch, elbow)
        self.phases = torch.rand(num_envs, 2, device=device) * 2 * math.pi
        # Different frequencies for each joint for richer perturbation
        self.freqs = torch.tensor([PERTURB_FREQ_HZ, PERTURB_FREQ_HZ * 0.7], device=device)
        self.amplitudes = torch.tensor([PERTURB_AMPLITUDE, PERTURB_AMPLITUDE * 0.8], device=device)
        self.step_count = torch.zeros(num_envs, device=device)

    def get_targets(self) -> torch.Tensor:
        """Returns position offsets for right arm joints [num_envs, 7].

        Only shoulder_pitch (index 0 in right arm) and elbow (index 3) are perturbed.
        Other arm joints stay at zero offset.
        """
        t = self.step_count * PERTURB_DT  # [num_envs]
        offsets = torch.zeros(self.num_envs, 7, device=self.device)

        for j in range(2):
            # j=0: shoulder_pitch, j=1: elbow
            target_idx = 0 if j == 0 else 3  # indices within right arm 7-vector
            phase = 2 * math.pi * self.freqs[j] * t + self.phases[:, j]
            offsets[:, target_idx] = self.amplitudes[j] * torch.sin(phase)

        self.step_count += 1
        return offsets

    def reset(self, env_ids: torch.Tensor):
        """Reset phase and step counter for specific envs."""
        self.phases[env_ids] = torch.rand(len(env_ids), 2, device=self.device) * 2 * math.pi
        self.step_count[env_ids] = 0


# ---------------------------------------------------------------------------
# Observation functions
# ---------------------------------------------------------------------------

def obs_finger_pos(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]


def obs_finger_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]


def obs_contact_force(env) -> torch.Tensor:
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    force_magnitudes = right_forces.norm(dim=-1)
    return force_magnitudes.clamp(0, 5.0)


def obs_target_force(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 3, device=env.device) * TARGET_FORCE


def obs_object_grasped(env) -> torch.Tensor:
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    contact_count = (right_forces.norm(dim=-1) > 0.1).sum(dim=-1)
    grasped = ((block_z > BLOCK_INITIAL_Z - 0.05) & (contact_count >= 2)).float()
    return grasped.unsqueeze(-1)


def obs_arm_vel(env) -> torch.Tensor:
    """Right arm joint velocities — disturbance context [num_envs, 7]."""
    return env.scene["robot"].data.joint_vel[:, RIGHT_ARM_INDICES]


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_hold_success(env) -> torch.Tensor:
    return obs_object_grasped(env).squeeze(-1)


def reward_force_regulation(env) -> torch.Tensor:
    forces = obs_contact_force(env)
    target = torch.ones_like(forces) * TARGET_FORCE
    error = (forces - target).abs().sum(dim=-1)
    return -error


def reward_object_dropped(env) -> torch.Tensor:
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    dropped = (block_z < BLOCK_INITIAL_Z - 0.1).float()
    return -dropped


def reward_action_smoothness(env) -> torch.Tensor:
    action = env.action_manager.action
    return -(action ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def terminate_object_dropped(env) -> torch.Tensor:
    block_z = env.scene["block"].data.root_pos_w[:, 2]
    return block_z < BLOCK_INITIAL_Z - 0.1


def terminate_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class HoldSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_hold_robot_cfg()

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
class HoldActionsCfg:
    """Only finger joints are policy-controlled. Arm is driven externally."""
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_.*_joint"],
        scale=CFG.action_scale,
        use_default_offset=True,
    )


@configclass
class HoldObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_pos = ObsTerm(func=obs_finger_pos)
        finger_vel = ObsTerm(func=obs_finger_vel)
        contact_force = ObsTerm(func=obs_contact_force)
        target_force = ObsTerm(func=obs_target_force)
        object_grasped = ObsTerm(func=obs_object_grasped)
        arm_vel = ObsTerm(func=obs_arm_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HoldRewardsCfg:
    hold_success = RewTerm(func=reward_hold_success, weight=CFG.grasp_reward_weight)
    force_regulation = RewTerm(func=reward_force_regulation, weight=CFG.force_penalty_weight)
    object_dropped = RewTerm(func=reward_object_dropped, weight=10.0)  # higher than Skill 1
    action_smoothness = RewTerm(func=reward_action_smoothness, weight=CFG.smooth_penalty_weight)


@configclass
class HoldTerminationsCfg:
    object_dropped = DoneTerm(func=terminate_object_dropped)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class HoldEventCfg:
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


@configclass
class HoldEnvCfg(ManagerBasedRLEnvCfg):
    scene: HoldSceneCfg = HoldSceneCfg(
        num_envs=CFG.num_envs,
        env_spacing=CFG.env_spacing,
    )
    observations: HoldObservationsCfg = HoldObservationsCfg()
    actions: HoldActionsCfg = HoldActionsCfg()
    rewards: HoldRewardsCfg = HoldRewardsCfg()
    terminations: HoldTerminationsCfg = HoldTerminationsCfg()
    events: HoldEventCfg = HoldEventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        self.episode_length_s = 15.0  # longer episodes: 300 steps
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

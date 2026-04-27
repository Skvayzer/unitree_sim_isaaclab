"""
System 0 Block Stacking Environment.

ManagerBasedRLEnv with:
- Robot (G1 + Dex3) with arm search DOF + finger RL control
- Table (kinematic cuboid)
- Block (dynamic cuboid on table)
- Contact sensor on hand links (filtered to block only)
- Ground plane + dome light

Actions: 12 DOF — 5 arm/wrist (shoulder_pitch, shoulder_roll, elbow, wrist_roll, wrist_pitch)
                 + 7 right-hand finger joints
Arm joints locked/controllable split: shoulder_yaw + wrist_yaw remain locked (kinematic
redundancy); the 5 controllable joints allow table-sweep + wrist reorientation on contact.
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

from experiments.system0_skills.block_stack_config import BlockStackConfig

CFG = BlockStackConfig()
project_root = os.environ.get("PROJECT_ROOT")

RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_ARM_INDICES = CFG.right_arm_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices


# ---------------------------------------------------------------------------
# Robot config
# ---------------------------------------------------------------------------

def _make_robot_cfg() -> ArticulationCfg:
    """Robot with arm joints controlled via medium stiffness (for scripted trajectory),
    legs/waist locked, hand joints soft for RL control."""
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
                # Left arm — zero
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                # Right arm — start at hover position
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.5,
                "right_shoulder_yaw_joint": 0.1,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Left hand — zero
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand — start OPEN
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
            # Lock legs
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                                  ".*_hip_pitch_joint", ".*_knee_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            # Lock waist
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
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
            # Left arm locked
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            # Right arm: yaw joints stay locked (kinematic redundancy — no workspace gain)
            "right_arm_locked": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_yaw_joint", "right_wrist_yaw_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,
                stiffness={".*": 5000.0},
                damping={".*": 500.0},
            ),
            # Right arm: controllable joints for blindfold-search behavior
            # Stiffness 800 → holds pose against gravity but moves under policy command
            "right_arm_controllable": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                ],
                effort_limit=1000.0,
                velocity_limit=100.0,
                stiffness={".*": 800.0},
                damping={".*": 80.0},
            ),
            # Hands — soft, RL-controlled (right) and locked (left)
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_hand_.*_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness={".*": 10000.0},
                damping={".*": 10000.0},
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_hand_.*_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness={".*": CFG.hand_stiffness},
                damping={".*": CFG.hand_damping},
                armature={".*": 0.1},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Observation functions (used by ManagerBasedRLEnv)
# These return dummy obs — real obs are built in the training loop
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
    """Phase one-hot [num_envs, 8]. Filled by training loop via _phase_onehot buffer."""
    if hasattr(env, '_phase_onehot') and env._phase_onehot is not None:
        return env._phase_onehot
    return torch.zeros(env.num_envs, 8, device=env.device)


# ---------------------------------------------------------------------------
# Reward function (dummy — real reward computed in training loop)
# ---------------------------------------------------------------------------

def reward_dummy(env) -> torch.Tensor:
    """Placeholder reward. Real reward is computed in training loop."""
    return torch.zeros(env.num_envs, device=env.device)


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
class BlockStackSceneCfg(InteractiveSceneCfg):
    """Scene: robot + table + block + contact sensor + ground + light."""

    robot: ArticulationCfg = _make_robot_cfg()

    # Table: kinematic cuboid
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

    # Block: dynamic cuboid
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CFG.block_initial_pos,
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CFG.block_size, CFG.block_size, CFG.block_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CFG.block_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=CFG.block_friction,
                dynamic_friction=CFG.block_friction * 0.8,
                restitution=0.01,
            ),
        ),
    )

    fingertip_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_hand_.*_link",
        filter_prim_paths_expr=["/World/envs/env_.*/Block"],
        history_length=2,
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
    """12 DOF: 5 arm/wrist joints first, then 7 finger joints.

    Ordering (action[:, :5] = arm, action[:, 5:] = fingers):
      0: right_shoulder_pitch_joint  — forward/back sweep
      1: right_shoulder_roll_joint   — lateral sweep
      2: right_elbow_joint           — height + reach
      3: right_wrist_roll_joint      — pronation (thumb into opposition)
      4: right_wrist_pitch_joint     — palm tilt
      5-11: right hand finger joints (thumb_0/1/2, middle_0/1, index_0/1)
    """
    arm_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        scale=1.5,
        use_default_offset=True,
    )
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
        contact_binary = ObsTerm(func=obs_contact_binary)
        phase_onehot = ObsTerm(func=obs_phase_onehot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    dummy = RewTerm(func=reward_dummy, weight=1.0)


@configclass
class TerminationsCfg:
    block_fallen = DoneTerm(func=terminate_block_fallen)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class EventCfg:
    """Reset block position with small random offset."""
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.008, 0.008),
                "y": (-0.008, 0.008),
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
class BlockStackEnvCfg(ManagerBasedRLEnvCfg):
    """ManagerBasedRLEnv configuration for block stacking."""

    scene: BlockStackSceneCfg = BlockStackSceneCfg(
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
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 512 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024  # fix patch buffer overflow at 8K+ envs
        self.sim.physx.gpu_heap_capacity = 128 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.friction_correlation_distance = 0.00625

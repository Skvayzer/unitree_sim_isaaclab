"""
System 0 Multi-Block Tower Stacking Environment.

ManagerBasedRLEnv with:
- Robot (G1 + Dex3) with arm controlled by scripted trajectory
- Table (kinematic cuboid)
- 3 Blocks (dynamic cuboids: red, yellow, green)
- Contact sensor on hand links
- Ground plane + dome light

Actions: 7 right hand finger joint position deltas
Observations: finger_pos(7) + finger_vel(7) + contact_force(3) + binary_contact(3) + phase_onehot(8) + block_idx_onehot(3) = 31
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

from experiments.system0_skills.multi_block_config import MultiBlockConfig

CFG = MultiBlockConfig()
project_root = os.environ.get("PROJECT_ROOT")

RIGHT_HAND_INDICES = CFG.right_hand_indices
RIGHT_ARM_INDICES = CFG.right_arm_indices
RIGHT_FINGERTIP_CONTACT_INDICES = CFG.right_fingertip_contact_indices  # legacy (3 tips)
RIGHT_HAND_MODULE_INDICES = CFG.right_hand_module_contact_indices       # all 9 modules


# ---------------------------------------------------------------------------
# Robot config (same as single-block)
# ---------------------------------------------------------------------------

def _make_robot_cfg() -> ArticulationCfg:
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
                "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.5,
                "right_shoulder_yaw_joint": 0.1, "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "left_hand_index_0_joint": 0.0, "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                "right_hand_index_0_joint": 0.1, "right_hand_middle_0_joint": 0.1,
                "right_hand_thumb_0_joint": 0.0, "right_hand_index_1_joint": 0.1,
                "right_hand_middle_1_joint": 0.1, "right_hand_thumb_1_joint": 0.0,
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
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=100.0,
                stiffness={".*": CFG.arm_stiffness}, damping={".*": CFG.arm_damping},
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
# Observation functions
# ---------------------------------------------------------------------------

def obs_finger_pos(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, RIGHT_HAND_INDICES]

def obs_finger_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel[:, RIGHT_HAND_INDICES]

def obs_contact_force(env) -> torch.Tensor:
    """Legacy 3-tip force magnitudes. Kept for backward compat; not used in obs."""
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, RIGHT_FINGERTIP_CONTACT_INDICES, :]
    return right_forces.norm(dim=-1).clamp(0, 10.0)

def obs_contact_binary(env) -> torch.Tensor:
    """Legacy 3-tip binary contact. Kept for backward compat; not used in obs."""
    return (obs_contact_force(env) > 0.1).float()


# ---------------------------------------------------------------------------
# Hemisphere normals for pressure cell simulation (12 directions, shape [12, 3]).
# 3 rings of 4 azimuths each, covering elevations 30°/60°/80° from the z-axis.
# Each cell fires proportional to the component of the contact force along its
# normal, matching how real resistive pressure sensors respond to load direction.
# ---------------------------------------------------------------------------
_PRESS_NORMALS_CPU = torch.tensor([
    # elevation 30° — azimuths 0°, 90°, 180°, 270°
    [ 0.5000,  0.0000,  0.8660],
    [ 0.0000,  0.5000,  0.8660],
    [-0.5000,  0.0000,  0.8660],
    [ 0.0000, -0.5000,  0.8660],
    # elevation 60° — azimuths 45°, 135°, 225°, 315° (staggered ring)
    [ 0.6124,  0.6124,  0.5000],
    [-0.6124,  0.6124,  0.5000],
    [-0.6124, -0.6124,  0.5000],
    [ 0.6124, -0.6124,  0.5000],
    # elevation 80° — azimuths 0°, 90°, 180°, 270°
    [ 0.9848,  0.0000,  0.1736],
    [ 0.0000,  0.9848,  0.1736],
    [-0.9848,  0.0000,  0.1736],
    [ 0.0000, -0.9848,  0.1736],
], dtype=torch.float32)  # (12, 3)

_press_normals_cache: dict = {}
_press_dds_last_ms: int = 0


def _get_press_normals(device) -> torch.Tensor:
    key = str(device)
    if key not in _press_normals_cache:
        _press_normals_cache[key] = _PRESS_NORMALS_CPU.to(device)
    return _press_normals_cache[key]


def obs_press_sensor_modules(env) -> torch.Tensor:
    """Simulate all 9 Dex3-1 pressure sensor modules (108D total).

    For each of the 9 right-hand rigid bodies the net contact force is projected
    onto 12 fixed hemisphere normals, producing non-negative pressure readings
    that match the layout of hardware PressSensorState_.pressure[12].

    Returns:
        Tensor (B, 108) — 9 modules × 12 readings, values in [0, 1].
    """
    global _press_dds_last_ms

    forces = env.scene["fingertip_contacts"].data.net_forces_w  # (B, 18, 3)
    module_forces = forces[:, RIGHT_HAND_MODULE_INDICES, :]      # (B,  9, 3)
    normals = _get_press_normals(forces.device)                  # (12, 3)

    # Project: (B, 9, 3) × (3, 12) → (B, 9, 12), then normalise to [0, 1]
    pressures = torch.einsum("bnd,kd->bnk", module_forces, normals)
    pressures = pressures.clamp(0.0, CFG.press_force_scale) / CFG.press_force_scale

    # Publish first-env pressure to DDS at ≤50 Hz
    try:
        import time
        now_ms = int(time.time() * 1000)
        if now_ms - _press_dds_last_ms >= 20:
            dex3_dds = _get_dex3_dds_instance()
            if dex3_dds and hasattr(dex3_dds, "write_hand_pressure"):
                press_np = pressures[0].detach().cpu().numpy()  # (9, 12)
                dex3_dds.write_hand_pressure("right", press_np)
                _press_dds_last_ms = now_ms
    except Exception:
        pass

    return pressures.reshape(pressures.shape[0], -1)  # (B, 108)

def obs_phase_onehot(env) -> torch.Tensor:
    if hasattr(env, '_phase_onehot') and env._phase_onehot is not None:
        return env._phase_onehot
    return torch.zeros(env.num_envs, 8, device=env.device)

def obs_block_idx_onehot(env) -> torch.Tensor:
    """Block index one-hot [num_envs, 3]. Set by training loop."""
    if hasattr(env, '_block_idx_onehot') and env._block_idx_onehot is not None:
        return env._block_idx_onehot
    return torch.zeros(env.num_envs, 3, device=env.device)


# ---------------------------------------------------------------------------
# Reward / Termination functions
# ---------------------------------------------------------------------------

def reward_dummy(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

_DISABLE_BLOCK_FALLEN = False  # Set True for eval (state machine manages lifecycle)

def terminate_any_block_fallen(env) -> torch.Tensor:
    """Terminate if any block falls far below table."""
    if _DISABLE_BLOCK_FALLEN:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(CFG.num_blocks):
        block_z = env.scene[f"block_{i}"].data.root_pos_w[:, 2]
        fallen = fallen | (block_z < CFG.block_initial_z - 0.15)
    return fallen

def terminate_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Helper: create block config
# ---------------------------------------------------------------------------

def _make_block_cfg(idx: int) -> RigidObjectCfg:
    pos = CFG.block_positions[idx]
    color = CFG.block_colors[idx]
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Block_{idx}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=CFG.block_friction,
                dynamic_friction=CFG.block_friction * 0.8,
                restitution=0.01,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class MultiBlockSceneCfg(InteractiveSceneCfg):
    """Scene: robot + table + 3 blocks + contact sensor + ground + light."""

    robot: ArticulationCfg = _make_robot_cfg()

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CFG.table_pos, rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=CFG.table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.01,
            ),
        ),
    )

    # 3 blocks
    block_0: RigidObjectCfg = _make_block_cfg(0)
    block_1: RigidObjectCfg = _make_block_cfg(1)
    block_2: RigidObjectCfg = _make_block_cfg(2)

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
        press_sensor_modules = ObsTerm(func=obs_press_sensor_modules)
        phase_onehot = ObsTerm(func=obs_phase_onehot)
        block_idx_onehot = ObsTerm(func=obs_block_idx_onehot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    dummy = RewTerm(func=reward_dummy, weight=1.0)


@configclass
class TerminationsCfg:
    block_fallen = DoneTerm(func=terminate_any_block_fallen)
    time_out = DoneTerm(func=terminate_timeout, time_out=True)


@configclass
class EventCfg:
    """Reset all 3 block positions with small random offset."""
    reset_block_0 = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.008, 0.008), "y": (-0.008, 0.008), "z": (0.0, 0.002)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block_0"),
        },
    )
    reset_block_1 = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.008, 0.008), "y": (-0.008, 0.008), "z": (0.0, 0.002)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block_1"),
        },
    )
    reset_block_2 = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.008, 0.008), "y": (-0.008, 0.008), "z": (0.0, 0.002)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block_2"),
        },
    )
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset",
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
class MultiBlockEnvCfg(ManagerBasedRLEnvCfg):
    scene: MultiBlockSceneCfg = MultiBlockSceneCfg(
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
        self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024
        self.sim.physx.gpu_heap_capacity = 128 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.friction_correlation_distance = 0.00625

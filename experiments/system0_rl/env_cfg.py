"""
DDS-free env config for System 0 standalone RL training.

Same scene/actions/events as the original task, but observations
are a simple joint position readout (no DDS, no cameras).
The System 0 training loop reads all data directly from env.scene.
"""

import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_scene.base_scene_stack_rgyblock import TableRedGreenYellowBlockSceneCfg

# Import only the non-DDS mdp functions we need
from tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3 import mdp


# ── Simple observation: just joint positions from scene API (no DDS) ──

def get_joint_pos_direct(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read joint positions directly from scene — no DDS needed."""
    return env.scene["robot"].data.joint_pos


@configclass
class SceneCfg(TableRedGreenYellowBlockSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
    )
    fingertip_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_hand_.*_link",
        history_length=2,
        track_air_time=False,
        debug_vis=False,
    )
    # No cameras — System 0 doesn't need them and they crash with multi-env
    # front_camera_left = CameraPresets.g1_front_camera_left()
    # front_camera_right = CameraPresets.g1_front_camera_right()
    # left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    # right_wrist_camera = CameraPresets.right_dex3_wrist_camera()


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=get_joint_pos_direct)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _block_oob_check(env) -> torch.Tensor:
    """Terminate if any block fell off table or drifted beyond arm reach."""
    N = env.num_envs
    done = torch.zeros(N, dtype=torch.bool, device=env.device)
    robot_pos = env.scene["robot"].data.root_pos_w  # (N, 3)
    for block_name in ["red_block", "yellow_block", "green_block"]:
        try:
            bpos = env.scene[block_name].data.root_pos_w  # (N, 3)
            fell = bpos[:, 2] < 0.75
            hdist = ((bpos[:, :2] - robot_pos[:, :2]) ** 2).sum(dim=1).sqrt()
            too_far = hdist > 0.5
            done = done | fell | too_far
        except (KeyError, AttributeError):
            pass
    return done


def _zero_reward(env) -> torch.Tensor:
    """Dummy reward — actual reward computed in training loop."""
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class TerminationsCfg:
    block_oob = DoneTerm(func=_block_oob_check)


@configclass
class RewardsCfg:
    dummy = RewTerm(func=_zero_reward, weight=1.0)


@configclass
class EventCfg:
    reset_red_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("red_block"),
        },
    )
    reset_yellow_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("yellow_block"),
        },
    )
    reset_green_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("green_block"),
        },
    )


@configclass
class System0TrainEnvCfg(ManagerBasedRLEnvCfg):
    """DDS-free env config for System 0 RL training."""

    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        # Viewer camera facing robot
        self.viewer.eye = (-4.2, -5.0, 1.5)
        self.viewer.lookat = (-4.2, -3.7, 0.9)
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.contact_offset = 0.01
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 4
        self.scene.fingertip_contacts.update_period = self.sim.dt

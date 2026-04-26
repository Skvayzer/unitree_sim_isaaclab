#!/usr/bin/env python3
"""
Step 0: Print all joint names, indices, limits, and contact sensor info.
Run this FIRST and use the output to set constants in config.py.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/test_env.py --num_envs 1
"""

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PROJECT_ROOT"] = project_root
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from tasks.common_config import G1RobotPresets


def _get_joint_pos(env):
    return env.scene["robot"].data.joint_pos


def _always_false(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class DiagSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(0.0, 0.0, 0.76),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )
    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.15, 0.84),
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.5
            ),
        ),
    )
    fingertip_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_hand_.*_link",
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )


@configclass
class DiagObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=_get_joint_pos)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class DiagActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
    )


@configclass
class DiagTermCfg:
    dummy = DoneTerm(func=_always_false)


@configclass
class DiagRewardCfg:
    dummy = RewTerm(func=_zero_reward, weight=1.0)


@configclass
class DiagEnvCfg(ManagerBasedRLEnvCfg):
    scene: DiagSceneCfg = DiagSceneCfg(num_envs=1, env_spacing=5.0)
    observations: DiagObsCfg = DiagObsCfg()
    actions: DiagActionsCfg = DiagActionsCfg()
    terminations: DiagTermCfg = DiagTermCfg()
    rewards: DiagRewardCfg = DiagRewardCfg()
    commands = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


def main():
    env_cfg = DiagEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    print("\n" + "=" * 80)
    print("SYSTEM 0 DIAGNOSTIC: JOINT INFORMATION")
    print("=" * 80)

    joint_names = list(robot.data.joint_names)
    print(f"\nTotal joints: {len(joint_names)}")

    right_hand_indices = []
    left_hand_indices = []
    right_arm_indices = []
    left_arm_indices = []
    other_indices = []

    for i, name in enumerate(joint_names):
        pos = robot.data.joint_pos[0, i].item()
        lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
        default = robot.data.default_joint_pos[0, i].item()

        category = "other"
        if "right_hand" in name:
            right_hand_indices.append(i)
            category = "R_HAND"
        elif "left_hand" in name:
            left_hand_indices.append(i)
            category = "L_HAND"
        elif "right_shoulder" in name or "right_elbow" in name or "right_wrist" in name:
            right_arm_indices.append(i)
            category = "R_ARM"
        elif "left_shoulder" in name or "left_elbow" in name or "left_wrist" in name:
            left_arm_indices.append(i)
            category = "L_ARM"
        else:
            other_indices.append(i)

        print(f"  [{i:2d}] {category:7s} {name:40s} pos={pos:+.4f}  "
              f"limits=[{lo:+.4f}, {hi:+.4f}]  default={default:+.4f}")

    print(f"\n--- INDEX SUMMARY ---")
    print(f"Right hand indices ({len(right_hand_indices)}): {right_hand_indices}")
    print(f"Left hand indices  ({len(left_hand_indices)}):  {left_hand_indices}")
    print(f"Right arm indices  ({len(right_arm_indices)}):  {right_arm_indices}")
    print(f"Left arm indices   ({len(left_arm_indices)}):   {left_arm_indices}")
    print(f"Other indices      ({len(other_indices)}):      {other_indices}")

    # Contact sensor info
    print(f"\n--- CONTACT SENSOR ---")
    try:
        contacts = env.scene["fingertip_contacts"]
        forces = contacts.data.net_forces_w[0]
        print(f"Contact force shape: {forces.shape}")
        print(f"Contact force values: {forces}")
        if hasattr(contacts.data, 'body_names'):
            print(f"Contact body names: {contacts.data.body_names}")
        # Try to find right hand fingertip indices
        if hasattr(contacts, 'body_physx_view'):
            print(f"PhysX view body count: {contacts.body_physx_view.count}")
    except Exception as e:
        print(f"Error reading contacts: {e}")

    # Right wrist position
    print(f"\n--- RIGHT WRIST POSITION ---")
    body_names = list(robot.data.body_names)
    for i, name in enumerate(body_names):
        if "right_wrist" in name or "right_hand" in name:
            pos = robot.data.body_pos_w[0, i].cpu().numpy()
            print(f"  [{i}] {name}: {pos}")

    # Block position
    print(f"\n--- BLOCK POSITION ---")
    block_pos = env.scene["block"].data.root_pos_w[0].cpu().numpy()
    print(f"  Block: {block_pos}")

    # Finger close direction
    print(f"\n--- FINGER CLOSE DIRECTION ---")
    for i in right_hand_indices:
        name = joint_names[i]
        hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
        lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
        print(f"  {name}: lower={lo:+.4f}, upper={hi:+.4f}, "
              f"close_dir={'POSITIVE' if abs(hi) > abs(lo) else 'NEGATIVE'}")

    print(f"\n{'=' * 80}")
    print("COPY THE INDEX LISTS ABOVE INTO config.py")
    print("=" * 80)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

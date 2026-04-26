#!/usr/bin/env python3
"""
Quick targeted arm sweep: test specific configs to find placement positions.

Usage:
    cd ~/unitree_sim_isaaclab
    python -u experiments/system0_skills/quick_sweep.py --num_envs 1 --headless
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg


def set_joints_and_settle(env, robot, joint_pos_tensor, n_steps=30):
    zero_vel = torch.zeros_like(joint_pos_tensor)
    robot.write_joint_state_to_sim(joint_pos_tensor, zero_vel)
    robot.data.joint_pos_target[:] = joint_pos_tensor
    for _ in range(n_steps):
        robot.data.joint_pos_target[:] = joint_pos_tensor
        robot.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)


def _make_robot_cfg():
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
            pos=(0.0, 0.0, 0.76), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.05, "left_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.15, "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.05, "right_knee_joint": 0.2,
                "right_ankle_pitch_joint": -0.15, "right_ankle_roll_joint": 0.0,
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
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=100.0,
                stiffness={".*": 5000.0}, damping={".*": 500.0},
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                  ".*_hand_thumb_.*_joint"],
                effort_limit=300, velocity_limit=100.0,
                stiffness={".*": 500.0}, damping={".*": 50.0},
            ),
        },
    )


def _get_joint_pos(env):
    return env.scene["robot"].data.joint_pos

def _always_false(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class SweepSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_robot_cfg()
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )

@configclass
class SweepObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=_get_joint_pos)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()

@configclass
class SweepActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
    )

@configclass
class SweepTermCfg:
    dummy = DoneTerm(func=_always_false)

@configclass
class SweepRewardCfg:
    dummy = RewTerm(func=_zero_reward, weight=1.0)

@configclass
class SweepEnvCfg(ManagerBasedRLEnvCfg):
    scene: SweepSceneCfg = SweepSceneCfg(num_envs=1, env_spacing=5.0)
    observations: SweepObsCfg = SweepObsCfg()
    actions: SweepActionsCfg = SweepActionsCfg()
    terminations: SweepTermCfg = SweepTermCfg()
    rewards: SweepRewardCfg = SweepRewardCfg()
    commands = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 100.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


def main():
    env_cfg = SweepEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # Build indices
    arm_joint_map = {}
    right_hand_indices = []
    for i, name in enumerate(joint_names):
        if "right_shoulder_pitch" in name: arm_joint_map["sp"] = i
        elif "right_shoulder_roll" in name: arm_joint_map["sr"] = i
        elif "right_shoulder_yaw" in name: arm_joint_map["sy"] = i
        elif "right_elbow" in name and "right" in name: arm_joint_map["el"] = i
        elif "right_wrist_roll" in name: arm_joint_map["wr"] = i
        elif "right_wrist_pitch" in name: arm_joint_map["wp"] = i
        elif "right_wrist_yaw" in name: arm_joint_map["wy"] = i
        elif "right_hand" in name:
            right_hand_indices.append(i)

    palm_idx = None
    tip_indices = {}
    for i, name in enumerate(body_names):
        if "right_wrist_yaw_link" in name and palm_idx is None:
            palm_idx = i
        if "right_hand_thumb_2_link" in name: tip_indices["thumb"] = i
        elif "right_hand_index_1_link" in name: tip_indices["index"] = i
        elif "right_hand_middle_1_link" in name: tip_indices["middle"] = i

    if palm_idx is None:
        for i, name in enumerate(body_names):
            if "right_wrist" in name:
                palm_idx = i
                break

    print(f"Palm: [{palm_idx}] {body_names[palm_idx]}")
    print(f"Tips: {tip_indices}")
    print(f"Arm joint map: {arm_joint_map}")
    sys.stdout.flush()

    # Test specific configs
    configs = []
    configs.append({"sp": 0.0, "sr": -0.5, "sy": 0.1, "el": 0.0, "label": "GRASP_REF"})
    configs.append({"sp": -0.3, "sr": -0.3, "sy": 0.1, "el": 0.0, "label": "TRANSPORT_CURR"})
    configs.append({"sp": 0.0, "sr": -0.3, "sy": 0.1, "el": 0.0, "label": "PLACE_CURR"})
    # Sweep sr at sp=0, sy=0.1
    for sr in [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2]:
        configs.append({"sp": 0.0, "sr": sr, "sy": 0.1, "el": 0.0, "label": f"PLACE_sr{sr:+.1f}"})
    # Sweep sr at sp=-0.3, sy=0.1
    for sr in [-0.4, -0.2, -0.1, 0.0, 0.1]:
        configs.append({"sp": -0.3, "sr": sr, "sy": 0.1, "el": 0.0, "label": f"TRANS_sr{sr:+.1f}"})
    # Vary sy at sr=-0.1
    for sy in [0.0, 0.2, 0.3, 0.5]:
        configs.append({"sp": 0.0, "sr": -0.1, "sy": sy, "el": 0.0, "label": f"PL_sr-0.1_sy{sy:.1f}"})
        configs.append({"sp": -0.3, "sr": -0.1, "sy": sy, "el": 0.0, "label": f"TR_sr-0.1_sy{sy:.1f}"})

    print(f"\nTesting {len(configs)} configs...")
    print(f"{'label':30s} | {'sp':>5} {'sr':>5} {'sy':>5} | {'palm_x':>7} {'palm_y':>7} {'palm_z':>7} | {'tip_x':>6} {'tip_y':>6} {'tip_z':>6}")
    print("-" * 105)
    sys.stdout.flush()

    for cfg in configs:
        new_pos = robot.data.default_joint_pos.clone()
        new_pos[0, arm_joint_map["sp"]] = cfg["sp"]
        new_pos[0, arm_joint_map["sr"]] = cfg["sr"]
        new_pos[0, arm_joint_map["sy"]] = cfg["sy"]
        new_pos[0, arm_joint_map["el"]] = cfg["el"]
        new_pos[0, arm_joint_map["wr"]] = 0.0
        new_pos[0, arm_joint_map["wp"]] = 0.0
        new_pos[0, arm_joint_map["wy"]] = 0.0
        # 50% closure
        for hi_idx in right_hand_indices:
            lo = robot.data.soft_joint_pos_limits[0, hi_idx, 0].item()
            hi = robot.data.soft_joint_pos_limits[0, hi_idx, 1].item()
            name = joint_names[hi_idx]
            if "thumb_2" in name:
                val = hi + (lo - hi) * 0.5
            elif abs(hi) > abs(lo):
                val = lo + (hi - lo) * 0.5
            else:
                val = hi + (lo - hi) * 0.5
            new_pos[0, hi_idx] = val

        set_joints_and_settle(env, robot, new_pos, n_steps=25)

        palm_pos = robot.data.body_pos_w[0, palm_idx].cpu().numpy()
        tips_pos = {}
        for tname, tidx in tip_indices.items():
            tips_pos[tname] = robot.data.body_pos_w[0, tidx].cpu().numpy()

        avg_tip_x = np.mean([tips_pos[k][0] for k in tips_pos])
        avg_tip_y = np.mean([tips_pos[k][1] for k in tips_pos])
        avg_tip_z = np.mean([tips_pos[k][2] for k in tips_pos])

        is_down = palm_pos[2] > avg_tip_z + 0.01
        marker = "OK" if is_down and palm_pos[0] > 0.05 else "BAD"

        print(f"{cfg['label']:30s} | {cfg['sp']:5.1f} {cfg['sr']:5.1f} {cfg['sy']:5.1f} | "
              f"{palm_pos[0]:7.3f} {palm_pos[1]:7.3f} {palm_pos[2]:7.3f} | "
              f"{avg_tip_x:6.3f} {avg_tip_y:6.3f} {avg_tip_z:6.3f} [{marker}]")
        sys.stdout.flush()

    print("\nDONE")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

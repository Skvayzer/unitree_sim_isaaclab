#!/usr/bin/env python3
"""
System 0 Table Grasp Demo — Natural grasping from a table.

Scene: Robot standing behind a table. Block on the table.
Phase 1 (reach): Arm moves down to position hand around the block (scripted).
Phase 2 (grasp): Trained finger policy closes fingers to grasp.
Phase 3 (lift):  Arm lifts back up while finger policy maintains grip.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/eval_table_grasp.py --num_envs 1
    python experiments/system0_skills/eval_table_grasp.py --num_envs 1 --headless
"""

import os
import sys
import math
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="System 0 Table Grasp Demo")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--grasp_ckpt", type=str, default="logs/system0_grasp/model_999.pt")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
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
project_root = os.environ.get("PROJECT_ROOT")

# --- Scene dimensions ---
TABLE_HEIGHT = 0.72        # table surface height (robot pelvis at 0.76)
TABLE_SIZE = (0.6, 0.8, TABLE_HEIGHT)  # length x width x height
TABLE_POS = (0.35, 0.0, TABLE_HEIGHT / 2)  # center of table cuboid

BLOCK_SIZE = 0.04
BLOCK_ON_TABLE_Z = TABLE_HEIGHT + BLOCK_SIZE / 2 + 0.001  # sitting on table
BLOCK_POS = (0.30, -0.15, BLOCK_ON_TABLE_Z)  # in front of right hand

# Arm joint targets for reach and lift phases
# These position the right hand near the block on the table
# Right arm indices in joint array: [12, 16, 20, 22, 24, 26, 28]
# = [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
ARM_REST = {
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Reach pose: shoulder pitched forward and down, elbow bent, to bring hand to table level
ARM_REACH = {
    "right_shoulder_pitch_joint": 0.6,    # lean forward
    "right_shoulder_roll_joint": -0.3,    # reach toward midline
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,            # bend elbow
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Lift pose: shoulder pitched less, elbow slightly bent — lifts block off table
ARM_LIFT = {
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.4,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def _make_table_grasp_robot() -> ArticulationCfg:
    """Robot with arm that can follow position targets (medium stiffness)."""
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
                # Legs — locked
                "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.05, "left_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.15, "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.05, "right_knee_joint": 0.2,
                "right_ankle_pitch_joint": -0.15, "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
                # Left arm — locked at rest
                "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                # Right arm — starts at rest, will be moved to reach
                "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Left hand — open
                "left_hand_index_0_joint": 0.0, "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                # Right hand — open (will close around block)
                "right_hand_index_0_joint": 0.1, "right_hand_middle_0_joint": 0.1,
                "right_hand_thumb_0_joint": -0.05, "right_hand_index_1_joint": 0.1,
                "right_hand_middle_1_joint": 0.1, "right_hand_thumb_1_joint": -0.05,
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
            # Right arm — soft enough to follow scripted targets
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*_joint", "right_elbow_joint",
                                  "right_wrist_.*_joint"],
                effort_limit=300.0, velocity_limit=10.0,
                stiffness={".*": 200.0}, damping={".*": 50.0},
            ),
            # Left arm — locked
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*_joint", "left_elbow_joint",
                                  "left_wrist_.*_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            # Hands — soft, controlled by policy
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                  ".*_hand_thumb_.*_joint"],
                effort_limit=300, velocity_limit=100.0,
                stiffness={".*": 100.0}, damping={".*": 10.0},
                armature={".*": 0.1},
            ),
        },
    )


# --- Observation (same 21D as grasp env for policy compatibility) ---

def obs_finger_pos(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, CFG.right_hand_indices]

def obs_finger_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel[:, CFG.right_hand_indices]

def obs_contact_force(env) -> torch.Tensor:
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    rf = forces[:, CFG.right_fingertip_contact_indices, :]
    return rf.norm(dim=-1).clamp(0, 5.0)

def obs_target_force(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 3, device=env.device) * CFG.target_force

def obs_object_grasped(env) -> torch.Tensor:
    bz = env.scene["block"].data.root_pos_w[:, 2]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    rf = forces[:, CFG.right_fingertip_contact_indices, :]
    cc = (rf.norm(dim=-1) > 0.1).sum(dim=-1)
    grasped = (bz > BLOCK_ON_TABLE_Z - 0.05).float()
    return grasped.unsqueeze(-1)

def dummy_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

def never_done(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def timeout_done(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# --- Scene ---

@configclass
class TableGraspSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_table_grasp_robot()

    # Table
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # wood brown
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.5,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # Block on table
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


# --- Env config ---

@configclass
class TableGraspActionsCfg:
    finger_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["right_hand_.*_joint"],
        scale=CFG.action_scale, use_default_offset=True,
    )

@configclass
class TableGraspObsCfg:
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
class TableGraspRewardsCfg:
    dummy = RewTerm(func=dummy_reward, weight=1.0)

@configclass
class TableGraspTerminationsCfg:
    time_out = DoneTerm(func=timeout_done, time_out=True)

@configclass
class TableGraspEventCfg:
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_block = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )

@configclass
class TableGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: TableGraspSceneCfg = TableGraspSceneCfg(num_envs=1, env_spacing=3.0)
    observations: TableGraspObsCfg = TableGraspObsCfg()
    actions: TableGraspActionsCfg = TableGraspActionsCfg()
    rewards: TableGraspRewardsCfg = TableGraspRewardsCfg()
    terminations: TableGraspTerminationsCfg = TableGraspTerminationsCfg()
    events: TableGraspEventCfg = TableGraspEventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = CFG.decimation
        self.episode_length_s = 60.0  # long episode for demo
        self.sim.dt = CFG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


# =========================================================================
# Main
# =========================================================================

def load_grasp_actor(ckpt_path: str, device: torch.device):
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(21, 128), nn.ELU(),
                nn.Linear(128, 128), nn.ELU(),
                nn.Linear(128, 7),
            )
        def forward(self, obs): return self.net(obs)

    actor = Actor().to(device)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        asd = {k.replace("actor.", "net.", 1): v for k, v in sd.items() if k.startswith("actor.")}
        actor.load_state_dict(asd)
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"Loaded grasp actor from {ckpt_path}")
    return actor


def interpolate_arm_targets(robot, arm_indices, start_dict, end_dict, t, device):
    """Smoothly interpolate arm joint targets from start to end. t in [0, 1]."""
    # Smooth easing
    t = t * t * (3 - 2 * t)  # smoothstep

    joint_names = list(robot.data.joint_names)
    targets = robot.data.joint_pos_target.clone()

    for joint_name in start_dict:
        idx = joint_names.index(joint_name)
        start_val = start_dict[joint_name]
        end_val = end_dict[joint_name]
        targets[:, idx] = start_val + (end_val - start_val) * t

    robot.data.joint_pos_target[:] = targets


def main():
    num_envs = args.num_envs

    env_cfg = TableGraspEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    grasp_actor = load_grasp_actor(args.grasp_ckpt, device)
    robot = env.scene["robot"]
    arm_indices = torch.tensor(CFG.right_arm_indices, device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    prev_action = torch.zeros(num_envs, 7, device=device)

    # Phase timing (in steps, at dt=0.005 * decimation=2 → 100 Hz)
    REACH_STEPS = 200     # 2 seconds to reach
    GRASP_STEPS = 200     # 2 seconds to grasp
    LIFT_STEPS = 200      # 2 seconds to lift
    HOLD_STEPS = 400      # 4 seconds to show holding
    TOTAL_STEPS = REACH_STEPS + GRASP_STEPS + LIFT_STEPS + HOLD_STEPS

    print(f"\n{'='*60}")
    print(f"System 0 Table Grasp Demo")
    print(f"  Phase 1 (0-2s):   Reach — arm moves to block")
    print(f"  Phase 2 (2-4s):   Grasp — fingers close around block")
    print(f"  Phase 3 (4-6s):   Lift  — arm lifts block off table")
    print(f"  Phase 4 (6-10s):  Hold  — demonstrate stable grip")
    print(f"{'='*60}\n")

    for step in range(TOTAL_STEPS):
        # --- Phase 1: Reach ---
        if step < REACH_STEPS:
            phase = "REACH"
            t = step / REACH_STEPS
            interpolate_arm_targets(robot, arm_indices, ARM_REST, ARM_REACH, t, device)
            # Fingers stay open (zero action)
            action = torch.zeros(num_envs, 7, device=device)

        # --- Phase 2: Grasp ---
        elif step < REACH_STEPS + GRASP_STEPS:
            phase = "GRASP"
            # Keep arm at reach position
            interpolate_arm_targets(robot, arm_indices, ARM_REACH, ARM_REACH, 1.0, device)
            # Use trained grasp policy
            with torch.no_grad():
                raw_action = grasp_actor(obs).clamp(-1.0, 1.0)
            action = 0.7 * raw_action + 0.3 * prev_action

        # --- Phase 3: Lift ---
        elif step < REACH_STEPS + GRASP_STEPS + LIFT_STEPS:
            phase = "LIFT"
            t = (step - REACH_STEPS - GRASP_STEPS) / LIFT_STEPS
            interpolate_arm_targets(robot, arm_indices, ARM_REACH, ARM_LIFT, t, device)
            # Continue grasp policy to maintain grip
            with torch.no_grad():
                raw_action = grasp_actor(obs).clamp(-1.0, 1.0)
            action = 0.7 * raw_action + 0.3 * prev_action

        # --- Phase 4: Hold ---
        else:
            phase = "HOLD"
            interpolate_arm_targets(robot, arm_indices, ARM_LIFT, ARM_LIFT, 1.0, device)
            with torch.no_grad():
                raw_action = grasp_actor(obs).clamp(-1.0, 1.0)
            action = 0.7 * raw_action + 0.3 * prev_action

        prev_action = action.clone()

        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        # Log every 50 steps
        if step % 50 == 0:
            block_pos = env.scene["block"].data.root_pos_w[0].cpu().numpy()
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            rf = forces[:, CFG.right_fingertip_contact_indices, :]
            fm = rf.norm(dim=-1)
            cc = (fm > 0.1).float().sum(dim=-1).mean().item()
            mf = fm.mean().item()
            t_sec = step * CFG.sim_dt * CFG.decimation
            print(
                f"[{t_sec:5.1f}s] {phase:5s} | "
                f"block=[{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}] | "
                f"contacts={cc:.1f} force={mf:.1f}"
            )

    # Final report
    block_z = env.scene["block"].data.root_pos_w[0, 2].item()
    lifted = block_z > TABLE_HEIGHT + 0.02
    print(f"\n{'='*60}")
    print(f"Result: block_z = {block_z:.4f}")
    if lifted:
        print(f"SUCCESS — Block lifted {block_z - TABLE_HEIGHT:.3f}m above table!")
    else:
        print(f"FAILED — Block still on table or dropped.")
    print(f"{'='*60}")

    # Keep viewer alive
    if not args.headless:
        print("\nViewer open. Close window or Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                with torch.no_grad():
                    raw_action = grasp_actor(obs).clamp(-1.0, 1.0)
                action = 0.7 * raw_action + 0.3 * prev_action
                prev_action = action.clone()
                obs_dict, _, _, _, _ = env.step(action)
                obs = obs_dict["policy"]
        except KeyboardInterrupt:
            print("\nStopped.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

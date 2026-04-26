#!/usr/bin/env python3
"""
DIAGNOSTIC: Find arm workspace and finger sweep envelope.

This script systematically sweeps arm joint angles and records where
the hand ends up. It then determines the optimal table height and block
position for grasping training.

USAGE:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/diagnostic.py --num_envs 1 --headless
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
from isaaclab.actuators import ImplicitActuatorCfg


# ---------------------------------------------------------------------------
# Minimal robot config with arms controllable (medium stiffness)
# ---------------------------------------------------------------------------
def _make_diagnostic_robot_cfg() -> ArticulationCfg:
    """Robot with arms at medium stiffness so we can set joint targets and they track."""
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
            pos=(0.0, 0.0, 0.76),
            rot=(1.0, 0.0, 0.0, 0.0),
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
                # Fingers — start open
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
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
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint",
                    ".*_hip_pitch_joint", ".*_knee_joint",
                ],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            # Lock feet
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=1000.0, velocity_limit=0.0,
                stiffness={".*": 10000.0}, damping={".*": 10000.0},
            ),
            # Arms — high stiffness for precise position control during diagnostic
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint",
                ],
                effort_limit=1000.0, velocity_limit=100.0,
                stiffness={".*": 5000.0}, damping={".*": 500.0},
            ),
            # Hands — controllable
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                    ".*_hand_thumb_.*_joint",
                ],
                effort_limit=300, velocity_limit=100.0,
                stiffness={".*": 500.0}, damping={".*": 50.0},
            ),
        },
    )


# ---------------------------------------------------------------------------
# Observation / termination / reward stubs
# ---------------------------------------------------------------------------
def _get_joint_pos(env):
    return env.scene["robot"].data.joint_pos

def _always_false(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


# ---------------------------------------------------------------------------
# Scene config — robot only, no block needed for workspace sweep
# ---------------------------------------------------------------------------
@configclass
class DiagSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_diagnostic_robot_cfg()

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
        self.decimation = 1
        self.episode_length_s = 100.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


# ---------------------------------------------------------------------------
# Helper: set joint positions and settle
# ---------------------------------------------------------------------------
def set_joints_and_settle(env, robot, joint_pos_tensor, n_steps=30):
    """Write joint positions as targets and step sim to settle.

    Uses write_joint_state_to_sim to teleport joints, then steps
    with the action manager writing position targets.
    """
    zero_vel = torch.zeros_like(joint_pos_tensor)
    robot.write_joint_state_to_sim(joint_pos_tensor, zero_vel)
    # Also set the position target via the articulation's internal API
    robot.data.joint_pos_target[:] = joint_pos_tensor
    for _ in range(n_steps):
        # Write targets each step to keep actuators driving to desired pos
        robot.data.joint_pos_target[:] = joint_pos_tensor
        robot.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def main():
    env_cfg = DiagEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # ======================================================================
    # 1. Print ALL joint names, indices, limits
    # ======================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: ALL JOINTS")
    print("=" * 80)
    print(f"Total joints: {len(joint_names)}")

    right_hand_indices = []
    right_arm_indices = []

    for i, name in enumerate(joint_names):
        pos = robot.data.joint_pos[0, i].item()
        lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
        default = robot.data.default_joint_pos[0, i].item()

        category = ""
        if "right_hand" in name:
            right_hand_indices.append(i)
            category = "R_HAND"
        elif any(x in name for x in ["right_shoulder", "right_elbow", "right_wrist"]):
            right_arm_indices.append(i)
            category = "R_ARM"

        print(f"  [{i:2d}] {category:7s} {name:40s} pos={pos:+.4f}  "
              f"limits=[{lo:+.4f}, {hi:+.4f}]  default={default:+.4f}")

    print(f"\n--- INDEX SUMMARY ---")
    print(f"Right arm indices  ({len(right_arm_indices)}):  {right_arm_indices}")
    print(f"Right hand indices ({len(right_hand_indices)}): {right_hand_indices}")

    # ======================================================================
    # 2. Print ALL body names with positions
    # ======================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: ALL BODY NAMES (right hand/wrist)")
    print("=" * 80)
    print(f"Total bodies: {len(body_names)}")

    for i, name in enumerate(body_names):
        if "right" in name and ("hand" in name or "wrist" in name):
            pos = robot.data.body_pos_w[0, i].cpu().numpy()
            quat = robot.data.body_quat_w[0, i].cpu().numpy()
            print(f"  [{i:2d}] {name:40s} pos={pos}  quat={quat}")

    # Also print ALL bodies with "hand" for contact sensor mapping
    print(f"\nALL hand bodies (for contact sensor):")
    for i, name in enumerate(body_names):
        if "hand" in name:
            pos = robot.data.body_pos_w[0, i].cpu().numpy()
            print(f"  [{i:2d}] {name:40s} pos={pos}")

    # ======================================================================
    # 3. Identify right arm joint names in order
    # ======================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: RIGHT ARM JOINT MAPPING")
    print("=" * 80)
    for idx in right_arm_indices:
        lo = robot.data.soft_joint_pos_limits[0, idx, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, idx, 1].item()
        print(f"  [{idx:2d}] {joint_names[idx]:40s} limits=[{lo:+.4f}, {hi:+.4f}]")

    print(f"\nRight hand joints:")
    for idx in right_hand_indices:
        lo = robot.data.soft_joint_pos_limits[0, idx, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, idx, 1].item()
        print(f"  [{idx:2d}] {joint_names[idx]:40s} limits=[{lo:+.4f}, {hi:+.4f}]")

    # ======================================================================
    # 4. Sweep arm joint angles systematically
    # ======================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: ARM WORKSPACE SWEEP")
    print("=" * 80)

    # Map arm joint names to their indices
    arm_joint_map = {}
    for idx in right_arm_indices:
        name = joint_names[idx]
        if "shoulder_pitch" in name:
            arm_joint_map["sp"] = idx
        elif "shoulder_roll" in name:
            arm_joint_map["sr"] = idx
        elif "shoulder_yaw" in name:
            arm_joint_map["sy"] = idx
        elif "elbow" in name:
            arm_joint_map["el"] = idx
        elif "wrist_roll" in name:
            arm_joint_map["wr"] = idx
        elif "wrist_pitch" in name:
            arm_joint_map["wp"] = idx
        elif "wrist_yaw" in name:
            arm_joint_map["wy"] = idx

    print(f"Arm joint map: {arm_joint_map}")
    for key, idx in arm_joint_map.items():
        lo = robot.data.soft_joint_pos_limits[0, idx, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, idx, 1].item()
        print(f"  {key} ({joint_names[idx]}): [{lo:+.4f}, {hi:+.4f}]")

    # Find right hand body indices for palm and fingertips
    palm_idx = None
    tip_indices = {}
    for i, name in enumerate(body_names):
        if "right_hand_palm" in name or "right_wrist_yaw_link" in name:
            if palm_idx is None:
                palm_idx = i
                print(f"\nUsing palm body: [{i}] {name}")
        # Fingertip links — the distal links
        if "right_hand_thumb_2_link" in name:
            tip_indices["thumb_tip"] = i
        elif "right_hand_index_1_link" in name:
            tip_indices["index_tip"] = i
        elif "right_hand_middle_1_link" in name:
            tip_indices["middle_tip"] = i
        # Also track proximal links for reference
        if "right_hand_thumb_0_link" in name:
            tip_indices["thumb_base"] = i
        elif "right_hand_index_0_link" in name:
            tip_indices["index_base"] = i
        elif "right_hand_middle_0_link" in name:
            tip_indices["middle_base"] = i

    print(f"Fingertip body indices: {tip_indices}")

    # If no palm found, use wrist link
    if palm_idx is None:
        for i, name in enumerate(body_names):
            if "right_wrist" in name:
                palm_idx = i
                print(f"Falling back to wrist body: [{i}] {name}")
                break

    best_configs = []

    # Sweep: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    # shoulder_pitch: forward reach (positive = forward for G1)
    # shoulder_roll: lateral (negative = away from body for right arm)
    # shoulder_yaw: rotation
    # elbow: bend

    sp_range = np.arange(0.0, 1.6, 0.2)
    sr_range = np.arange(-0.5, 0.3, 0.1)
    sy_range = np.arange(-0.5, 1.0, 0.3)
    el_range = np.arange(0.0, 1.6, 0.2)

    total_configs = len(sp_range) * len(sr_range) * len(sy_range) * len(el_range)
    print(f"\nSweeping {total_configs} configurations...")
    print(f"  sp: {list(sp_range)}")
    print(f"  sr: {list(sr_range)}")
    print(f"  sy: {list(sy_range)}")
    print(f"  el: {list(el_range)}")

    count = 0
    for sp in sp_range:
        for sr in sr_range:
            for sy in sy_range:
                for el in el_range:
                    # Set arm joints
                    new_pos = robot.data.default_joint_pos.clone()
                    new_pos[0, arm_joint_map["sp"]] = sp
                    new_pos[0, arm_joint_map["sr"]] = sr
                    new_pos[0, arm_joint_map["sy"]] = sy
                    new_pos[0, arm_joint_map["el"]] = el
                    # Keep wrist neutral
                    new_pos[0, arm_joint_map["wr"]] = 0.0
                    new_pos[0, arm_joint_map["wp"]] = 0.0
                    new_pos[0, arm_joint_map["wy"]] = 0.0
                    # Keep fingers open
                    for hi_idx in right_hand_indices:
                        new_pos[0, hi_idx] = robot.data.default_joint_pos[0, hi_idx]

                    set_joints_and_settle(env, robot, new_pos, n_steps=20)

                    # Read positions
                    palm_pos = robot.data.body_pos_w[0, palm_idx].cpu().numpy()
                    tips = {}
                    for tname, tidx in tip_indices.items():
                        tips[tname] = robot.data.body_pos_w[0, tidx].cpu().numpy()

                    # Check: palm pointing downward (palm z > fingertip z)
                    tip_z_vals = [tips[k][2] for k in ["thumb_tip", "index_tip", "middle_tip"] if k in tips]
                    if tip_z_vals:
                        avg_tip_z = np.mean(tip_z_vals)
                    else:
                        avg_tip_z = palm_pos[2]

                    is_downward = palm_pos[2] > avg_tip_z + 0.01

                    # Is it in front of robot? (x > 0)
                    is_forward = palm_pos[0] > 0.05

                    if is_downward and is_forward:
                        best_configs.append({
                            "sp": float(sp), "sr": float(sr),
                            "sy": float(sy), "el": float(el),
                            "palm": palm_pos.copy(),
                            "tips": {k: v.copy() for k, v in tips.items()},
                            "avg_tip_z": float(avg_tip_z),
                        })

                    count += 1
                    if count % 100 == 0:
                        print(f"  ... {count}/{total_configs} configs tested, "
                              f"{len(best_configs)} viable so far")

    # Sort by forward reach (x) of palm
    best_configs.sort(key=lambda c: c["palm"][0], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"SECTION 4 RESULTS: TOP 20 REACHING CONFIGURATIONS")
    print(f"{'=' * 80}")
    print(f"Total viable configs (palm down, forward): {len(best_configs)}")
    for i, cfg in enumerate(best_configs[:20]):
        print(f"\n  #{i}: sp={cfg['sp']:.1f} sr={cfg['sr']:.1f} "
              f"sy={cfg['sy']:.1f} el={cfg['el']:.1f}")
        print(f"       palm={cfg['palm']}  avg_tip_z={cfg['avg_tip_z']:.4f}")
        for tname, tpos in cfg['tips'].items():
            print(f"       {tname:20s}: {tpos}")

    # ======================================================================
    # 5. For the BEST config, sweep finger closure 0% to 100%
    # ======================================================================
    if not best_configs:
        print("\nERROR: No viable reaching configurations found!")
        env.close()
        simulation_app.close()
        return

    best = best_configs[0]
    print(f"\n{'=' * 80}")
    print(f"SECTION 5: FINGER SWEEP ENVELOPE")
    print(f"Best arm config: sp={best['sp']:.1f} sr={best['sr']:.1f} "
          f"sy={best['sy']:.1f} el={best['el']:.1f}")
    print(f"{'=' * 80}")

    finger_sweep_data = {}  # closure_pct -> {body_name: pos}

    for closure_pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        new_pos = robot.data.default_joint_pos.clone()
        # Set arm to best config
        new_pos[0, arm_joint_map["sp"]] = best["sp"]
        new_pos[0, arm_joint_map["sr"]] = best["sr"]
        new_pos[0, arm_joint_map["sy"]] = best["sy"]
        new_pos[0, arm_joint_map["el"]] = best["el"]
        new_pos[0, arm_joint_map["wr"]] = 0.0
        new_pos[0, arm_joint_map["wp"]] = 0.0
        new_pos[0, arm_joint_map["wy"]] = 0.0

        # Set finger closure
        for hi_idx in right_hand_indices:
            lo = robot.data.soft_joint_pos_limits[0, hi_idx, 0].item()
            hi = robot.data.soft_joint_pos_limits[0, hi_idx, 1].item()
            name = joint_names[hi_idx]
            # Determine close direction
            # For index and middle: closing = increasing (positive direction)
            # For thumb: thumb_0 is yaw, thumb_1 is pitch, thumb_2 is curl
            if "thumb_2" in name:
                # thumb_2 has negative limits, closing = more negative
                val = hi + (lo - hi) * closure_pct / 100.0
            elif abs(hi) > abs(lo):
                # Close = towards upper limit
                val = lo + (hi - lo) * closure_pct / 100.0
            else:
                # Close = towards lower limit
                val = hi + (lo - hi) * closure_pct / 100.0
            new_pos[0, hi_idx] = val

        set_joints_and_settle(env, robot, new_pos, n_steps=30)

        print(f"\n  Closure {closure_pct:3d}%:")
        sweep_entry = {}
        for i, name in enumerate(body_names):
            if "right_hand" in name:
                pos = robot.data.body_pos_w[0, i].cpu().numpy()
                print(f"    [{i:2d}] {name:40s} pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
                sweep_entry[name] = pos.copy()
        finger_sweep_data[closure_pct] = sweep_entry

    # ======================================================================
    # 6. Also try a few wrist pitch variations to tilt hand more downward
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"SECTION 6: WRIST PITCH SWEEP (at best arm config)")
    print(f"{'=' * 80}")

    wp_lo = robot.data.soft_joint_pos_limits[0, arm_joint_map["wp"], 0].item()
    wp_hi = robot.data.soft_joint_pos_limits[0, arm_joint_map["wp"], 1].item()
    print(f"Wrist pitch limits: [{wp_lo:+.4f}, {wp_hi:+.4f}]")

    for wp_val in np.arange(wp_lo, wp_hi + 0.1, 0.2):
        wp_val = float(np.clip(wp_val, wp_lo, wp_hi))
        new_pos = robot.data.default_joint_pos.clone()
        new_pos[0, arm_joint_map["sp"]] = best["sp"]
        new_pos[0, arm_joint_map["sr"]] = best["sr"]
        new_pos[0, arm_joint_map["sy"]] = best["sy"]
        new_pos[0, arm_joint_map["el"]] = best["el"]
        new_pos[0, arm_joint_map["wp"]] = wp_val
        # Fingers at 50% closure
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

        set_joints_and_settle(env, robot, new_pos, n_steps=30)

        palm_pos = robot.data.body_pos_w[0, palm_idx].cpu().numpy()
        print(f"\n  wrist_pitch={wp_val:+.3f}:")
        print(f"    palm: [{palm_pos[0]:+.4f}, {palm_pos[1]:+.4f}, {palm_pos[2]:+.4f}]")
        for tname, tidx in tip_indices.items():
            if "tip" in tname:
                pos = robot.data.body_pos_w[0, tidx].cpu().numpy()
                print(f"    {tname:15s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    # ======================================================================
    # 7. Compute recommended positions
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"SECTION 7: RECOMMENDED POSITIONS")
    print(f"{'=' * 80}")

    # Get 0% and 100% fingertip positions for the best arm config
    tips_open = {}
    tips_closed = {}
    tips_50 = {}

    for tname in ["thumb_tip", "index_tip", "middle_tip"]:
        # Find the body name that matches
        for bname in finger_sweep_data.get(0, {}):
            if tname.replace("_tip", "") in bname and ("2_link" in bname or "1_link" in bname):
                if "thumb" in tname and "thumb_2" in bname:
                    tips_open[tname] = finger_sweep_data[0][bname]
                    tips_closed[tname] = finger_sweep_data[100][bname]
                    tips_50[tname] = finger_sweep_data[50][bname]
                elif "index" in tname and "index_1" in bname:
                    tips_open[tname] = finger_sweep_data[0][bname]
                    tips_closed[tname] = finger_sweep_data[100][bname]
                    tips_50[tname] = finger_sweep_data[50][bname]
                elif "middle" in tname and "middle_1" in bname:
                    tips_open[tname] = finger_sweep_data[0][bname]
                    tips_closed[tname] = finger_sweep_data[100][bname]
                    tips_50[tname] = finger_sweep_data[50][bname]

    print(f"\nFingertips OPEN (0% closure):")
    for tname, pos in tips_open.items():
        print(f"  {tname:15s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    print(f"\nFingertips 50% closure:")
    for tname, pos in tips_50.items():
        print(f"  {tname:15s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    print(f"\nFingertips CLOSED (100% closure):")
    for tname, pos in tips_closed.items():
        print(f"  {tname:15s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    # Compute center of finger sweep at 50% closure
    if tips_50:
        avg_x_50 = np.mean([p[0] for p in tips_50.values()])
        avg_y_50 = np.mean([p[1] for p in tips_50.values()])
        avg_z_50 = np.mean([p[2] for p in tips_50.values()])

        avg_x_open = np.mean([p[0] for p in tips_open.values()])
        avg_y_open = np.mean([p[1] for p in tips_open.values()])
        avg_z_open = np.mean([p[2] for p in tips_open.values()])

        avg_x_closed = np.mean([p[0] for p in tips_closed.values()])
        avg_y_closed = np.mean([p[1] for p in tips_closed.values()])
        avg_z_closed = np.mean([p[2] for p in tips_closed.values()])

        # Sweep center = midpoint of open and closed
        sweep_center_x = (avg_x_open + avg_x_closed) / 2.0
        sweep_center_y = (avg_y_open + avg_y_closed) / 2.0
        sweep_center_z = (avg_z_open + avg_z_closed) / 2.0

        block_size = 0.04
        # Table height: fingertip z at 50% closure minus half block height
        # The block top should be at the fingertip z at ~40-50% closure
        table_height = avg_z_50 - block_size / 2.0

        # Block center should be in the sweep path
        block_x = sweep_center_x
        block_y = sweep_center_y
        block_z = table_height + block_size / 2.0

        # Table position: centered under the block
        table_x = block_x
        table_y = block_y

        print(f"\n--- COMPUTED RECOMMENDATIONS ---")
        print(f"Sweep center (open-closed midpoint): "
              f"x={sweep_center_x:.4f}, y={sweep_center_y:.4f}, z={sweep_center_z:.4f}")
        print(f"50% closure avg: x={avg_x_50:.4f}, y={avg_y_50:.4f}, z={avg_z_50:.4f}")
        print(f"")
        print(f"RECOMMENDED TABLE HEIGHT: {table_height:.4f}")
        print(f"RECOMMENDED TABLE POS:    ({table_x:.4f}, {table_y:.4f}, {table_height/2:.4f})")
        print(f"RECOMMENDED BLOCK POS:    ({block_x:.4f}, {block_y:.4f}, {block_z:.4f})")
        print(f"BLOCK SIZE: {block_size}")
        print(f"")
        print(f"ARM REACH (GRASP) CONFIG:")
        print(f"  right_shoulder_pitch_joint: {best['sp']:.2f}")
        print(f"  right_shoulder_roll_joint:  {best['sr']:.2f}")
        print(f"  right_shoulder_yaw_joint:   {best['sy']:.2f}")
        print(f"  right_elbow_joint:          {best['el']:.2f}")
        print(f"  right_wrist_roll_joint:     0.00")
        print(f"  right_wrist_pitch_joint:    0.00")
        print(f"  right_wrist_yaw_joint:      0.00")

        # LIFT config: reduce shoulder_pitch slightly to raise hand
        lift_sp = max(0.0, best["sp"] - 0.3)
        print(f"")
        print(f"ARM LIFT CONFIG (sp reduced by 0.3):")
        print(f"  right_shoulder_pitch_joint: {lift_sp:.2f}")
        print(f"  right_shoulder_roll_joint:  {best['sr']:.2f}")
        print(f"  right_shoulder_yaw_joint:   {best['sy']:.2f}")
        print(f"  right_elbow_joint:          {best['el']:.2f}")

    # ======================================================================
    # 8. Contact sensor body indices
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"SECTION 8: CONTACT SENSOR INFO")
    print(f"{'=' * 80}")

    try:
        contacts = env.scene["fingertip_contacts"]
        forces = contacts.data.net_forces_w[0]
        print(f"Contact force tensor shape: {forces.shape}")
        print(f"Number of bodies in contact sensor: {forces.shape[0]}")

        # The sensor matches .*_hand_.*_link — enumerate which bodies
        # We need to figure out the mapping from sensor index to body name
        if hasattr(contacts, 'body_physx_view'):
            print(f"PhysX view count: {contacts.body_physx_view.count}")

        # Print forces to see the shape
        print(f"Force values (should be ~zero at rest):")
        for bi in range(forces.shape[0]):
            f = forces[bi].cpu().numpy()
            print(f"  sensor_body[{bi:2d}]: force=[{f[0]:+.4f}, {f[1]:+.4f}, {f[2]:+.4f}]")

        # Try to identify bodies via body_names attribute
        if hasattr(contacts.data, 'body_names'):
            print(f"\nContact sensor body names: {contacts.data.body_names}")
        elif hasattr(contacts, '_body_names'):
            print(f"\nContact sensor body names: {contacts._body_names}")

        # The contact sensor prim_path is .*_hand_.*_link
        # Let's enumerate all matching bodies from the robot
        print(f"\nBodies matching '.*_hand_.*_link' pattern:")
        hand_bodies = []
        for i, name in enumerate(body_names):
            if "hand" in name and "link" in name:
                hand_bodies.append((i, name))
                print(f"  robot_body[{i:2d}] = {name}")

        print(f"\nTotal hand link bodies: {len(hand_bodies)}")
        print(f"Contact sensor reports {forces.shape[0]} bodies")
        print(f"\nAssuming sensor bodies are in same order as USD traversal:")
        print(f"The contact sensor body indices for right fingertips are:")
        print(f"(Look for right_hand_thumb_2_link, right_hand_index_1_link, "
              f"right_hand_middle_1_link)")
        for sensor_idx, (robot_idx, name) in enumerate(hand_bodies):
            if any(x in name for x in ["right_hand_thumb_2", "right_hand_index_1", "right_hand_middle_1"]):
                print(f"  >> sensor_body[{sensor_idx}] = robot_body[{robot_idx}] = {name}")

    except Exception as e:
        print(f"Error reading contacts: {e}")
        import traceback
        traceback.print_exc()

    # ======================================================================
    # 9. Try alternative arm configs (top 5) with finger sweep
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"SECTION 9: TOP 5 CONFIGS - QUICK FINGER SWEEP")
    print(f"{'=' * 80}")

    for cfg_idx, cfg in enumerate(best_configs[:5]):
        print(f"\n--- Config #{cfg_idx}: sp={cfg['sp']:.1f} sr={cfg['sr']:.1f} "
              f"sy={cfg['sy']:.1f} el={cfg['el']:.1f} ---")
        print(f"    Palm: {cfg['palm']}")

        for closure_pct in [0, 50, 100]:
            new_pos = robot.data.default_joint_pos.clone()
            new_pos[0, arm_joint_map["sp"]] = cfg["sp"]
            new_pos[0, arm_joint_map["sr"]] = cfg["sr"]
            new_pos[0, arm_joint_map["sy"]] = cfg["sy"]
            new_pos[0, arm_joint_map["el"]] = cfg["el"]

            for hi_idx in right_hand_indices:
                lo = robot.data.soft_joint_pos_limits[0, hi_idx, 0].item()
                hi = robot.data.soft_joint_pos_limits[0, hi_idx, 1].item()
                name = joint_names[hi_idx]
                if "thumb_2" in name:
                    val = hi + (lo - hi) * closure_pct / 100.0
                elif abs(hi) > abs(lo):
                    val = lo + (hi - lo) * closure_pct / 100.0
                else:
                    val = hi + (lo - hi) * closure_pct / 100.0
                new_pos[0, hi_idx] = val

            set_joints_and_settle(env, robot, new_pos, n_steps=20)

            print(f"    Closure {closure_pct:3d}%:")
            for tname, tidx in tip_indices.items():
                if "tip" in tname:
                    pos = robot.data.body_pos_w[0, tidx].cpu().numpy()
                    print(f"      {tname:15s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    # ======================================================================
    # Final summary
    # ======================================================================
    print(f"\n{'=' * 80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

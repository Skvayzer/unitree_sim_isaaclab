#!/usr/bin/env python3
"""
Diagnostic: measure fingertip positions at various SR (shoulder_roll) values.

Purpose: determine the valid SR range for grasping by checking where
fingertips end up when the arm is set to the DESCEND_TO_GRASP position
(sp=0.0, sy=0.1, el=0.0) at each SR value.

CRITICAL: uses write_joint_state_to_sim() to FULLY reset robot state
between each SR test. A previous script only used env.reset() which
left residual joint velocities/positions from the prior config, producing
wrong measurements at extreme SR values.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/diagnostic_sr_sweep.py --headless
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="SR diagnostic sweep for fingertip positions")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.arm_trajectory import ARM_JOINT_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CFG = BlockStackConfig()

# SR values to test (from most negative / far from body to least negative / close)
SR_VALUES = [-0.70, -0.60, -0.50, -0.40, -0.35, -0.30, -0.25, -0.20]

# Grasp position: sp=0.0, sy=0.1, el=0.0, wrist all zero
SP = 0.0

# Number of sim steps to settle after writing joint state
SETTLE_STEPS = 100

# Block half-size for reachability check
BLOCK_HALF = CFG.block_size / 2.0  # 0.02m

# Fingertip triangle must be within this radius of block center to count as reachable
REACH_THRESHOLD = 0.03  # 3cm


def main():
    # ----- Create env with 1 environment -----
    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # ----- Build joint index maps -----
    arm_joint_map = {}
    hand_indices = []
    for i, name in enumerate(joint_names):
        if "right_shoulder_pitch" in name:
            arm_joint_map["sp"] = i
        elif "right_shoulder_roll" in name:
            arm_joint_map["sr"] = i
        elif "right_shoulder_yaw" in name:
            arm_joint_map["sy"] = i
        elif "right_elbow" in name:
            arm_joint_map["el"] = i
        elif "right_wrist_roll" in name:
            arm_joint_map["wr"] = i
        elif "right_wrist_pitch" in name:
            arm_joint_map["wp"] = i
        elif "right_wrist_yaw" in name:
            arm_joint_map["wy"] = i
        if "right_hand" in name:
            hand_indices.append(i)

    # ----- Build fingertip body indices -----
    tip_indices = {}
    for i, name in enumerate(body_names):
        if "right_hand_thumb_2_link" in name:
            tip_indices["thumb"] = i
        elif "right_hand_index_1_link" in name:
            tip_indices["index"] = i
        elif "right_hand_middle_1_link" in name:
            tip_indices["middle"] = i

    print(f"Arm joint indices: {arm_joint_map}")
    print(f"Fingertip body indices: {tip_indices}")
    print(f"Right hand joint indices ({len(hand_indices)}): {hand_indices}")
    print()

    # Base arm config from BlockStackConfig (grasp position)
    base_grasp = torch.tensor(
        [CFG.arm_grasp_joints[name] for name in ARM_JOINT_NAMES],
        device=env.device, dtype=torch.float32,
    )

    # Config-level arm joint indices (into the full joint array)
    arm_idx = torch.tensor(CFG.right_arm_indices, device=env.device)

    # Env origin for converting world coords to local
    env_origin = env.scene.env_origins[0]  # [3] on device

    # ----- SR -> expected block y mapping -----
    # The linear approximation: y = (sr - sr_ref) / slope + y_ref
    # But we know this is approximate. The diagnostic DATA itself is what
    # parameterized_trajectory.py uses. Here we just report raw positions.

    print("=" * 110)
    print("SR DIAGNOSTIC SWEEP")
    print(f"Arm config: sp={SP:.1f}, sy=0.1, el=0.0, wrist=0.0 (DESCEND_TO_GRASP position)")
    print(f"Fingers: OPEN (default position)")
    print(f"Settle steps: {SETTLE_STEPS}")
    print(f"Full robot reset via write_joint_state_to_sim() between each SR value")
    print("=" * 110)
    print()

    header = (
        f"{'SR':>6} | {'thumb_xyz':>28} | {'index_xyz':>28} | {'middle_xyz':>28} "
        f"| {'center_y':>8} | {'center_z':>8} | {'spread':>6} | reachable?"
    )
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    results = []

    for sr in SR_VALUES:
        # ===== FULL ROBOT RESET =====
        # Build complete joint position vector from defaults
        full_pos = robot.data.default_joint_pos.clone()  # [1, num_joints]
        full_vel = torch.zeros_like(full_pos)

        # Set arm joints: sp, sr, sy, el, wrist all zero
        full_pos[0, arm_joint_map["sp"]] = SP
        full_pos[0, arm_joint_map["sr"]] = sr
        full_pos[0, arm_joint_map["sy"]] = 0.1
        full_pos[0, arm_joint_map["el"]] = 0.0
        full_pos[0, arm_joint_map["wr"]] = 0.0
        full_pos[0, arm_joint_map["wp"]] = 0.0
        full_pos[0, arm_joint_map["wy"]] = 0.0

        # Fingers OPEN (use default positions from init_state — already in default_joint_pos)
        # No modification needed; defaults have fingers open.

        # Write full robot state to sim (position + zero velocity)
        robot.write_joint_state_to_sim(full_pos, full_vel)

        # Also set joint position targets so PD controller holds this config
        robot.data.joint_pos_target[:] = full_pos

        # Build arm-only target for repeated application
        arm_target = base_grasp.clone().unsqueeze(0)  # [1, 7]
        arm_target[0, 1] = sr  # shoulder_roll

        # ===== SETTLE: step sim with arm target held =====
        for step_i in range(SETTLE_STEPS):
            # Keep driving arm to target
            robot.data.joint_pos_target[:, arm_idx] = arm_target
            robot.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)

        # ===== READ FINGERTIP POSITIONS =====
        body_pos = robot.data.body_pos_w  # [1, num_bodies, 3]

        thumb_w = body_pos[0, tip_indices["thumb"]].cpu().numpy()
        index_w = body_pos[0, tip_indices["index"]].cpu().numpy()
        middle_w = body_pos[0, tip_indices["middle"]].cpu().numpy()

        # Center of fingertip triangle (world coords)
        center_x = (thumb_w[0] + index_w[0] + middle_w[0]) / 3.0
        center_y = (thumb_w[1] + index_w[1] + middle_w[1]) / 3.0
        center_z = (thumb_w[2] + index_w[2] + middle_w[2]) / 3.0

        # Fingertip spread (max pairwise distance in y)
        ys = [thumb_w[1], index_w[1], middle_w[1]]
        spread_y = max(ys) - min(ys)

        # Convert center to local coords (subtract env origin)
        origin = env_origin.cpu().numpy()
        local_center_y = center_y - origin[1]

        # Expected block y for this SR: the block would be placed so its center
        # aligns with the fingertip center. We report what y-position this SR
        # can reach.
        # From parameterized_trajectory.py: fingertip offset ~0.014m
        # block_y = center_y_local + 0.014
        fingertip_offset = 0.0144
        block_y_for_sr = local_center_y + fingertip_offset

        # Check if fingertips are at a reasonable height for grasping
        # The block sits at z=0.819 (table_height + half_block + 0.002)
        # Fingertips should be near block height
        z_offset = abs(center_z - CFG.block_initial_z)
        z_ok = z_offset < 0.05  # within 5cm of block height

        # Check actual SR reached
        actual_sr = robot.data.joint_pos[0, arm_joint_map["sr"]].item()
        sr_err = abs(actual_sr - sr)

        # A grasp is reachable if:
        # 1. SR was actually achieved (no joint limit violation)
        # 2. Fingertips are at reasonable z height
        # 3. Spread is reasonable (fingers didn't collide/freeze)
        reachable = sr_err < 0.05 and z_ok and spread_y > 0.01

        note = "YES" if reachable else "NO"
        if sr_err > 0.05:
            note += f" (sr_err={sr_err:.3f})"
        if not z_ok:
            note += f" (z_off={z_offset*100:.1f}cm)"
        if spread_y < 0.01:
            note += " (fingers_frozen)"

        print(
            f"{sr:+.2f}  | "
            f"({thumb_w[0]:.3f},{thumb_w[1]:+.4f},{thumb_w[2]:.3f}) | "
            f"({index_w[0]:.3f},{index_w[1]:+.4f},{index_w[2]:.3f}) | "
            f"({middle_w[0]:.3f},{middle_w[1]:+.4f},{middle_w[2]:.3f}) | "
            f"{center_y:+.4f} | {center_z:.4f} | {spread_y:.4f} | "
            f"{note}"
        )
        sys.stdout.flush()

        results.append({
            "sr": sr,
            "actual_sr": actual_sr,
            "center_y": center_y,
            "center_y_local": local_center_y,
            "center_z": center_z,
            "block_y": block_y_for_sr,
            "spread_y": spread_y,
            "z_offset": z_offset,
            "reachable": reachable,
        })

    # ===== SUMMARY TABLE =====
    print()
    print("=" * 90)
    print("SUMMARY: SR -> block y-position mapping")
    print("=" * 90)
    print(f"{'SR':>6} | {'actual_sr':>9} | {'center_y':>9} | {'block_y':>8} | {'spread':>6} | {'z_off_cm':>8} | reachable?")
    print("-" * 75)

    for r in results:
        print(
            f"{r['sr']:+.2f}  | {r['actual_sr']:+.4f}   | {r['center_y_local']:+.4f}   | "
            f"{r['block_y']:+.4f} | {r['spread_y']:.4f} | {r['z_offset']*100:6.2f}   | "
            f"{'YES' if r['reachable'] else 'NO'}"
        )

    # Valid range
    valid = [r for r in results if r["reachable"]]
    if valid:
        sr_min = min(r["sr"] for r in valid)
        sr_max = max(r["sr"] for r in valid)
        y_min = min(r["block_y"] for r in valid)
        y_max = max(r["block_y"] for r in valid)
        print()
        print(f"Valid SR range:    [{sr_min:+.2f}, {sr_max:+.2f}]")
        print(f"Valid block_y range: [{y_min:+.4f}, {y_max:+.4f}]")
        print(f"Block y span:      {(y_max - y_min)*100:.1f} cm")
        print()
        print("Diagnostic data for parameterized_trajectory.py (center_y, SR) pairs:")
        for r in valid:
            print(f"  center_y={r['center_y_local']:+.4f}  sr={r['sr']:+.2f}")
    else:
        print()
        print("WARNING: No valid SR values found!")

    print()
    print("DONE")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

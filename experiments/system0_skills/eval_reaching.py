"""Evaluate CraftNet System 1+2 reaching with IK prior + cameras + depth in IsaacLab.

Full pipeline:
  IsaacLab sim -> cameras (RGB+depth) -> eval_reaching -> System 1 (DiT+IK) -> actions -> sim
  System 1 <-> System 2 (Qwen3-VL backbone, ~1Hz via ZMQ)

Uses the Dex3-1 environment (G1 + Dex3 hand) with ALL 4 cameras + depth.
Previous version used Dex1 env which only had 2 head cameras (wrist cameras were black).

Start servers first (see README below), then run this script.
"""

import argparse
import json
import io
import os
import sys
import time

# -- IsaacLab bootstrap --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--system1_host", default="127.0.0.1")
parser.add_argument("--system1_port", type=int, default=5556)
parser.add_argument("--steps_per_pos", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

# Add both experiments/system0_skills/ and repo root to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, "../.."))
sys.path.insert(0, _this_dir)
sys.path.insert(0, _repo_root)

# ==============================================================
# Config
# ==============================================================

TEST_POSITIONS = [
    (0.0, 0.0, "center"),
    (0.0, -0.03, "y-3cm"),
    (0.0, +0.03, "y+3cm"),
    (0.0, -0.06, "y-6cm"),
    (0.0, +0.06, "y+6cm"),
    (-0.03, 0.0, "x-3cm"),
    (+0.03, 0.0, "x+3cm"),
    (-0.03, -0.03, "diag-NW"),
    (+0.03, +0.03, "diag-SE"),
]

BLOCK_POS_FILE = "/tmp/sim_block_positions.json"
REACH_THRESHOLD_CM = 8.0

# CraftNet 28D state/action packing:
# [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
#
# URDF joint indices for G1 Dex3-1 robot.
# Arm indices are the same as Dex1 (arm structure unchanged).
# Hand indices follow the DDS protocol ordering from dex3_state.py:
#   get_robot_girl_joint_names() ordering:
#     L: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
#     R: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
#   gripper_joint_indices = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
#   Left hand (first 7):  [31, 37, 41, 30, 36, 29, 35]
#   Right hand (last 7):  [34, 40, 42, 33, 39, 32, 38]

LEFT_ARM_URDF = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_URDF = [12, 16, 20, 22, 24, 26, 28]
LEFT_HAND_URDF = [31, 37, 41, 30, 36, 29, 35]
RIGHT_HAND_URDF = [34, 40, 42, 33, 39, 32, 38]

ROBOT_BASE_Z = 0.79  # approximate robot torso height in sim

# Camera scene names in the Dex3 env -> CraftNet policy image keys
CAMERA_MAP = {
    "front_camera_left": "cam_left_high",
    "front_camera_right": "cam_right_high",
    "left_wrist_camera": "cam_left_wrist",
    "right_wrist_camera": "cam_right_wrist",
}


# ==============================================================
# Helpers
# ==============================================================

def compress_jpeg(img_np, quality=85):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def publish_block_pos(block_world, env_origin):
    """Write robot-relative block position for IK prior."""
    rel = (block_world - env_origin).cpu().numpy()
    rel_robot = [float(rel[0]), float(rel[1]), float(rel[2] - ROBOT_BASE_Z)]
    data = {
        "block_0": {"rel": rel_robot},
        "red_block": {"rel": rel_robot},
        "yellow_block": {"rel": rel_robot},
        "green_block": {"rel": rel_robot},
    }
    with open(BLOCK_POS_FILE, "w") as f:
        json.dump(data, f)


def get_hand_distance(robot, block_world):
    """Distance from right hand base to block (meters)."""
    for i, name in enumerate(robot.data.body_names):
        if "right_hand_base" in name.lower():
            return torch.norm(robot.data.body_pos_w[0, i] - block_world).item()
    for i, name in enumerate(robot.data.body_names):
        if "right_wrist_yaw" in name.lower() or "right_wrist_roll" in name.lower():
            return torch.norm(robot.data.body_pos_w[0, i] - block_world).item()
    return float("inf")


def build_28d_state(robot):
    """Build CraftNet-compatible 28D state: [L_arm(7), R_arm(7), L_hand(7), R_hand(7)]."""
    jp = robot.data.joint_pos[0]
    state = torch.zeros(28, device=jp.device)
    for i, idx in enumerate(LEFT_ARM_URDF):
        if idx < jp.shape[0]:
            state[i] = jp[idx]
    for i, idx in enumerate(RIGHT_ARM_URDF):
        if idx < jp.shape[0]:
            state[7 + i] = jp[idx]
    for i, idx in enumerate(LEFT_HAND_URDF):
        if idx < jp.shape[0]:
            state[14 + i] = jp[idx]
    for i, idx in enumerate(RIGHT_HAND_URDF):
        if idx < jp.shape[0]:
            state[21 + i] = jp[idx]
    return state.cpu().numpy().astype(np.float32)


def build_43d_action(action_28d, robot, device):
    """Convert CraftNet 28D to Dex3 env 43D action (pass through env.step).

    Uncontrolled joints hold current position. The Dex3 env action manager
    applies all 43 joint targets via PD control.
    """
    action_43d = robot.data.joint_pos[0].clone()  # hold current
    a28 = torch.tensor(action_28d, device=device, dtype=torch.float32)
    for i, idx in enumerate(LEFT_ARM_URDF):
        if idx < 43: action_43d[idx] = a28[i]
    for i, idx in enumerate(RIGHT_ARM_URDF):
        if idx < 43: action_43d[idx] = a28[7 + i]
    for i, idx in enumerate(LEFT_HAND_URDF):
        if idx < 43: action_43d[idx] = a28[14 + i]
    for i, idx in enumerate(RIGHT_HAND_URDF):
        if idx < 43: action_43d[idx] = a28[21 + i]
    return action_43d.unsqueeze(0)  # [1, 43]


def connect_system1(host, port):
    import zmq, msgpack
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 10000)
    sock.setsockopt(zmq.SNDTIMEO, 10000)
    sock.connect(f"tcp://{host}:{port}")
    try:
        sock.send(msgpack.packb({"type": "ping"}, use_bin_type=True))
        resp = msgpack.unpackb(sock.recv(), raw=False)
        if resp.get("status") == "ok":
            print(f"[Eval] System 1 connected at {host}:{port}")
            return sock, ctx
    except Exception as e:
        print(f"[Eval] System 1 ping failed: {e}")
    return None, None


def send_predict(sock, images_jpeg, state_28d, task="pick up the block"):
    import msgpack
    req = {"type": "predict", "images": images_jpeg,
           "state": state_28d.tolist(), "task": task}
    sock.send(msgpack.packb(req, use_bin_type=True))
    resp = msgpack.unpackb(sock.recv(), raw=False)
    action = resp.get("action")
    return np.array(action, dtype=np.float32) if action is not None else None


# ==============================================================
# Env creation using Dex3 config (ALL cameras + depth)
# ==============================================================

def create_env():
    """Create the Dex3 block stacking environment with depth enabled on all cameras."""
    # The tasks/__init__.py auto-imports ALL task packages, including
    # pick_place_cylinder_g1_29dof_dex1 which imports pink->pinocchio.
    # pinocchio requires CXXABI_1.3.15 which isn't available in unitree_sim_env.
    # Fix: mock the 'pink' module before importing tasks, so the import chain
    # doesn't crash.
    import types
    pink_mock = types.ModuleType("pink")
    pink_mock.tasks = types.ModuleType("pink.tasks")
    pink_mock.tasks.FrameTask = type("FrameTask", (), {})
    sys.modules["pink"] = pink_mock
    sys.modules["pink.tasks"] = pink_mock.tasks
    sys.modules["pink.configuration"] = types.ModuleType("pink.configuration")

    # Now we can safely import tasks.common_config
    import tasks.common_config.camera_configs as cam_cfg_module
    cam_cfg_module._default_data_types = ["rgb", "distance_to_camera"]

    from tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3.stack_rgyblock_g1_29dof_dex3_joint_env_cfg import (
        StackRgyBlockG129DEX3BaseFixEnvCfg,
    )

    env_cfg = StackRgyBlockG129DEX3BaseFixEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0
    env_cfg.episode_length_s = 999.0
    return ManagerBasedRLEnv(cfg=env_cfg)


# ==============================================================
# Debug: save camera frames to verify no black images
# ==============================================================

def save_debug_frames(env):
    """Save one frame from each camera to /tmp/debug/ for verification."""
    debug_dir = "/tmp/debug"
    os.makedirs(debug_dir, exist_ok=True)

    all_ok = True
    for scene_cam_name in CAMERA_MAP:
        try:
            cam = env.scene[scene_cam_name]
        except KeyError:
            print(f"  [DEBUG] Camera '{scene_cam_name}' NOT FOUND in scene -- FATAL")
            all_ok = False
            continue

        cam.update(dt=0.1)

        # RGB check
        rgb = cam.data.output.get("rgb")
        if rgb is not None:
            img = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(f"{debug_dir}/{scene_cam_name}_rgb.png")
            mean_val = img.mean()
            if mean_val < 1.0:
                print(f"  [DEBUG] {scene_cam_name}: RGB is ALL BLACK (mean={mean_val:.2f}) -- FATAL")
                all_ok = False
            else:
                print(f"  [DEBUG] {scene_cam_name}: RGB OK shape={img.shape} mean={mean_val:.1f}")
        else:
            print(f"  [DEBUG] {scene_cam_name}: NO RGB output -- FATAL")
            all_ok = False

        # Depth check
        depth = cam.data.output.get("distance_to_camera")
        if depth is not None:
            d = depth[0].cpu().numpy()
            if d.ndim == 3:
                d = d[:, :, 0]
            d_norm = (np.clip(d, 0, 2) / 2 * 255).astype(np.uint8)
            Image.fromarray(d_norm).save(f"{debug_dir}/{scene_cam_name}_depth.png")
            print(f"  [DEBUG] {scene_cam_name}: depth OK shape={d.shape} range=[{d.min():.3f}, {d.max():.3f}]")
        else:
            print(f"  [DEBUG] {scene_cam_name}: NO depth output -- WARNING")

    return all_ok


# ==============================================================
# Main
# ==============================================================

def main():
    device = "cuda:0"
    import msgpack

    # Connect System 1
    sock, ctx = connect_system1(args.system1_host, args.system1_port)
    if sock is None:
        print("[Eval] ERROR: System 1 not available")
        simulation_app.close()
        return

    sock.send(msgpack.packb({"type": "reset"}, use_bin_type=True))
    msgpack.unpackb(sock.recv(), raw=False)

    # Create env (Dex3 with all cameras + depth)
    print("[Eval] Creating Dex3 env with ALL 4 cameras + depth...")
    env = create_env()
    robot = env.scene["robot"]
    red_block = env.scene["red_block"]
    env_origin = env.scene.env_origins[0]

    # List available cameras
    cams = {}
    for scene_name, policy_name in CAMERA_MAP.items():
        try:
            cams[scene_name] = env.scene[scene_name]
            print(f"[Eval] Camera '{scene_name}' -> '{policy_name}' OK")
        except KeyError:
            print(f"[Eval] Camera '{scene_name}' NOT available -- FATAL for Dex3 env")

    if len(cams) != 4:
        print(f"[Eval] FATAL: Only {len(cams)}/4 cameras available. Dex3 env requires all 4.")
        print("[Eval] Aborting to prevent evaluation with black camera streams.")
        env.close()
        simulation_app.close()
        return

    # Print robot joint info for verification
    joint_count = robot.data.joint_pos.shape[1]
    print(f"[Eval] Robot joints: {joint_count}")
    jp = robot.data.joint_pos[0]
    print(f"[Eval] Joint positions sample (first 10): {jp[:10].cpu().numpy().round(4)}")

    # Verify 28D state construction
    state_28d = build_28d_state(robot)
    print(f"[Eval] 28D state: L_arm={state_28d[:7].round(3)}, R_arm={state_28d[7:14].round(3)}")
    print(f"[Eval]            L_hand={state_28d[14:21].round(3)}, R_hand={state_28d[21:28].round(3)}")

    # Warm up sim so cameras produce valid frames
    for _ in range(20):
        env.step(torch.zeros(1, env.action_space.shape[-1], device=device))

    # Save debug camera frames and verify
    print("[Eval] Saving debug camera frames to /tmp/debug/...")
    cameras_ok = save_debug_frames(env)
    if not cameras_ok:
        print("[Eval] FATAL: One or more cameras produced black/invalid images.")
        print("[Eval] Check /tmp/debug/ for saved frames. Aborting.")
        env.close()
        sock.close()
        ctx.term()
        simulation_app.close()
        return

    # Default block pos (use red_block as target)
    base_block = red_block.data.root_pos_w[0].clone()
    base_local = (base_block - env_origin).cpu().numpy()
    print(f"[Eval] Red block default (local): {base_local.round(4)}")

    # Print robot body names for distance measurement debugging
    for i, name in enumerate(robot.data.body_names):
        if "right" in name.lower() and ("hand" in name.lower() or "wrist" in name.lower()):
            print(f"[Eval] Body {i}: {name}")

    # Print all joint names for mapping verification
    print("[Eval] Full joint map:")
    for i, name in enumerate(robot.data.joint_names):
        marker = ""
        if i in LEFT_ARM_URDF:
            pos = LEFT_ARM_URDF.index(i)
            marker = f" <- L_arm[{pos}]"
        elif i in RIGHT_ARM_URDF:
            pos = RIGHT_ARM_URDF.index(i)
            marker = f" <- R_arm[{pos}]"
        elif i in LEFT_HAND_URDF:
            pos = LEFT_HAND_URDF.index(i)
            marker = f" <- L_hand[{pos}]"
        elif i in RIGHT_HAND_URDF:
            pos = RIGHT_HAND_URDF.index(i)
            marker = f" <- R_hand[{pos}]"
        print(f"  [{i:2d}] {name}: {jp[i].item():.4f}{marker}")

    results = []

    for pos_idx, (dx, dy, label) in enumerate(TEST_POSITIONS):
        print(f"\n{'='*60}")
        print(f"  Position {pos_idx+1}/{len(TEST_POSITIONS)}: {label}")
        print(f"{'='*60}")

        # Teleport red block
        new_local = base_local.copy()
        new_local[0] += dx
        new_local[1] += dy
        new_world = torch.tensor(new_local, device=device, dtype=torch.float32) + env_origin

        state = red_block.data.root_state_w[0:1].clone()
        state[0, :3] = new_world
        state[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device, dtype=state.dtype)
        state[0, 7:] = 0
        red_block.write_root_state_to_sim(state, env_ids=torch.tensor([0], device=device))
        publish_block_pos(new_world, env_origin)

        # Warm up sim
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[-1], device=device))

        min_dist = float("inf")
        dists = []

        for step in range(args.steps_per_pos):
            # -- Collect camera images --
            images_jpeg = {}
            for scene_name, policy_name in CAMERA_MAP.items():
                if scene_name not in cams:
                    continue
                cam = cams[scene_name]
                cam.update(dt=0.1)

                # RGB
                rgb = cam.data.output.get("rgb")
                if rgb is not None:
                    img = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
                    images_jpeg[f"observation.images.{policy_name}"] = compress_jpeg(img)

                # Depth -> grayscale replicated to 3 channels
                depth = cam.data.output.get("distance_to_camera")
                if depth is not None:
                    d = depth[0].cpu().numpy()
                    if d.ndim == 3:
                        d = d[:, :, 0]
                    d = np.clip(d, 0, 2.0) / 2.0 * 255
                    d_uint8 = d.astype(np.uint8)
                    d_rgb = np.stack([d_uint8] * 3, axis=-1)
                    images_jpeg[f"observation.images.depth_{policy_name}"] = compress_jpeg(d_rgb)

            # Verify ALL 8 image keys are present (no black fallback!)
            all_keys = [
                "observation.images.cam_left_high", "observation.images.cam_right_high",
                "observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
                "observation.images.depth_cam_left_high", "observation.images.depth_cam_right_high",
                "observation.images.depth_cam_left_wrist", "observation.images.depth_cam_right_wrist",
            ]
            missing = [k for k in all_keys if k not in images_jpeg]
            if missing and step == 0:
                print(f"  [WARNING] Missing image keys: {missing}")
                print(f"  [WARNING] This should NOT happen with Dex3 env. Check camera config.")

            # -- Build 28D state --
            state_28d = build_28d_state(robot)

            # -- Send to System 1 --
            try:
                action = send_predict(sock, images_jpeg, state_28d)
                if action is not None and len(action) >= 28:
                    # Build 43D action for env.step (maps 28D CraftNet → 43 URDF joints)
                    action_43d = build_43d_action(action, robot, device)
                    env.step(action_43d)

                    if step == 0:
                        print(f"  Action R_arm: {action[7:14].round(3)}")
                        print(f"  State  R_arm: {state_28d[7:14].round(3)}")
                        print(f"  Action R_hand: {action[21:28].round(3)}")
                else:
                    env.step(torch.zeros(1, env.action_space.shape[-1], device=device))
            except Exception as e:
                if step % 50 == 0:
                    print(f"  [step={step}] Error: {e}")
                env.step(torch.zeros(1, env.action_space.shape[-1], device=device))

            # -- Measure --
            dist = get_hand_distance(robot, new_world)
            dist_cm = dist * 100
            dists.append(dist_cm)
            min_dist = min(min_dist, dist_cm)

            if step % 50 == 0:
                print(f"  [step={step}] dist={dist_cm:.1f}cm, min={min_dist:.1f}cm")

        reached = min_dist < REACH_THRESHOLD_CM
        results.append({"label": label, "reached": reached,
                        "min_cm": round(min_dist, 1), "final_cm": round(dists[-1], 1)})
        print(f"  -> {'REACHED' if reached else 'FAILED'} (min={min_dist:.1f}cm)")

    # -- Summary --
    env.close()
    sock.close()
    ctx.term()

    print(f"\n{'='*60}")
    print(f"  REACHING RESULTS (Dex3 env, System2 + System1 + IK Prior + 4 Cameras + Depth)")
    print(f"{'='*60}")
    n = sum(1 for r in results if r["reached"])
    print(f"  Reached: {n}/{len(results)} (<{REACH_THRESHOLD_CM}cm)")
    print(f"\n  {'Pos':<12} {'OK':<6} {'Min':<8} {'Final'}")
    print(f"  {'-'*36}")
    for r in results:
        print(f"  {r['label']:<12} {'YES' if r['reached'] else 'no':<6} {r['min_cm']:<8.1f} {r['final_cm']:.1f}")

    out = os.path.expanduser("~/unitree_sim_isaaclab/logs/reaching_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": results, "threshold_cm": REACH_THRESHOLD_CM,
                    "robot": "G1_Dex3-1", "cameras": 4, "depth": True}, f, indent=2)
    print(f"\n  Saved: {out}")
    simulation_app.close()


if __name__ == "__main__":
    main()

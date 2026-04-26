#!/usr/bin/env python3
"""Test CraftNet reaching: place blocks at various positions, measure EE-to-block distance.

Measures actual end-effector to target block distance using sim-published EE positions.
Each test restarts the eval client with the correct task text so System 2 gets the
right language instruction ("pick up the red/yellow/green block").

Pipeline must be running:
  T1: sim_main.py (Dex3 with cameras + block teleport + EE publisher)
  T2: image_compositor.py
  T3: System 1 server (port 5556)
  T4: System 2 server (remote, port 5557)

The eval client (eval_g1_sim.py) is started/stopped per test by this script.

Usage:
  python experiments/system0_skills/test_reaching_positions.py
"""

import json
import os
import subprocess
import signal
import sys
import time
import numpy as np

LAYOUT_FILE = "/tmp/sim_block_layout.json"
POSITIONS_FILE = "/tmp/sim_block_positions.json"
ARM_DEBUG_FILE = "/tmp/arm_debug.txt"

TABLE_Z = 0.82

TEST_LAYOUTS = [
    {
        "name": "1. Red LEFT, Yellow CENTER, Green RIGHT",
        "task": "pick up the red block",
        "target": "red_block",
        "blocks": {
            "red_block":    [-0.15, -0.25, TABLE_Z],
            "yellow_block": [-0.05, -0.40, TABLE_Z],
            "green_block":  [ 0.05, -0.55, TABLE_Z],
        }
    },
    {
        "name": "2. Red CLOSE, Yellow MID, Green FAR",
        "task": "pick up the red block",
        "target": "red_block",
        "blocks": {
            "red_block":    [-0.15, -0.35, TABLE_Z],
            "yellow_block": [-0.05, -0.35, TABLE_Z],
            "green_block":  [ 0.05, -0.35, TABLE_Z],
        }
    },
    {
        "name": "3. Target YELLOW (scattered)",
        "task": "pick up the yellow block",
        "target": "yellow_block",
        "blocks": {
            "red_block":    [-0.10, -0.25, TABLE_Z],
            "yellow_block": [ 0.00, -0.45, TABLE_Z],
            "green_block":  [-0.10, -0.55, TABLE_Z],
        }
    },
    {
        "name": "4. Target GREEN (far position)",
        "task": "pick up the green block",
        "target": "green_block",
        "blocks": {
            "red_block":    [-0.05, -0.30, TABLE_Z],
            "yellow_block": [-0.05, -0.40, TABLE_Z],
            "green_block":  [-0.05, -0.55, TABLE_Z],
        }
    },
    {
        "name": "5. DIAGONAL layout (target red)",
        "task": "pick up the red block",
        "target": "red_block",
        "blocks": {
            "red_block":    [-0.15, -0.30, TABLE_Z],
            "yellow_block": [-0.05, -0.40, TABLE_Z],
            "green_block":  [ 0.05, -0.50, TABLE_Z],
        }
    },
    {
        "name": "6. CLOSE together (target red)",
        "task": "pick up the red block",
        "target": "red_block",
        "blocks": {
            "red_block":    [-0.10, -0.35, TABLE_Z],
            "yellow_block": [-0.05, -0.45, TABLE_Z],
            "green_block":  [ 0.00, -0.35, TABLE_Z],
        }
    },
]


def read_positions():
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except:
        return None


def compute_ee_block_distance(positions, target_name, hand="left"):
    """Compute distance between hand EE and target block (both in robot-relative coords)."""
    if positions is None:
        return None
    ee_key = f"{hand}_ee"
    ee = positions.get(ee_key, {}).get("rel")
    block = positions.get(target_name, {}).get("rel")
    if ee is None or block is None:
        return None
    return np.linalg.norm(np.array(ee) - np.array(block))


def start_eval_client(task_text):
    """Start eval_g1_sim.py with the given task text. Returns process."""
    env = os.environ.copy()
    env["UNITREE_DDS_IFACE"] = "lo"
    env["UNITREE_DDS_DOMAIN_ID"] = "1"
    env["CYCLONEDDS_URI"] = "file:///home/cosmos/cyclonedds_local.xml"

    cmd = [
        "conda", "run", "-n", "unitree_lerobot",
        "python", "unitree_lerobot/eval_robot/eval_g1_sim.py",
        "--repo_id", "unitreerobotics/G1_Dex3_BlockStacking_Dataset",
        "--ee", "dex3", "--sim", "true",
        "--remote_policy_host", "127.0.0.1",
        "--remote_policy_port", "5556",
        "--task_override", f"'{task_text}'",
    ]
    proc = subprocess.Popen(
        cmd, cwd="/home/cosmos/unitree_IL_lerobot",
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, preexec_fn=os.setsid,
    )
    # Send 's' to start
    try:
        proc.stdin.write(b"s\n")
        proc.stdin.flush()
    except:
        pass
    return proc


def stop_eval_client(proc):
    """Stop eval client process group."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except:
        pass
    try:
        proc.wait(timeout=5)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except:
            pass


def main():
    SECONDS_PER_TEST = 45
    WARMUP_SECONDS = 15  # wait for eval client to connect and start producing actions

    print("=" * 70)
    print("  CraftNet Reaching Test — EE-to-Block Distance Measurement")
    print("  Each test restarts eval client with correct task text")
    print("=" * 70)

    # Verify sim is publishing
    pos = read_positions()
    if pos is None:
        print("[ERROR] /tmp/sim_block_positions.json not found")
        return
    if "left_ee" not in pos and "right_ee" not in pos:
        print("[WARNING] EE positions not in sim publisher. Restart sim with updated sim_main.py.")
        print("  Available keys:", list(pos.keys()))

    results = []

    for i, test in enumerate(TEST_LAYOUTS):
        print(f"\n{'=' * 70}")
        print(f"  TEST {i+1}/{len(TEST_LAYOUTS)}: {test['name']}")
        print(f"  Task to System 2: \"{test['task']}\"")
        print(f"  Target block: {test['target']}")
        print(f"{'=' * 70}")

        # 1. Teleport blocks
        with open(LAYOUT_FILE, "w") as f:
            json.dump(test["blocks"], f)
        time.sleep(3)  # wait for sim to process

        # 2. Start eval client with correct task text
        print(f"  Starting eval client with task: \"{test['task']}\"")
        eval_proc = start_eval_client(test["task"])
        print(f"  Warming up ({WARMUP_SECONDS}s)...")
        time.sleep(WARMUP_SECONDS)

        # 3. Read block positions
        pos = read_positions()
        if pos:
            for bname in ["red_block", "yellow_block", "green_block"]:
                bp = pos.get(bname, {}).get("rel", [0, 0, 0])
                print(f"    {bname:15s} rel=({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")
            for ee_name in ["left_ee", "right_ee"]:
                ep = pos.get(ee_name, {}).get("rel")
                if ep:
                    print(f"    {ee_name:15s} rel=({ep[0]:+.3f}, {ep[1]:+.3f}, {ep[2]:+.3f})")

        # 4. Monitor EE-to-block distance
        print(f"\n  Measuring for {SECONDS_PER_TEST}s...")
        min_left_dist = float("inf")
        min_right_dist = float("inf")
        dist_history = []

        for sec in range(SECONDS_PER_TEST):
            time.sleep(1)
            pos = read_positions()
            if pos is None:
                continue

            left_d = compute_ee_block_distance(pos, test["target"], "left")
            right_d = compute_ee_block_distance(pos, test["target"], "right")

            if left_d is not None:
                min_left_dist = min(min_left_dist, left_d)
            if right_d is not None:
                min_right_dist = min(min_right_dist, right_d)

            best_d = min(left_d or 999, right_d or 999)
            dist_history.append(best_d)

            if sec % 5 == 4:
                l_str = f"{left_d*100:.1f}cm" if left_d else "N/A"
                r_str = f"{right_d*100:.1f}cm" if right_d else "N/A"
                print(f"    [t={sec+1:2d}s] L_ee→block: {l_str}  R_ee→block: {r_str}"
                      f"  min_L={min_left_dist*100:.1f}cm  min_R={min_right_dist*100:.1f}cm")

        # 5. Stop eval client
        stop_eval_client(eval_proc)
        time.sleep(2)

        # 6. Record result
        best_hand = "left" if min_left_dist < min_right_dist else "right"
        best_dist = min(min_left_dist, min_right_dist)
        reached = best_dist < 0.08  # 8cm threshold

        result = {
            "test": test["name"],
            "task": test["task"],
            "target": test["target"],
            "min_left_ee_cm": round(min_left_dist * 100, 1) if min_left_dist < 900 else None,
            "min_right_ee_cm": round(min_right_dist * 100, 1) if min_right_dist < 900 else None,
            "best_hand": best_hand,
            "best_dist_cm": round(best_dist * 100, 1) if best_dist < 900 else None,
            "reached": reached,
        }
        results.append(result)

        status = "REACHED" if reached else "NOT YET"
        print(f"\n  Result: {best_hand} hand closest at {best_dist*100:.1f}cm — {status}")

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"  REACHING DISTANCE RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  {'Test':<40} {'Task':<25} {'L_ee':<10} {'R_ee':<10} {'Best':<10} {'OK?'}")
    print(f"  {'-' * 100}")
    for r in results:
        l = f"{r['min_left_ee_cm']:.1f}" if r['min_left_ee_cm'] else "N/A"
        ri = f"{r['min_right_ee_cm']:.1f}" if r['min_right_ee_cm'] else "N/A"
        b = f"{r['best_dist_cm']:.1f}" if r['best_dist_cm'] else "N/A"
        ok = "YES" if r['reached'] else "no"
        print(f"  {r['test']:<40} {r['task']:<25} {l:<10} {ri:<10} {b:<10} {ok}")

    n_reached = sum(1 for r in results if r['reached'])
    print(f"\n  Reached: {n_reached}/{len(results)} (<8cm)")

    out = os.path.expanduser("~/unitree_sim_isaaclab/logs/reaching_distance_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tests": results, "seconds_per_test": SECONDS_PER_TEST}, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()

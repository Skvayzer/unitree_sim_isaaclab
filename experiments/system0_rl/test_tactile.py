#!/usr/bin/env python3
"""
Finger-by-finger tactile test for System 0 / Dex3 in Isaac Lab.

Tests each finger pad independently by closing one finger at a time
while a block object is present, then verifying correct sensor responses.

Usage:
    cd unitree_sim_isaaclab
    python experiments/system0_rl/test_tactile.py --enable_cameras
"""

import os
import sys
import argparse
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PROJECT_ROOT"] = project_root
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dex3 finger-by-finger tactile test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # need to see what's happening

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg

try:
    import tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3  # noqa
except ImportError:
    import tasks  # noqa

from experiments.system0_rl.env_cfg import System0TrainEnvCfg, SceneCfg
from tasks.common_config import G1RobotPresets

# ── All 8 links per hand in URDF order ──────────────────────────────────────
DEX3_LEFT_LINKS = [
    "left_hand_palm_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
]
DEX3_RIGHT_LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
]

# ── Finger test groups: joints to close per test, and expected sensor pads ──
# Each entry: (test_name, joint_indices_relative_to_hand, expected_link_names)
# Joint order in HAND_NAMES from train.py:
#   [0] left_hand_thumb_0_joint
#   [1] left_hand_thumb_1_joint
#   [2] left_hand_thumb_2_joint
#   [3] left_hand_middle_0_joint
#   [4] left_hand_middle_1_joint
#   [5] left_hand_index_0_joint
#   [6] left_hand_index_1_joint
#   [7..13] same for right hand

HAND_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

FINGER_TESTS = [
    # (label, hand_joint_indices, expected_sensor_links)
    ("LEFT THUMB",  [0, 1, 2],    ["left_hand_thumb_0_link",  "left_hand_thumb_1_link",  "left_hand_thumb_2_link"]),
    ("LEFT MIDDLE", [3, 4],       ["left_hand_middle_0_link", "left_hand_middle_1_link"]),
    ("LEFT INDEX",  [5, 6],       ["left_hand_index_0_link",  "left_hand_index_1_link"]),
    ("RIGHT THUMB", [7, 8, 9],    ["right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link"]),
    ("RIGHT MIDDLE",[10, 11],     ["right_hand_middle_0_link","right_hand_middle_1_link"]),
    ("RIGHT INDEX", [12, 13],     ["right_hand_index_0_link", "right_hand_index_1_link"]),
]

CONTACT_THRESHOLD = 0.1  # N — force above this counts as contact


def build_per_link_sensor_cfg(link_names: list[str]) -> ContactSensorCfg:
    """Single ContactSensorCfg covering all given links via regex."""
    pattern = "|".join(link_names)
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/({pattern})",
        history_length=2,
        track_air_time=False,
        debug_vis=False,
        # Filter: only report forces FROM objects (blocks), not self-contacts
        filter_prim_paths_expr=["/World/envs/env_.*/.*block.*"],
    )


def make_env():
    """Create env with per-link contact sensors for all hand pads."""
    cfg = System0TrainEnvCfg()
    cfg.scene.num_envs = 1

    # Replace the broad fingertip sensor with full per-link coverage
    # (keeping same name for compatibility, but covering all links + object filter)
    all_hand_links = DEX3_LEFT_LINKS + DEX3_RIGHT_LINKS
    link_regex = "(" + "|".join(all_hand_links) + ")"
    cfg.scene.fingertip_contacts = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{link_regex}",
        history_length=2,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/.*(block|Block).*"],
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    return env


def get_contact_per_link(env, body_names_ordered: list[str]) -> dict[str, float]:
    """Read contact force magnitude per named link."""
    sensor = env.scene["fingertip_contacts"]
    forces = sensor.data.net_forces_w  # (1, n_bodies, 3)
    all_body_names = list(sensor.body_names)

    result = {}
    for name in body_names_ordered:
        if name in all_body_names:
            idx = all_body_names.index(name)
            f_mag = forces[0, idx, :].norm().item()
            result[name] = f_mag
        else:
            result[name] = -1.0  # not found
    return result


def close_finger(env, sim_hand_indices, hand_joint_local_indices, close_frac=0.7):
    """Close specific finger joints (by local index into HAND_NAMES)."""
    robot = env.scene["robot"]
    action = robot.data.default_joint_pos[0].clone()  # start from default

    for local_i in hand_joint_local_indices:
        sim_i = sim_hand_indices[local_i]
        lo = robot.data.soft_joint_pos_limits[0, sim_i, 0].item()
        hi = robot.data.soft_joint_pos_limits[0, sim_i, 1].item()
        # close = move toward the limit that has larger absolute value
        target = hi * close_frac if abs(hi) > abs(lo) else lo * close_frac
        action[sim_i] = target

    full_action = action.unsqueeze(0)
    return full_action


def run_finger_test(env, sim_hand_indices, test_name, joint_indices, expected_links, n_steps=80):
    """Close one finger, wait, read contacts, report."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  Closing joints: {[HAND_NAMES[i] for i in joint_indices]}")
    print(f"  Expected contacts: {expected_links}")
    print(f"{'='*60}")

    all_links = DEX3_LEFT_LINKS + DEX3_RIGHT_LINKS

    # Step 1: open all fingers first (reset to default)
    robot = env.scene["robot"]
    open_action = robot.data.default_joint_pos[0:1].clone()
    for _ in range(30):
        env.step(open_action)
        simulation_app.update()

    # Step 2: close the target finger
    close_action = close_finger(env, sim_hand_indices, joint_indices, close_frac=0.8)
    contacts_over_time = {name: [] for name in all_links}

    for step in range(n_steps):
        env.step(close_action)
        if step % 5 == 0:
            simulation_app.update()

        forces = get_contact_per_link(env, all_links)
        for name, f in forces.items():
            contacts_over_time[name].append(f)

    # Step 3: summarize
    print(f"\n  Results (max contact force over {n_steps} steps):")
    any_active = False
    passed = True
    for name in all_links:
        vals = [v for v in contacts_over_time[name] if v >= 0]
        max_f = max(vals) if vals else 0.0
        active = max_f > CONTACT_THRESHOLD
        is_expected = name in expected_links
        tag = ""
        if active and is_expected:
            tag = " ✓ CORRECT"
        elif active and not is_expected:
            tag = " ⚠ UNEXPECTED (phantom?)"
            passed = False
        elif not active and is_expected:
            tag = " ✗ MISSING — sensor not firing!"
            passed = False

        if active or is_expected:
            print(f"    {name:40s}: {max_f:6.3f} N{tag}")
            any_active = True

    if not any_active:
        print("    !! No contacts detected at all !!")
        passed = False

    status = "PASS" if passed else "FAIL"
    print(f"\n  >> {test_name}: {status}")
    return passed


def check_sensor_bodies(env):
    """Print what bodies the contact sensor actually covers."""
    sensor = env.scene["fingertip_contacts"]
    body_names = list(sensor.body_names)
    print(f"\n{'='*60}")
    print(f"Contact sensor covers {len(body_names)} bodies:")
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")
    print(f"{'='*60}")

    # Check coverage vs expected
    all_expected = DEX3_LEFT_LINKS + DEX3_RIGHT_LINKS
    missing = [n for n in all_expected if n not in body_names]
    extra = [n for n in body_names if n not in all_expected]
    if missing:
        print(f"  MISSING expected links: {missing}")
    if extra:
        print(f"  EXTRA unexpected links: {extra}")
    if not missing and not extra:
        print("  Coverage: PERFECT — all 16 expected links present")


def build_hand_index_map(env):
    """Map HAND_NAMES to sim joint indices."""
    robot = env.scene["robot"]
    joint_names = robot.data.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    sim_hand = []
    for name in HAND_NAMES:
        idx = name_to_idx.get(name)
        if idx is None:
            print(f"  WARNING: hand joint '{name}' not found in sim!")
            sim_hand.append(0)
        else:
            sim_hand.append(idx)
    print(f"\nHand joint sim indices: {sim_hand}")
    return sim_hand


def main():
    print("\n" + "="*60)
    print("Dex3 Finger-by-Finger Tactile Verification Test")
    print("="*60)

    env = make_env()
    env.reset()

    simulation_app.update()
    time.sleep(1.0)

    # 1. Report what the sensor actually covers
    check_sensor_bodies(env)

    # 2. Build joint index map
    sim_hand_indices = build_hand_index_map(env)

    # 3. Baseline: no action, check for phantom contacts
    print(f"\n{'='*60}")
    print("BASELINE TEST: Open hand, no block contact expected")
    open_action = env.scene["robot"].data.default_joint_pos[0:1].clone()
    phantom_forces = {}
    for _ in range(40):
        env.step(open_action)
    for _ in range(20):
        env.step(open_action)
        forces = get_contact_per_link(env, DEX3_LEFT_LINKS + DEX3_RIGHT_LINKS)
        for name, f in forces.items():
            phantom_forces[name] = max(phantom_forces.get(name, 0.0), f)

    phantoms = {k: v for k, v in phantom_forces.items() if v > CONTACT_THRESHOLD}
    if phantoms:
        print(f"  ⚠ PHANTOM CONTACTS detected (self-collision filter may be needed):")
        for name, f in phantoms.items():
            print(f"    {name}: {f:.3f} N")
    else:
        print("  ✓ No phantom contacts in open-hand baseline")

    # 4. Run per-finger tests
    results = {}
    for test_name, joint_indices, expected_links in FINGER_TESTS:
        passed = run_finger_test(
            env, sim_hand_indices, test_name, joint_indices, expected_links
        )
        results[test_name] = passed

    # 5. Summary
    print(f"\n{'='*60}")
    print("TACTILE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_pass = False
    print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

    # 6. Sim-vs-real gap summary
    print(f"\n{'='*60}")
    print("SIM vs REAL ALIGNMENT CHECK")
    print(f"{'='*60}")
    sensor = env.scene["fingertip_contacts"]
    n_bodies = len(sensor.body_names)
    print(f"  Sim sensor bodies:    {n_bodies} (8 per hand × 2 = 16 expected)")
    print(f"  Real SDK output:      9 per hand × 2 = 18 scalar pressures")
    print(f"  Current policy obs:   18-dim (tactile_state.py: 6 tips × 3 xyz)")
    print(f"  Recommended mapping:  16 links → magnitude → 16-dim OR binarize")
    print(f"  Self-collision filter: {'ACTIVE' if '[filter]' in str(sensor.cfg) else 'CHECK cfg'}")
    print(f"\n  Alignment action items:")
    print(f"  1. Use scalar magnitudes (not xyz) for each pad link → 16-dim")
    print(f"  2. Add per-pad force gain DR: U(0.7, 1.3)")
    print(f"  3. Add per-pad additive noise: N(0, 0.1 N)")
    print(f"  4. Add binary thresholding option for robust sim2real")
    print(f"  5. Update train.py to use tactile_state.py (not raw reshape[:18])")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tactile sensor test for Dex3 in IsaacLab.

What this does:
  Runs the scripted controller to drive the arm toward the red block.
  Once the controller reaches GRASP phase (fingers closing on block),
  it prints contact force for every hand pad link every few steps.
  At the end, reports which pads registered contact and which did not.

Expected result: thumb_2, middle_1, index_1 tips (and their proximal links)
on the active hand should show nonzero forces during GRASP/LIFT.

Run:
    cd ~/robotics/unitree_sim_isaaclab
    conda activate unitree_sim_env
    python experiments/system0_rl/test_tactile.py
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

try:
    import tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3  # noqa
except ImportError:
    import tasks  # noqa

from experiments.system0_rl.env_cfg import System0TrainEnvCfg
from experiments.system0_rl.config import TrainConfig
from experiments.system0_rl.scripted_controller import ScriptedController, Phase
from tasks.common_observations.tactile_state import get_tactile_obs, DEX3_PAD_LINKS

HAND_NAMES = [
    "left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",  "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint",  "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

ARM_NAMES = [
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_roll_joint",      "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def build_index_maps(env):
    names = env.scene["robot"].data.joint_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    sim_arm  = [name_to_idx[n] for n in ARM_NAMES  if n in name_to_idx]
    sim_hand = [name_to_idx[n] for n in HAND_NAMES if n in name_to_idx]
    return sim_arm, sim_hand


def map_to_sim(coarse, env, sim_arm, sim_hand):
    full = env.scene["robot"].data.default_joint_pos[0:1].clone()
    full[0, sim_arm] = coarse[:14]
    full[0, sim_hand] = coarse[14:28]
    return full


def print_tactile(tactile_row: torch.Tensor, active_hand: str, step: int, phase: Phase):
    """Print contact forces for all 16 pads, highlighting the active hand."""
    print(f"\n  [step {step:4d}] phase={phase.name}  active_hand={active_hand}")
    print(f"  {'Pad link':<35} {'Force (N)':>10}  {'':>6}")
    for i, link in enumerate(DEX3_PAD_LINKS):
        f = tactile_row[i].item()
        bar = "█" * min(int(f * 10), 20) if f > 0.05 else ""
        marker = " ← CONTACT" if f > 0.1 else ""
        print(f"  {link:<35} {f:>10.3f}  {bar}{marker}")


def main():
    config = TrainConfig(num_envs=1, headless=getattr(args, "headless", False))

    env_cfg = System0TrainEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    simulation_app.update()

    sim_arm, sim_hand = build_index_maps(env)

    # Verify sensor coverage on startup
    sensor = env.scene["fingertip_contacts"]
    covered = list(sensor.body_names)
    missing = [l for l in DEX3_PAD_LINKS if l not in covered]
    print(f"\nContact sensor covers {len(covered)} bodies.")
    if missing:
        print(f"  !! MISSING links (fix ContactSensorCfg): {missing}")
    else:
        print(f"  All 16 pad links present. ✓")

    controller = ScriptedController(
        config, device=torch.device("cpu"),
        sim_arm_indices=sim_arm, sim_hand_indices=sim_hand, env_idx=0,
    )

    # Track max force seen per pad across the whole run
    max_forces = torch.zeros(16)
    phase_reached = set()
    PRINT_EVERY = 30  # print tactile every N steps in GRASP/LIFT

    print(f"\nRunning scripted controller for up to 2000 steps...")
    print(f"Will print tactile readings once arm reaches GRASP phase.\n")

    for step in range(2000):
        coarse, intent = controller.step(env)
        sim_action = map_to_sim(coarse, env, sim_arm, sim_hand)
        env.step(sim_action)

        if step % 5 == 0:
            simulation_app.update()

        phase = controller.phase
        phase_reached.add(phase)

        tactile = get_tactile_obs(env)[0]  # (16,)
        max_forces = torch.maximum(max_forces, tactile)

        if phase in (Phase.GRASP, Phase.LIFT) and step % PRINT_EVERY == 0:
            print_tactile(tactile, controller.hand, step, phase)

        if phase == Phase.DONE or controller.current_block_idx >= 1:
            print(f"\n  Controller finished block 0 at step {step}. Stopping.")
            break

    # Final summary
    print(f"\n{'='*55}")
    print(f"TACTILE TEST SUMMARY — max force seen per pad")
    print(f"{'='*55}")
    any_contact = False
    for i, link in enumerate(DEX3_PAD_LINKS):
        f = max_forces[i].item()
        status = "CONTACT ✓" if f > 0.1 else "no contact"
        if f > 0.1:
            any_contact = True
        print(f"  {link:<35} {f:>8.3f} N  {status}")

    print(f"\nPhases reached: {sorted([p.name for p in phase_reached])}")
    print(f"\nOverall: {'TACTILE WORKING ✓' if any_contact else 'NO CONTACTS DETECTED — check sensor setup'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

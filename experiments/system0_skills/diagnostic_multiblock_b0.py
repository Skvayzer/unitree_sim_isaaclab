#!/usr/bin/env python3
"""
Diagnostic: Why does block 0 drop from 70% (single-block) to 15% (multi-block)?

Tests block 0 in the multi-block env but using the SAME eval logic as single-block.
This isolates whether the drop comes from:
  A) The multi-block ENV (3 blocks in scene changes physics)
  B) The STATE MACHINE (different action pipeline, timing, etc.)

Usage:
    cd ~/unitree_sim_isaaclab
    ACCEPT_EULA=Y python -u experiments/system0_skills/diagnostic_multiblock_b0.py \
        --grasp_checkpoint logs/system0_pos_inv_finetune/checkpoint_1500.pt \
        --release_checkpoint logs/system0_release_28d/best_model.pt \
        --num_envs 16 --num_episodes 64 --headless
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Diagnostic: Block 0 in Multi-Block Env")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--grasp_checkpoint", type=str, required=True)
parser.add_argument("--release_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=64)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import math
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg
from experiments.system0_skills.multi_block_config import MultiBlockConfig
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.stacking_state_machine import BlockStackingStateMachine
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()
MCFG = MultiBlockConfig()


def build_obs(env, hand_idx, phase_onehot, device, block_name="block"):
    """Build 28D observation (same as single-block eval)."""
    robot = env.scene["robot"]
    finger_pos = robot.data.joint_pos[:, hand_idx]
    finger_vel = robot.data.joint_vel[:, hand_idx]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
    force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
    contact_binary = (force_mags > 0.1).float()
    return torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                      phase_onehot], dim=-1)


def eval_single_block_env(grasp_actor, release_actor, device, num_envs, num_episodes):
    """Test 1: Single-block env, direct trajectory (baseline)."""
    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(CFG.right_hand_indices, device=device)
    traj = ParameterizedArmTrajectory(CFG, device, num_envs)

    prev_action = torch.zeros(num_envs, 7, device=device)
    total_eps = 0
    total_lifted = 0

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        prev_action.zero_()

        for step in range(350):
            ep_step = env.episode_length_buf
            arm_targets = traj.get_arm_targets(ep_step)
            env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets

            phase_ids = traj.get_phase_ids(ep_step)
            phase_onehot = traj.get_phase_onehot(ep_step)
            obs = build_obs(env, hand_idx, phase_onehot, device)

            is_release = (phase_ids == Phase.RELEASE_HOLD)
            with torch.no_grad():
                grasp_mean, _ = grasp_actor(obs)
                action = grasp_mean.clamp(-1.0, 1.0)
                if is_release.any():
                    release_mean, _ = release_actor(obs)
                    action[is_release] = release_mean[is_release].clamp(-1.0, 1.0)

            is_open = (phase_ids == Phase.HOVER_ABOVE) | (phase_ids == Phase.DESCEND_TO_GRASP) | (phase_ids == Phase.RETREAT)
            action[is_open] = -1.0

            smoothed = CFG.action_ema_alpha * action + (1 - CFG.action_ema_alpha) * prev_action
            prev_action = smoothed.clone()

            obs_dict, _, _, _, _ = env.step(smoothed)
            env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets

            # Check lift during LIFT phase
            is_lift = (phase_ids == Phase.LIFT)
            if is_lift.any():
                block_z = env.scene["block"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                lifted = is_lift & (block_z > CFG.block_initial_z + 0.03)
                # Track per-env (sticky)
                if not hasattr(eval_single_block_env, '_lifted'):
                    eval_single_block_env._lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
                eval_single_block_env._lifted |= lifted

        # Count results
        if hasattr(eval_single_block_env, '_lifted'):
            total_lifted += eval_single_block_env._lifted.sum().item()
            total_eps += num_envs
            eval_single_block_env._lifted.zero_()

    env.close()
    return total_lifted / max(total_eps, 1)


def eval_multiblock_env_direct(grasp_actor, release_actor, device, num_envs, num_episodes):
    """Test 2: Multi-block env, direct trajectory (NO state machine).
    Tests whether the multi-block env itself causes the drop."""
    env_cfg = MultiBlockEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 30.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(CFG.right_hand_indices, device=device)
    traj = ParameterizedArmTrajectory(CFG, device, num_envs)

    # Configure trajectory for block 0 position (y=-0.180, same as single-block)
    block_y = torch.full((num_envs,), -0.180, device=device)
    traj.set_block_positions(block_y)

    prev_action = torch.zeros(num_envs, 7, device=device)
    total_eps = 0
    total_lifted = 0
    lifted_tracker = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        prev_action.zero_()
        lifted_tracker.zero_()

        for step in range(350):
            # Use step counter directly (like single-block eval)
            step_t = torch.full((num_envs,), step, dtype=torch.long, device=device)
            arm_targets = traj.get_arm_targets(step_t)
            env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets

            phase_ids = traj.get_phase_ids(step_t)
            phase_onehot = traj.get_phase_onehot(step_t)
            obs = build_obs(env, hand_idx, phase_onehot, device, block_name="block_0")

            is_release = (phase_ids == Phase.RELEASE_HOLD)
            with torch.no_grad():
                grasp_mean, _ = grasp_actor(obs)
                action = grasp_mean.clamp(-1.0, 1.0)
                if is_release.any():
                    release_mean, _ = release_actor(obs)
                    action[is_release] = release_mean[is_release].clamp(-1.0, 1.0)

            is_open = (phase_ids == Phase.HOVER_ABOVE) | (phase_ids == Phase.DESCEND_TO_GRASP) | (phase_ids == Phase.RETREAT)
            action[is_open] = -1.0

            smoothed = CFG.action_ema_alpha * action + (1 - CFG.action_ema_alpha) * prev_action
            prev_action = smoothed.clone()

            obs_dict, _, _, _, _ = env.step(smoothed)
            env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets

            # Check lift
            is_lift = (phase_ids == Phase.LIFT)
            if is_lift.any():
                block_z = env.scene["block_0"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                lifted = is_lift & (block_z > MCFG.block_initial_z + 0.03)
                lifted_tracker |= lifted

        total_lifted += lifted_tracker.sum().item()
        total_eps += num_envs

    env.close()
    return total_lifted / max(total_eps, 1)


def eval_multiblock_env_statemachine(grasp_actor, release_actor, device, num_envs, num_episodes):
    """Test 3: Multi-block env with state machine (1 block cycle only).
    Tests whether the state machine causes additional drop."""
    env_cfg = MultiBlockEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 30.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    sm = BlockStackingStateMachine(
        grasp_actor=grasp_actor,
        release_actor=release_actor,
        device=device,
        num_envs=num_envs,
        num_blocks=1,  # Only 1 block cycle
        action_ema=CFG.action_ema_alpha,
    )
    sm.configure_blocks([-0.180])  # Block 0 position

    total_eps = 0
    total_lifted = 0
    lifted_tracker = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        sm.reset_all()
        lifted_tracker.zero_()

        for step in range(350):
            if sm.is_all_done():
                break

            action = sm.step(env)
            obs_dict, _, _, _, _ = env.step(action)

            phase_ids = sm.traj.get_phase_ids(sm.cycle_step)
            is_lift = (phase_ids == Phase.LIFT)
            if is_lift.any():
                block_z = env.scene["block_0"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                lifted = is_lift & (block_z > MCFG.block_initial_z + 0.03)
                lifted_tracker |= lifted

        total_lifted += lifted_tracker.sum().item()
        total_eps += num_envs

    env.close()
    return total_lifted / max(total_eps, 1)


def main():
    device = "cuda:0"

    # Load models
    grasp_ckpt = torch.load(args.grasp_checkpoint, map_location=device, weights_only=False)
    grasp_actor = System0Actor(28, 7, 128).to(device)
    grasp_actor.load_state_dict(grasp_ckpt["actor"])
    grasp_actor.eval()

    release_ckpt = torch.load(args.release_checkpoint, map_location=device, weights_only=False)
    release_actor = System0Actor(28, 7, 128).to(device)
    release_actor.load_state_dict(release_ckpt["actor"])
    release_actor.eval()

    print(f"\nGrasp:   {args.grasp_checkpoint}")
    print(f"Release: {args.release_checkpoint}")
    print(f"Envs: {args.num_envs}, Episodes: {args.num_episodes}")
    print()

    # Test 1: Single-block env (baseline)
    print("=" * 60)
    print("TEST 1: Single-block env, direct trajectory (baseline)")
    print("=" * 60)
    t1 = time.time()
    lift_1 = eval_single_block_env(grasp_actor, release_actor, device, args.num_envs, args.num_episodes)
    print(f"  Lift rate: {lift_1:.1%}  ({time.time()-t1:.0f}s)")
    print()

    # Test 2: Multi-block env, direct trajectory (no state machine)
    print("=" * 60)
    print("TEST 2: Multi-block env, direct trajectory (NO state machine)")
    print("  -> Isolates env effect (3 blocks in scene)")
    print("=" * 60)
    t2 = time.time()
    lift_2 = eval_multiblock_env_direct(grasp_actor, release_actor, device, args.num_envs, args.num_episodes)
    print(f"  Lift rate: {lift_2:.1%}  ({time.time()-t2:.0f}s)")
    print()

    # Test 3: Multi-block env with state machine
    print("=" * 60)
    print("TEST 3: Multi-block env, state machine (1-block cycle)")
    print("  -> Isolates state machine effect")
    print("=" * 60)
    t3 = time.time()
    lift_3 = eval_multiblock_env_statemachine(grasp_actor, release_actor, device, args.num_envs, args.num_episodes)
    print(f"  Lift rate: {lift_3:.1%}  ({time.time()-t3:.0f}s)")
    print()

    # Summary
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (single-block env, direct):     {lift_1:.1%}")
    print(f"  Test 2 (multi-block env, direct):       {lift_2:.1%}")
    print(f"  Test 3 (multi-block env, state machine): {lift_3:.1%}")
    print()
    if lift_2 < lift_1 * 0.8:
        print("  >> ENV is causing the drop (multi-block env changes physics)")
        print("  >> Investigate: block collisions, solver settings, contact sensor differences")
    elif lift_3 < lift_2 * 0.8:
        print("  >> STATE MACHINE is causing the drop")
        print("  >> Investigate: action pipeline, phase timing, obs construction differences")
    else:
        print("  >> No significant drop detected with this checkpoint")
        print("  >> The original 70%->15% may have been checkpoint-specific")
    print("=" * 60)


if __name__ == "__main__":
    main()

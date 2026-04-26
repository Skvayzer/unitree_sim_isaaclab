#!/usr/bin/env python3
"""
Evaluate block stacking pipeline.

Mode 1 (default): Single-block grasp+release using ParameterizedArmTrajectory.
Mode 2 (--multi_block): 3-block sequential stacking using BlockStackingStateMachine.

Usage:
    cd ~/unitree_sim_isaaclab
    # Single block:
    ACCEPT_EULA=Y python -u experiments/system0_skills/eval_stacking.py \
        --grasp_checkpoint logs/system0_block_stack/best_model.pt \
        --release_checkpoint logs/system0_release/best_model.pt \
        --num_envs 64 --num_episodes 100 --headless

    # Multi block (3-block tower):
    ACCEPT_EULA=Y python -u experiments/system0_skills/eval_stacking.py \
        --grasp_checkpoint logs/system0_block_stack/best_model.pt \
        --release_checkpoint logs/system0_release/best_model.pt \
        --num_envs 64 --num_episodes 64 --multi_block --headless
"""

import os
import sys
import time
import json
import math

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Evaluate Block Stacking Pipeline")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--grasp_checkpoint", type=str, required=True)
parser.add_argument("--release_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--multi_block", action="store_true",
                    help="Run 3-block sequential stacking")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()
MULTI_CFG = None  # lazy import for multi-block


def build_obs(env, hand_idx, phase_onehot, device):
    """Build 28D observation matching specialist training format."""
    robot = env.scene["robot"]
    finger_pos = robot.data.joint_pos[:, hand_idx]
    finger_vel = robot.data.joint_vel[:, hand_idx]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
    force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
    contact_binary = (force_mags > 0.1).float()
    obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                     phase_onehot], dim=-1)
    return obs.nan_to_num(0.0).clamp(-10.0, 10.0)


def get_orient_deg(env):
    """Get block tilt angle in degrees."""
    q = env.scene["block"].data.root_state_w[:, 3:7]
    up_dot = (1.0 - 2.0 * (q[:, 1]**2 + q[:, 2]**2)).clamp(-1, 1)
    return torch.acos(up_dot) * 180.0 / math.pi


# ═══════════════════════════════════════════════════════════════════
# Single-block evaluation
# ═══════════════════════════════════════════════════════════════════

def run_single_block_eval(env, grasp_actor, release_actor, traj,
                           arm_idx, hand_idx, target_xy, device, num_envs,
                           num_episodes):
    """Run single-block grasp+release using ParameterizedArmTrajectory (same as training)."""
    prev_action = torch.zeros(num_envs, 7, device=device)
    block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    block_was_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    total_episodes = 0
    total_lifted = 0
    total_placed = 0

    start_time = time.time()
    max_steps = num_episodes * 500

    for step_i in range(max_steps):
        robot = env.scene["robot"]
        ep_step = env.episode_length_buf

        # Arm trajectory — identical to training
        arm_targets = traj.get_arm_targets(ep_step)
        robot.data.joint_pos_target[:, arm_idx] = arm_targets

        phase_ids = traj.get_phase_ids(ep_step)
        phase_onehot = traj.get_phase_onehot(ep_step)

        obs = build_obs(env, hand_idx, phase_onehot, device)

        # Select specialist based on phase
        is_release = (phase_ids == Phase.RELEASE_HOLD)

        with torch.no_grad():
            grasp_mean, _ = grasp_actor(obs)
            action = grasp_mean.clamp(-1.0, 1.0)
            if is_release.any():
                release_mean, _ = release_actor(obs)
                action[is_release] = release_mean[is_release].clamp(-1.0, 1.0)

        smoothed = CFG.action_ema_alpha * action + (1 - CFG.action_ema_alpha) * prev_action
        prev_action = smoothed.clone()

        obs_dict, _, terminated, truncated, _ = env.step(smoothed)
        robot.data.joint_pos_target[:, arm_idx] = arm_targets

        done = terminated | truncated

        # Track block state
        block_pos = env.scene["block"].data.root_pos_w
        env_origins = env.scene.env_origins
        block_local = block_pos - env_origins
        block_z = block_local[:, 2]
        block_xy_local = block_local[:, :2]
        target_xy_local = torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device)

        is_lift = (phase_ids == Phase.LIFT)
        just_lifted = is_lift & (block_z > CFG.block_initial_z + 0.03)
        block_was_lifted |= just_lifted

        is_retreat = (phase_ids == Phase.RETREAT)
        xy_err = (block_xy_local - target_xy_local).norm(dim=-1)
        z_err = (block_z - CFG.block_initial_z).abs()
        orient = get_orient_deg(env)
        placed = block_was_lifted & (xy_err < 0.05) & (z_err < 0.05) & (orient < 20) & is_retreat
        block_was_placed |= placed

        if done.any():
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                total_episodes += 1
                total_lifted += int(block_was_lifted[i].item())
                total_placed += int(block_was_placed[i].item())

            if total_episodes % 20 == 0:
                lr = total_lifted / total_episodes
                pr = total_placed / total_episodes
                print(f"  [{total_episodes}/{num_episodes}] lift={lr:.1%} place={pr:.1%}")

            block_was_lifted[done] = False
            block_was_placed[done] = False
            prev_action[done] = 0

            if total_episodes >= num_episodes:
                break

    elapsed = time.time() - start_time
    return {
        "lift_rate": total_lifted / max(total_episodes, 1),
        "place_rate": total_placed / max(total_episodes, 1),
        "total_episodes": total_episodes,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# Multi-block evaluation (3-block tower)
# ═══════════════════════════════════════════════════════════════════

def run_multi_block_eval(grasp_actor, release_actor, device, num_envs, num_episodes):
    """Run 3-block sequential stacking using BlockStackingStateMachine.

    Uses multi_block_env (3 blocks in scene) with extended episode length.
    Each 'episode' = one full 3-block attempt.
    """
    from experiments.system0_skills.stacking_state_machine import BlockStackingStateMachine
    from experiments.system0_skills.multi_block_config import MultiBlockConfig
    import experiments.system0_skills.multi_block_env as mb_env
    from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg

    MCFG = MultiBlockConfig()
    NUM_BLOCKS = MCFG.num_blocks

    # Disable block_fallen termination for eval — the state machine manages
    # its own lifecycle. Auto-resets from block_fallen cause fatal desync
    # (cycle_step vs env state) and the arm's initial motion can knock
    # nearby blocks off the table, triggering spurious terminations.
    mb_env._DISABLE_BLOCK_FALLEN = True

    env_cfg = MultiBlockEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 60.0  # generous: SM finishes well before timeout
    env = ManagerBasedRLEnv(cfg=env_cfg)

    arm_idx = torch.tensor(MCFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(MCFG.right_hand_indices, device=device)

    # Block dispenser pattern: ALL blocks picked from center (strongest grasp position).
    # Blocks 1,2 are teleported to center pick position at cycle transition.
    # This avoids the arm collision issue where blocks at y=-0.280/y=-0.080 are
    # outside the trained curriculum and cause 0% lift.
    CENTER_Y = MCFG.block_positions[0][1]  # -0.180 (center position)
    block_y_positions = [CENTER_Y] * NUM_BLOCKS  # all blocks picked from center

    sm = BlockStackingStateMachine(
        grasp_actor=grasp_actor,
        release_actor=release_actor,
        device=device,
        num_envs=num_envs,
        num_blocks=NUM_BLOCKS,
        action_ema=MCFG.action_ema_alpha,
    )
    sm.configure_blocks(block_y_positions)

    # Track which block index each env was on last step (for teleport detection)
    prev_block_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    per_block_lift = [0] * NUM_BLOCKS
    per_block_place = [0] * NUM_BLOCKS
    tower_count = 0
    total_attempts = 0

    target_xy = torch.tensor([MCFG.target_pos[0], MCFG.target_pos[1]], device=device)
    # Stack heights: block 0 at table, block 1 on block 0, block 2 on block 1
    block_size = 0.04  # 4cm block
    stack_heights = [MCFG.block_initial_z, MCFG.block_initial_z + block_size,
                     MCFG.block_initial_z + 2 * block_size]

    start_time = time.time()
    max_steps_per_attempt = sm.steps_per_cycle * NUM_BLOCKS + 100

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        sm.reset_all()
        env_origins = env.scene.env_origins

        ep_lifted = torch.zeros(num_envs, NUM_BLOCKS, dtype=torch.bool, device=device)
        ep_placed = torch.zeros(num_envs, NUM_BLOCKS, dtype=torch.bool, device=device)

        block_names = ["block_0", "block_1", "block_2"]

        for step in range(max_steps_per_attempt):
            if sm.is_all_done():
                break

            action = sm.step(env)
            obs_dict, _, terminated, truncated, _ = env.step(action)

            # === BLOCK DISPENSER: teleport next block to center pick position ===
            # When the state machine advances to a new block, teleport that block
            # from its staging position to the center pick position (y=-0.180).
            # Also reset arm+finger joints to prevent hysteresis between cycles.
            block_changed = sm.current_block != prev_block_idx
            teleport_mask = block_changed & ~sm.done
            if teleport_mask.any() or (step < 5 and step == 0):
                if teleport_mask.any():
                    changed_envs = torch.where(teleport_mask)[0].tolist()
                    changed_blocks = sm.current_block[teleport_mask].tolist()
                    print(f"    [step={step}] BLOCK CHANGE: envs={changed_envs[:4]} blocks={changed_blocks[:4]}")
            if teleport_mask.any():
                robot = env.scene["robot"]
                env_origins = env.scene.env_origins
                for ei in torch.where(teleport_mask)[0]:
                    bi = sm.current_block[ei].item()
                    if bi > 0 and bi < NUM_BLOCKS:
                        origin = env_origins[ei]
                        # Teleport current block to center pick position
                        block = env.scene[block_names[bi]]
                        state = block.data.root_state_w[ei:ei+1].clone()

                        state[0, 0] = origin[0] + MCFG.block_positions[0][0]  # x = center
                        state[0, 1] = origin[1] + CENTER_Y                     # y = center
                        state[0, 2] = origin[2] + MCFG.block_initial_z         # z = table level
                        state[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device, dtype=state.dtype)
                        state[0, 7:] = 0  # zero velocity
                        block.write_root_state_to_sim(state, env_ids=torch.tensor([ei], device=device))

                        # Reset arm and finger joints to home position
                        arm_idx_t = torch.tensor(MCFG.right_arm_indices, device=device)
                        hand_idx_t = torch.tensor(MCFG.right_hand_indices, device=device)
                        robot.data.joint_pos_target[ei, arm_idx_t] = sm.traj.get_arm_targets(
                            torch.zeros(1, dtype=torch.long, device=device))[0]
                        robot.data.joint_pos_target[ei, hand_idx_t] = 0.0
            prev_block_idx = sm.current_block.clone()

            # Detect auto-resets: IsaacLab auto-resets envs when
            # terminate_any_block_fallen fires. The state machine has no
            # knowledge of this, so cycle_step/current_block desync from
            # the actual env state. Mark auto-reset envs as done (failed).
            auto_reset = terminated | truncated
            if auto_reset.any():
                n_reset = auto_reset.sum().item()
                reset_steps = sm.cycle_step[auto_reset].cpu().tolist()
                reset_blocks = sm.current_block[auto_reset].cpu().tolist()
                print(f"    [step={step}] AUTO-RESET: {n_reset} envs "
                      f"at cycle_steps={reset_steps[:4]} blocks={reset_blocks[:4]}")
                sm.done[auto_reset] = True

            # Re-apply arm targets after env.step
            robot = env.scene["robot"]
            arm_idx_t = torch.tensor(MCFG.right_arm_indices, device=device)
            arm_targets = sm.traj.get_arm_targets(sm.cycle_step)
            robot.data.joint_pos_target[:, arm_idx_t] = arm_targets

            # Track each block's state
            phase_ids = sm.traj.get_phase_ids(sm.cycle_step)
            is_lift = (phase_ids == Phase.LIFT)
            is_retreat = (phase_ids == Phase.RETREAT)

            for bi in range(NUM_BLOCKS):
                block = env.scene[block_names[bi]]
                block_pos_local = block.data.root_pos_w - env_origins
                block_z = block_pos_local[:, 2]
                block_xy = block_pos_local[:, :2]

                # Check if this is the current block being manipulated
                is_current = (sm.current_block == bi) & ~sm.done

                # Lift detection
                just_lifted = is_current & is_lift & (block_z > MCFG.block_initial_z + 0.03)
                ep_lifted[:, bi] |= just_lifted

                # Place detection during retreat
                xy_err = (block_xy - target_xy).norm(dim=-1)
                z_err = (block_z - stack_heights[bi]).abs()
                just_placed = (is_current & is_retreat & ep_lifted[:, bi] &
                               (xy_err < 0.05) & (z_err < 0.05))
                ep_placed[:, bi] |= just_placed

        # Aggregate
        total_attempts += num_envs
        for bi in range(NUM_BLOCKS):
            per_block_lift[bi] += ep_lifted[:, bi].sum().item()
            per_block_place[bi] += ep_placed[:, bi].sum().item()
        tower = ep_placed.all(dim=1)
        tower_count += tower.sum().item()

        # Print results
        for ei in range(min(num_envs, 64)):
            lift_str = " ".join([f"b{bi}={'Y' if ep_lifted[ei, bi] else 'N'}"
                                 for bi in range(NUM_BLOCKS)])
            place_str = " ".join([f"b{bi}={'Y' if ep_placed[ei, bi] else 'N'}"
                                  for bi in range(NUM_BLOCKS)])
            tower_str = "YES!" if ep_placed[ei].all() else "no"
            print(f"  [{ep * num_envs + ei + 1:3d}] lift=[{lift_str}] "
                  f"place=[{place_str}] tower={tower_str}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  STACKING RESULTS ({total_attempts} episodes, {elapsed:.0f}s)")
    for bi in range(NUM_BLOCKS):
        lr = per_block_lift[bi] / total_attempts
        pr = per_block_place[bi] / total_attempts
        print(f"  Block {bi}: lift={lr:.1%} ({per_block_lift[bi]}/{total_attempts})"
              f"  place={pr:.1%} ({per_block_place[bi]}/{total_attempts})")
    tr = tower_count / total_attempts
    print(f"  Tower rate: {tr:.1%} ({tower_count}/{total_attempts})")
    print(f"{'='*60}")

    env.close()
    return {
        "per_block_lift": [l / total_attempts for l in per_block_lift],
        "per_block_place": [p / total_attempts for p in per_block_place],
        "tower_rate": tower_count / total_attempts,
        "total_attempts": total_attempts,
        "elapsed": elapsed,
    }


def main():
    num_envs = args.num_envs
    device = "cuda:0"

    # Load specialists
    grasp_ckpt = torch.load(args.grasp_checkpoint, map_location=device, weights_only=False)
    grasp_actor = System0Actor(28, 7, 128).to(device)
    grasp_actor.load_state_dict(grasp_ckpt["actor"])
    grasp_actor.eval()

    release_ckpt = torch.load(args.release_checkpoint, map_location=device, weights_only=False)
    release_actor = System0Actor(28, 7, 128).to(device)
    release_actor.load_state_dict(release_ckpt["actor"])
    release_actor.eval()

    print(f"\n[INFO] Grasp: {args.grasp_checkpoint}")
    print(f"[INFO] Release: {args.release_checkpoint}")

    if args.multi_block:
        print(f"\n{'='*60}")
        print(f"  MULTI-BLOCK STACKING EVAL (3-block tower)")
        print(f"  Envs: {num_envs}, Episodes: {args.num_episodes}")
        print(f"  Using BlockStackingStateMachine + ParameterizedArmTrajectory")
        print(f"{'='*60}\n")

        results = run_multi_block_eval(
            grasp_actor, release_actor, device, num_envs, args.num_episodes,
        )
    else:
        env_cfg = BlockStackEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.device

        arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
        hand_idx = torch.tensor(CFG.right_hand_indices, device=device)

        traj = ParameterizedArmTrajectory(CFG, device, num_envs)
        target_xy = env.scene.env_origins[:, :2] + torch.tensor(
            [CFG.target_pos[0], CFG.target_pos[1]], device=device)

        obs_dict, _ = env.reset()

        print(f"\n{'='*60}")
        print(f"  SINGLE-BLOCK EVAL (Grasp + Release Specialists)")
        print(f"  Envs: {num_envs}, Episodes: {args.num_episodes}")
        print(f"  Using ParameterizedArmTrajectory (same as training)")
        print(f"{'='*60}\n")

        results = run_single_block_eval(
            env, grasp_actor, release_actor, traj,
            arm_idx, hand_idx, target_xy, device, num_envs, args.num_episodes,
        )

        print(f"\n{'='*60}")
        print(f"  RESULTS ({results['total_episodes']} episodes, {results['elapsed']:.0f}s)")
        print(f"  Lift:  {results['lift_rate']:.1%}")
        print(f"  Place: {results['place_rate']:.1%}")
        print(f"{'='*60}\n")

        env.close()

    # Save results
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_stacking_eval")
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    mode = "multi_block" if args.multi_block else "single_block"
    save_path = os.path.join(save_dir, f"stacking_eval_{mode}_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

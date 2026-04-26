#!/usr/bin/env python3
"""
System 0 Block Stacking Training.

Trains a finger-only RL policy to grasp blocks and stack them,
guided by a scripted arm trajectory.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/train_block_stack.py --num_envs 512 --max_iterations 1500 --headless

    # On remote PC:
    nohup python -u experiments/system0_skills/train_block_stack.py \
        --num_envs 1024 --max_iterations 1500 --headless > train_block_stack.log 2>&1 &
"""

import os
import sys
import time
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="System 0 Block Stacking Training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Max PPO iterations")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--test_reachability", action="store_true",
                    help="Run reachability test instead of training")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import everything else ---
import torch
import torch.nn as nn
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from experiments.system0_skills.arm_trajectory import ArmTrajectory, Phase, ARM_JOINT_NAMES
from experiments.system0_skills.block_stack_rewards import compute_block_stack_reward
from experiments.system0_skills.policy import System0Actor, System0Critic

CFG = BlockStackConfig()


def run_reachability_test(env, num_steps=100):
    """Scripted test: move arm to GRASP position and close fingers via actions.
    Verify that contact forces become non-zero (fingers can reach block).

    We use actions (not direct joint targets) since the ActionManager controls hand joints.
    Action = +1 means close for index/middle (positive direction), action = -1 for thumb (negative).
    """
    print("\n" + "=" * 60)
    print("REACHABILITY TEST: Closing fingers at GRASP arm position")
    print("=" * 60)

    robot = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs

    arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(CFG.right_hand_indices, device=device)

    # Arm grasp position
    grasp_vals = torch.tensor([CFG.arm_grasp_joints[n] for n in ARM_JOINT_NAMES],
                               dtype=torch.float32, device=device)

    # Close action: +1 for index/middle joints (close=positive), -1 for thumb joints (close=negative)
    # Hand indices order: [32:idx0, 33:mid0, 34:thm0, 38:idx1, 39:mid1, 40:thm1, 42:thm2]
    # idx0, mid0 close positive; thm0 close negative; idx1, mid1 close positive; thm1, thm2 close negative
    close_action = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0], device=device)

    max_contact = 0.0
    block_moved = False
    initial_block_pos = env.scene["block"].data.root_pos_w[0].clone()
    max_contacts_per_finger = torch.zeros(3, device=device)

    for step in range(num_steps):
        # Set arm targets (before env.step so they survive action application)
        robot.data.joint_pos_target[:, arm_idx] = grasp_vals.unsqueeze(0).expand(num_envs, -1)

        # Gradually ramp up close action
        t = min(step / 30.0, 1.0)
        action = close_action.unsqueeze(0).expand(num_envs, -1) * t

        # Step environment with close action
        env.step(action)

        # Set arm targets AGAIN after step (in case env.step reset them)
        robot.data.joint_pos_target[:, arm_idx] = grasp_vals.unsqueeze(0).expand(num_envs, -1)

        # Check contacts
        forces = env.scene["fingertip_contacts"].data.net_forces_w
        right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
        force_mags = right_forces.norm(dim=-1)  # [num_envs, 3]
        contacts = (force_mags > 0.1).float()
        max_force = force_mags.max().item()
        max_contact = max(max_contact, max_force)
        max_contacts_per_finger = torch.max(max_contacts_per_finger, force_mags[0])

        # Check block position
        block_pos = env.scene["block"].data.root_pos_w[0]
        block_delta = (block_pos - initial_block_pos).norm().item()

        if step % 10 == 0:
            finger_pos = robot.data.joint_pos[0, hand_idx].cpu().numpy()
            tip_positions = {}
            body_names = list(robot.data.body_names)
            for bi, bname in enumerate(body_names):
                if "right_hand" in bname and ("index_1" in bname or "middle_1" in bname or "thumb_2" in bname):
                    tip_positions[bname] = robot.data.body_pos_w[0, bi].cpu().numpy()

            print(f"\n  Step {step:3d} (action_scale={t:.1f}):")
            print(f"    Finger pos: {finger_pos}")
            print(f"    Contact forces: idx={force_mags[0,0]:.2f} mid={force_mags[0,1]:.2f} thm={force_mags[0,2]:.2f}")
            print(f"    N contacts: {contacts[0].sum().item():.0f}")
            print(f"    Block pos: ({block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f})")
            print(f"    Block delta: {block_delta:.4f}")
            for name, pos in tip_positions.items():
                short = name.replace("right_hand_", "").replace("_link", "")
                print(f"    {short}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        if block_delta > 0.005:
            block_moved = True

    print(f"\n{'='*60}")
    print(f"REACHABILITY RESULT:")
    print(f"  Max contact force: {max_contact:.3f} N")
    print(f"  Per-finger max: idx={max_contacts_per_finger[0]:.2f} mid={max_contacts_per_finger[1]:.2f} thm={max_contacts_per_finger[2]:.2f}")
    print(f"  Block moved: {block_moved} (delta={block_delta:.4f})")
    if max_contact > 0.1:
        print(f"  STATUS: PASS - Fingers can reach the block")
        n_fingers_contacted = (max_contacts_per_finger > 0.1).sum().item()
        print(f"  Fingers that contacted: {n_fingers_contacted}/3")
        if n_fingers_contacted < 2:
            print(f"  WARNING: Only {n_fingers_contacted} finger(s) contacted. Grasp may be weak.")
    else:
        print(f"  STATUS: FAIL - Fingers CANNOT reach the block!")
        print(f"  Block is at: {env.scene['block'].data.root_pos_w[0].cpu().numpy()}")
        print(f"  NEED TO ADJUST BLOCK POSITION OR ARM CONFIGURATION")
    print(f"{'='*60}\n")

    return max_contact > 0.1


def main():
    num_envs = args.num_envs
    max_iterations = args.max_iterations

    # Create environment
    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    print(f"\n[INFO] Environment created with {num_envs} envs on {device}")
    print(f"[INFO] Action dim: {env.action_manager.total_action_dim}")
    print(f"[INFO] Obs dim: {env.observation_manager.group_obs_dim['policy']}")

    # Run reachability test if requested
    if args.test_reachability:
        env.reset()
        passed = run_reachability_test(env)
        env.close()
        return

    # Run brief reachability test before training
    print("\n[INFO] Running brief reachability test...")
    env.reset()
    reachable = run_reachability_test(env, num_steps=60)
    if not reachable:
        print("[WARN] Block may not be reachable! Continuing anyway...")

    # Initialize arm trajectory
    traj = ArmTrajectory(CFG, device=device)
    print(f"[INFO] Arm trajectory: {traj.total_steps} steps per cycle")

    # Initialize policy
    obs_dim = CFG.obs_dim
    act_dim = CFG.action_dim
    actor = System0Actor(obs_dim, act_dim, CFG.hidden_dim).to(device)
    critic = System0Critic(obs_dim, CFG.hidden_dim).to(device)

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print(f"[INFO] Loaded checkpoint from {args.load_checkpoint}")

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=CFG.lr,
    )

    # WandB
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(
                project=CFG.wandb_project,
                entity=CFG.wandb_entity,
                name=f"block_stack_{num_envs}envs",
                config={
                    "num_envs": num_envs,
                    "obs_dim": obs_dim,
                    "action_dim": act_dim,
                    "max_iterations": max_iterations,
                    "action_scale": CFG.action_scale,
                    "lr": CFG.lr,
                    "entropy_coeff": CFG.entropy_coeff,
                    "steps_per_rollout": CFG.steps_per_rollout,
                    "episode_length_s": CFG.episode_length_s,
                    "total_phase_steps": traj.total_steps,
                    "block_initial_pos": CFG.block_initial_pos,
                    "table_height": CFG.table_height,
                },
            )
            wandb = _wandb
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Setup tensors
    arm_idx_tensor = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx_tensor = torch.tensor(CFG.right_hand_indices, device=device)
    # Target position in LOCAL env coordinates; must add env_origins for world comparison
    target_xy_local = torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device)
    # Compute per-env world target_xy by adding env origins (each env is offset in world)
    env_origins = env.scene.env_origins  # [num_envs, 3]
    target_xy = env_origins[:, :2] + target_xy_local.unsqueeze(0)  # [num_envs, 2]

    prev_action = torch.zeros(num_envs, act_dim, device=device)
    obs_dict, _ = env.reset()
    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_block_stack")
    os.makedirs(save_dir, exist_ok=True)

    # Tracking metrics
    episode_blocks_lifted = torch.zeros(num_envs, device=device)
    episode_max_lift = torch.zeros(num_envs, device=device)
    block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)

    print(f"\n{'='*60}")
    print(f"BLOCK STACKING TRAINING")
    print(f"  num_envs={num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  iterations={max_iterations}, steps/rollout={CFG.steps_per_rollout}")
    print(f"  phase steps: hover={CFG.steps_hover}, descend={CFG.steps_descend}, "
          f"grasp={CFG.steps_grasp_hold}, lift={CFG.steps_lift}")
    print(f"  block pos: {CFG.block_initial_pos}")
    print(f"  table height: {CFG.table_height}")
    print(f"  target pos: {CFG.target_pos}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()
        sys.stdout.flush()

        rollout_obs = []
        rollout_act = []
        rollout_logp = []
        rollout_rew = []
        rollout_done = []
        rollout_val = []

        # Accumulate reward info for logging
        reward_info_accum = {}

        for step in range(CFG.steps_per_rollout):
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf  # [num_envs]

            # 1. Compute arm targets from trajectory (vectorized)
            arm_targets = traj.get_arm_targets(ep_step)  # [num_envs, 7]
            robot.data.joint_pos_target[:, arm_idx_tensor] = arm_targets

            # 2. Compute phase info
            phase_ids = traj.get_phase_ids(ep_step)  # [num_envs]
            phase_onehot = traj.get_phase_onehot(ep_step)  # [num_envs, 8]

            # Store phase_onehot on env for observation function
            env._phase_onehot = phase_onehot

            # 3. Build observation
            finger_pos = robot.data.joint_pos[:, hand_idx_tensor]
            finger_vel = robot.data.joint_vel[:, hand_idx_tensor]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()

            obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)
            obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)

            # 4. Policy forward pass
            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(0.15, 10.0)  # floor prevents entropy collapse
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().clamp(-1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = critic(obs)

            # 5. Smooth action
            smoothed_action = CFG.action_ema_alpha * raw_action + (1 - CFG.action_ema_alpha) * prev_action
            prev_action = smoothed_action.clone()

            rollout_obs.append(obs)
            rollout_act.append(raw_action)
            rollout_logp.append(log_prob)
            rollout_val.append(value)

            # 6. Step environment
            obs_dict, env_reward, terminated, truncated, info = env.step(smoothed_action)
            done = terminated | truncated

            # 7. Track block_was_lifted (sticky flag)
            block_pos = env.scene["block"].data.root_pos_w  # [num_envs, 3]
            block_z = block_pos[:, 2]
            block_xy = block_pos[:, :2]
            block_z_local = block_z - env.scene.env_origins[:, 2]

            is_lift_phase = (phase_ids == Phase.LIFT)
            just_lifted = is_lift_phase & (block_z_local > CFG.block_initial_z + 0.03)
            block_was_lifted = block_was_lifted | just_lifted

            # 8. Compute reward (with lifted gate)
            reward, rew_info = compute_block_stack_reward(
                phase_ids=phase_ids,
                block_z=block_z,
                block_xy=block_xy,
                target_xy=target_xy,
                contact_forces=force_mags,
                action=raw_action,
                block_initial_z=CFG.block_initial_z,
                stack_height=CFG.block_initial_z,  # for 1-block, stack at same height
                cfg=CFG,
                block_was_lifted=block_was_lifted,
            )

            # Track metrics
            lift = (block_z_local - CFG.block_initial_z).clamp(min=0.0)
            episode_max_lift = torch.max(episode_max_lift, lift)
            newly_lifted = (lift > 0.02) & (episode_blocks_lifted < 0.5)
            episode_blocks_lifted[newly_lifted] = 1.0

            # Clamp reward to prevent value function explosion
            reward = reward.clamp(-20.0, 20.0)
            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            # Accumulate reward info
            for k, v in rew_info.items():
                if k not in reward_info_accum:
                    reward_info_accum[k] = 0.0
                reward_info_accum[k] += v

            if done.any():
                prev_action[done] = 0.0
                block_was_lifted[done] = False
                episode_blocks_lifted[done] = 0.0
                episode_max_lift[done] = 0.0

        # Stack rollout
        rollout_obs = torch.stack(rollout_obs)
        rollout_act = torch.stack(rollout_act)
        rollout_logp = torch.stack(rollout_logp)
        rollout_rew = torch.stack(rollout_rew)
        rollout_done = torch.stack(rollout_done)
        rollout_val = torch.stack(rollout_val)

        # GAE
        with torch.no_grad():
            # Build final obs for bootstrap
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf
            phase_onehot = traj.get_phase_onehot(ep_step)
            env._phase_onehot = phase_onehot
            finger_pos = robot.data.joint_pos[:, hand_idx_tensor]
            finger_vel = robot.data.joint_vel[:, hand_idx_tensor]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()
            last_obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)
            last_obs = last_obs.nan_to_num(0.0).clamp(-10.0, 10.0)
            last_value = critic(last_obs)

        advantages = torch.zeros_like(rollout_rew)
        gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(CFG.steps_per_rollout)):
            next_val = last_value if t == CFG.steps_per_rollout - 1 else rollout_val[t + 1]
            next_non_terminal = 1.0 - rollout_done[t]
            delta = rollout_rew[t] + CFG.gamma * next_val * next_non_terminal - rollout_val[t]
            gae = delta + CFG.gamma * CFG.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + rollout_val

        T, N = CFG.steps_per_rollout, num_envs
        flat_obs = rollout_obs.reshape(T * N, obs_dim)
        flat_act = rollout_act.reshape(T * N, act_dim)
        flat_logp = rollout_logp.reshape(T * N)
        flat_adv = advantages.reshape(T * N)
        flat_ret = returns.reshape(T * N)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # PPO update
        batch_size = T * N
        mini_batch_size = batch_size // CFG.mini_batches

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(CFG.ppo_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                mb_obs = flat_obs[idx]
                mb_act = flat_act[idx]
                mb_old_logp = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                mean, std = actor(mb_obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.nan_to_num(1.0).clamp(0.15, 10.0)  # must match rollout floor
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs)

                ratio = (new_logp - mb_old_logp).clamp(-20, 20).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CFG.clip_eps, 1 + CFG.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Clip value loss to prevent explosion
                value_loss = 0.5 * (new_val - mb_ret).clamp(-50, 50).pow(2).mean()
                loss = policy_loss + CFG.value_coeff * value_loss - CFG.entropy_coeff * entropy

                # Skip update if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    CFG.max_grad_norm,
                )
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        if n_updates == 0:
            n_updates = 1

        # Logging
        avg_reward = rollout_rew.mean().item()
        max_reward = rollout_rew.max().item()
        min_reward = rollout_rew.min().item()
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        # Phase distribution
        with torch.no_grad():
            ep_step = env.episode_length_buf
            phase_ids = traj.get_phase_ids(ep_step)
            phase_counts = torch.zeros(8, device=device)
            for p in range(8):
                phase_counts[p] = (phase_ids == p).sum().item()

            block_z = env.scene["block"].data.root_pos_w[:, 2]
            mean_block_z = block_z.mean().item()
            max_block_z = block_z.max().item()
            blocks_above_table = (block_z > CFG.block_initial_z + 0.02).float().mean().item()

            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1)
            mean_force = force_mags.mean().item()
            fingers_touching = (force_mags > 0.1).float().sum(dim=-1).mean().item()

        # Average reward info
        for k in reward_info_accum:
            reward_info_accum[k] /= CFG.steps_per_rollout

        if iteration % 10 == 0 or iteration < 5:
            phase_str = " ".join(f"{int(c):3d}" for c in phase_counts.tolist())
            print(
                f"\n[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} (max={max_reward:+.3f}) | "
                f"p_loss={total_policy_loss/n_updates:.4f} v_loss={total_value_loss/n_updates:.4f} | "
                f"ent={total_entropy/n_updates:.3f} | fps={fps:.0f}"
            )
            print(
                f"  phases=[{phase_str}] | "
                f"block_z={mean_block_z:.3f} (max={max_block_z:.3f}) | "
                f"above_table={blocks_above_table:.2f} | "
                f"touch={fingers_touching:.1f} force={mean_force:.2f}"
            )
            print(
                f"  lift_rew={reward_info_accum.get('lift_reward', 0):.3f} | "
                f"contact_rew={reward_info_accum.get('contact_reward', 0):.3f} | "
                f"blocks_lifted={reward_info_accum.get('blocks_lifted', 0):.3f} | "
                f"n_contact_grasp={reward_info_accum.get('n_contacts_during_grasp', 0):.2f}"
            )
            print(
                f"  approach_rew={reward_info_accum.get('approach_reward', 0):.3f} | "
                f"release_rew={reward_info_accum.get('release_reward', 0):.3f} | "
                f"xy_err_release={reward_info_accum.get('xy_error_at_release', 0):.3f} | "
                f"blocks_placed={reward_info_accum.get('blocks_placed', 0):.3f} | "
                f"was_lifted={block_was_lifted.float().mean():.3f}"
            )
            sys.stdout.flush()

        if wandb is not None:
            log_dict = {
                "reward/mean": avg_reward,
                "reward/max": max_reward,
                "reward/min": min_reward,
                "loss/policy": total_policy_loss / n_updates,
                "loss/value": total_value_loss / n_updates,
                "loss/entropy": total_entropy / n_updates,
                "perf/fps": fps,
                "block/mean_z": mean_block_z,
                "block/max_z": max_block_z,
                "block/above_table_frac": blocks_above_table,
                "contact/mean_force": mean_force,
                "contact/fingers_touching": fingers_touching,
                "episode/blocks_lifted": reward_info_accum.get("blocks_lifted", 0),
                "episode/blocks_placed": reward_info_accum.get("blocks_placed", 0),
                "reward/lift_component": reward_info_accum.get("lift_reward", 0),
                "reward/contact_component": reward_info_accum.get("contact_reward", 0),
                "reward/place_component": reward_info_accum.get("place_reward", 0),
                "reward/approach_component": reward_info_accum.get("approach_reward", 0),
                "reward/release_component": reward_info_accum.get("release_reward", 0),
                "placement/xy_error_at_release": reward_info_accum.get("xy_error_at_release", 0),
                "iteration": iteration,
            }
            for p in range(8):
                log_dict[f"phase/{Phase(p).name}"] = phase_counts[p].item()
            wandb.log(log_dict, step=total_steps)

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": best_reward,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }, os.path.join(save_dir, "best_model.pt"))

        if (iteration + 1) % 500 == 0 or iteration == max_iterations - 1:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": avg_reward,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }, os.path.join(save_dir, f"checkpoint_{iteration+1}.pt"))
            print(f"  -> Saved checkpoint at iteration {iteration+1}")

    print(f"\nTraining complete. Best reward: {best_reward:.3f}")
    if wandb is not None:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

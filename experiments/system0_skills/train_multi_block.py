#!/usr/bin/env python3
"""
System 0 Multi-Block Tower Stacking Training.

Trains a finger-only RL policy to grasp 3 blocks and stack them into a tower,
guided by a scripted arm trajectory with 3 pick-place cycles.

Loads v4 single-block checkpoint and expands input layer from 28D to 31D.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/train_multi_block.py --num_envs 512 --max_iterations 3000 --headless

    # On remote PC:
    ACCEPT_EULA=Y nohup python -u experiments/system0_skills/train_multi_block.py \
        --num_envs 1024 --max_iterations 3000 --headless > train_multi_block.log 2>&1 &
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

parser = argparse.ArgumentParser(description="System 0 Multi-Block Tower Stacking Training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=3000, help="Max PPO iterations")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
parser.add_argument("--load_v4", type=str, default=None,
                    help="Path to v4 single-block checkpoint (28D obs, will expand to 31D)")
parser.add_argument("--load_checkpoint", type=str, default=None,
                    help="Resume from multi-block checkpoint (31D obs)")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import everything else ---
import torch
import torch.nn as nn
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from experiments.system0_skills.multi_block_config import MultiBlockConfig
from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg
from experiments.system0_skills.multi_block_trajectory import MultiBlockArmTrajectory
from experiments.system0_skills.multi_block_rewards import compute_multi_block_reward, compute_tower_bonus
from experiments.system0_skills.arm_trajectory import Phase, ARM_JOINT_NAMES
from experiments.system0_skills.policy import System0Actor, System0Critic

CFG = MultiBlockConfig()


def load_v4_checkpoint(actor, critic, ckpt_path, device):
    """Load v4 checkpoint (28D input) into 31D networks.

    Expands the first linear layer: copies 28 columns from checkpoint,
    zero-initializes the 3 new columns (block_idx one-hot).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    old_actor_sd = ckpt["actor"]
    old_critic_sd = ckpt["critic"]

    new_actor_sd = actor.state_dict()
    new_critic_sd = critic.state_dict()

    # Expand first layer of actor
    old_w = old_actor_sd["net.0.weight"]  # [hidden, 28]
    old_b = old_actor_sd["net.0.bias"]    # [hidden]
    new_w = new_actor_sd["net.0.weight"]  # [hidden, 31]
    new_w[:, :old_w.shape[1]] = old_w
    new_w[:, old_w.shape[1]:] = 0.0       # zero-init new columns
    new_actor_sd["net.0.weight"] = new_w
    new_actor_sd["net.0.bias"] = old_b

    # Copy remaining actor layers
    for key in old_actor_sd:
        if key not in ("net.0.weight", "net.0.bias"):
            if key in new_actor_sd and old_actor_sd[key].shape == new_actor_sd[key].shape:
                new_actor_sd[key] = old_actor_sd[key]

    # Expand first layer of critic
    old_w_c = old_critic_sd["net.0.weight"]  # [hidden, 28]
    old_b_c = old_critic_sd["net.0.bias"]
    new_w_c = new_critic_sd["net.0.weight"]  # [hidden, 31]
    new_w_c[:, :old_w_c.shape[1]] = old_w_c
    new_w_c[:, old_w_c.shape[1]:] = 0.0
    new_critic_sd["net.0.weight"] = new_w_c
    new_critic_sd["net.0.bias"] = old_b_c

    # Copy remaining critic layers
    for key in old_critic_sd:
        if key not in ("net.0.weight", "net.0.bias"):
            if key in new_critic_sd and old_critic_sd[key].shape == new_critic_sd[key].shape:
                new_critic_sd[key] = old_critic_sd[key]

    actor.load_state_dict(new_actor_sd)
    critic.load_state_dict(new_critic_sd)
    print(f"[INFO] Loaded v4 checkpoint from {ckpt_path}")
    print(f"[INFO] Expanded input layer: {old_w.shape[1]}D -> {new_w.shape[1]}D")
    print(f"[INFO] v4 iteration: {ckpt.get('iteration', '?')}, reward: {ckpt.get('reward', '?')}")


def main():
    num_envs = args.num_envs
    max_iterations = args.max_iterations

    # Create environment
    env_cfg = MultiBlockEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    print(f"\n[INFO] Multi-block environment created with {num_envs} envs on {device}")
    print(f"[INFO] Action dim: {env.action_manager.total_action_dim}")
    print(f"[INFO] Obs dim: {env.observation_manager.group_obs_dim['policy']}")
    print(f"[INFO] Episode length: {CFG.episode_length_s}s")
    print(f"[INFO] Blocks: {CFG.num_blocks}")
    for i, pos in enumerate(CFG.block_positions):
        print(f"[INFO]   Block {i}: pos={pos}, color={CFG.block_colors[i]}")

    # Initialize trajectory
    traj = MultiBlockArmTrajectory(CFG, device=device)
    print(f"[INFO] Trajectory: {traj.total_steps} total steps ({traj.steps_per_cycle}/cycle x {traj.num_blocks} blocks)")

    # Initialize policy
    obs_dim = CFG.obs_dim  # 31
    act_dim = CFG.action_dim  # 7
    actor = System0Actor(obs_dim, act_dim, CFG.hidden_dim).to(device)
    critic = System0Critic(obs_dim, CFG.hidden_dim).to(device)

    # Load checkpoint
    if args.load_v4:
        load_v4_checkpoint(actor, critic, args.load_v4, device)
    elif args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print(f"[INFO] Loaded multi-block checkpoint from {args.load_checkpoint}")

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
                name=f"multi_block_tower_{num_envs}envs",
                config={
                    "num_envs": num_envs,
                    "obs_dim": obs_dim,
                    "action_dim": act_dim,
                    "max_iterations": max_iterations,
                    "num_blocks": CFG.num_blocks,
                    "action_scale": CFG.action_scale,
                    "lr": CFG.lr,
                    "entropy_coeff": CFG.entropy_coeff,
                    "steps_per_rollout": CFG.steps_per_rollout,
                    "episode_length_s": CFG.episode_length_s,
                    "total_traj_steps": traj.total_steps,
                    "steps_per_cycle": traj.steps_per_cycle,
                    "block_positions": [list(p) for p in CFG.block_positions],
                    "stack_heights": CFG.stack_heights,
                    "target_pos": list(CFG.target_pos),
                    "loaded_v4": args.load_v4 is not None,
                },
            )
            wandb = _wandb
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Setup tensors
    arm_idx_tensor = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx_tensor = torch.tensor(CFG.right_hand_indices, device=device)
    target_xy_local = torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device)
    env_origins = env.scene.env_origins
    target_xy = env_origins[:, :2] + target_xy_local.unsqueeze(0)

    prev_action = torch.zeros(num_envs, act_dim, device=device)
    obs_dict, _ = env.reset()
    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_multi_block")
    os.makedirs(save_dir, exist_ok=True)

    # Per-episode tracking
    episode_blocks_lifted = torch.zeros(num_envs, device=device)
    episode_blocks_placed = torch.zeros(num_envs, device=device)

    print(f"\n{'='*70}")
    print(f"MULTI-BLOCK TOWER STACKING TRAINING")
    print(f"  num_envs={num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  iterations={max_iterations}, steps/rollout={CFG.steps_per_rollout}")
    print(f"  blocks={CFG.num_blocks}, steps/cycle={traj.steps_per_cycle}, total_steps={traj.total_steps}")
    print(f"  stack_heights={CFG.stack_heights}")
    print(f"  target_pos={CFG.target_pos}")
    print(f"  tower_bonus={CFG.reward_tower_bonus}")
    print(f"{'='*70}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()
        sys.stdout.flush()

        rollout_obs = []
        rollout_act = []
        rollout_logp = []
        rollout_rew = []
        rollout_done = []
        rollout_val = []

        reward_info_accum = {}
        tower_checks = 0
        tower_complete_count = 0
        blocks_correct_total = 0.0

        for step in range(CFG.steps_per_rollout):
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf

            # 1. Compute block index and arm targets
            block_idx = traj.get_block_idx(ep_step)  # [num_envs]
            arm_targets = traj.get_arm_targets(ep_step)
            robot.data.joint_pos_target[:, arm_idx_tensor] = arm_targets

            # 2. Phase info
            phase_ids = traj.get_phase_ids(ep_step)
            phase_onehot = traj.get_phase_onehot(ep_step)
            block_onehot = traj.get_block_onehot(ep_step)

            env._phase_onehot = phase_onehot
            env._block_idx_onehot = block_onehot

            # 3. Build observation (31D)
            finger_pos = robot.data.joint_pos[:, hand_idx_tensor]
            finger_vel = robot.data.joint_vel[:, hand_idx_tensor]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()

            obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                             phase_onehot, block_onehot], dim=-1)
            obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)

            # 4. Policy forward
            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(0.3, 10.0)  # floor prevents entropy collapse
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

            # 7. Compute reward for current block
            # Get position of the block currently being manipulated
            current_block_pos = torch.zeros(num_envs, 3, device=device)
            for bi in range(CFG.num_blocks):
                mask = (block_idx == bi)
                if mask.any():
                    current_block_pos[mask] = env.scene[f"block_{bi}"].data.root_pos_w[mask]

            reward, rew_info = compute_multi_block_reward(
                phase_ids=phase_ids,
                block_idx=block_idx,
                current_block_z=current_block_pos[:, 2],
                current_block_xy=current_block_pos[:, :2],
                target_xy=target_xy,
                contact_forces=force_mags,
                action=raw_action,
                block_initial_z=CFG.block_initial_z,
                stack_heights=CFG.stack_heights,
                cfg=CFG,
            )

            # 8. Tower bonus at end of episode (last block's RETREAT phase)
            at_episode_end = (block_idx == 2) & (phase_ids == Phase.RETREAT)
            if at_episode_end.any():
                block_positions = [
                    env.scene[f"block_{i}"].data.root_pos_w for i in range(CFG.num_blocks)
                ]
                tower_complete, blocks_correct, tower_info = compute_tower_bonus(
                    block_positions=block_positions,
                    target_xy=target_xy,
                    stack_heights=CFG.stack_heights,
                    xy_tolerance=CFG.tower_xy_tolerance,
                    z_tolerance=CFG.tower_z_tolerance,
                )
                reward[at_episode_end & tower_complete] += CFG.reward_tower_bonus

                tower_checks += at_episode_end.sum().item()
                tower_complete_count += (at_episode_end & tower_complete).sum().item()
                blocks_correct_total += blocks_correct[at_episode_end].sum().item()

            # Track
            lift = (current_block_pos[:, 2] - CFG.block_initial_z).clamp(min=0.0)
            newly_lifted = lift > 0.02
            episode_blocks_lifted = torch.max(episode_blocks_lifted,
                                               newly_lifted.float() * (block_idx.float() + 1))

            reward = reward.clamp(-20.0, 40.0)  # higher clamp for tower bonus
            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            for k, v in rew_info.items():
                if k not in reward_info_accum:
                    reward_info_accum[k] = 0.0
                reward_info_accum[k] += v

            if done.any():
                prev_action[done] = 0.0
                episode_blocks_lifted[done] = 0.0
                episode_blocks_placed[done] = 0.0

        # Stack rollout
        rollout_obs = torch.stack(rollout_obs)
        rollout_act = torch.stack(rollout_act)
        rollout_logp = torch.stack(rollout_logp)
        rollout_rew = torch.stack(rollout_rew)
        rollout_done = torch.stack(rollout_done)
        rollout_val = torch.stack(rollout_val)

        # GAE
        with torch.no_grad():
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf
            phase_onehot = traj.get_phase_onehot(ep_step)
            block_onehot = traj.get_block_onehot(ep_step)
            env._phase_onehot = phase_onehot
            env._block_idx_onehot = block_onehot
            finger_pos = robot.data.joint_pos[:, hand_idx_tensor]
            finger_vel = robot.data.joint_vel[:, hand_idx_tensor]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()
            last_obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                                  phase_onehot, block_onehot], dim=-1)
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
                std = std.nan_to_num(1.0).clamp(0.3, 10.0)  # must match rollout floor
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs)

                ratio = (new_logp - mb_old_logp).clamp(-20, 20).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CFG.clip_eps, 1 + CFG.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - mb_ret).clamp(-50, 50).pow(2).mean()
                loss = policy_loss + CFG.value_coeff * value_loss - CFG.entropy_coeff * entropy

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

        # Logging
        avg_reward = rollout_rew.mean().item()
        max_reward = rollout_rew.max().item()
        min_reward = rollout_rew.min().item()
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        # Current state
        with torch.no_grad():
            ep_step = env.episode_length_buf
            block_idx_now = traj.get_block_idx(ep_step)
            phase_ids_now = traj.get_phase_ids(ep_step)

            # Block heights
            block_zs = []
            for i in range(CFG.num_blocks):
                bz = env.scene[f"block_{i}"].data.root_pos_w[:, 2]
                block_zs.append(bz.mean().item())

            # Phase distribution
            phase_counts = torch.zeros(8, device=device)
            for p in range(8):
                phase_counts[p] = (phase_ids_now == p).sum().item()

            # Block distribution
            block_counts = torch.zeros(3, device=device)
            for b in range(3):
                block_counts[b] = (block_idx_now == b).sum().item()

            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags_now = right_forces.norm(dim=-1)
            fingers_touching = (force_mags_now > 0.1).float().sum(dim=-1).mean().item()

        for k in reward_info_accum:
            reward_info_accum[k] /= CFG.steps_per_rollout

        tower_complete_rate = tower_complete_count / max(tower_checks, 1)
        avg_blocks_correct = blocks_correct_total / max(tower_checks, 1)

        if iteration % 10 == 0 or iteration < 5:
            phase_str = " ".join(f"{int(c):3d}" for c in phase_counts.tolist())
            block_str = " ".join(f"{int(c):3d}" for c in block_counts.tolist())
            block_z_str = " ".join(f"{z:.3f}" for z in block_zs)
            print(
                f"\n[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} (max={max_reward:+.3f}) | "
                f"p_loss={total_policy_loss/max(n_updates,1):.4f} v_loss={total_value_loss/max(n_updates,1):.4f} | "
                f"ent={total_entropy/max(n_updates,1):.3f} | fps={fps:.0f}"
            )
            print(
                f"  phases=[{phase_str}] | blocks=[{block_str}] | "
                f"block_zs=[{block_z_str}] | touch={fingers_touching:.1f}"
            )
            print(
                f"  lift={reward_info_accum.get('lift_reward', 0):.3f} | "
                f"contact={reward_info_accum.get('contact_reward', 0):.3f} | "
                f"place={reward_info_accum.get('place_reward', 0):.3f} | "
                f"approach={reward_info_accum.get('approach_reward', 0):.3f}"
            )
            print(
                f"  tower_checks={tower_checks} tower_rate={tower_complete_rate:.3f} "
                f"avg_correct={avg_blocks_correct:.2f} | "
                f"blocks_lifted={reward_info_accum.get('blocks_lifted', 0):.3f} | "
                f"blocks_placed={reward_info_accum.get('blocks_placed', 0):.3f}"
            )
            sys.stdout.flush()

        if wandb is not None:
            log_dict = {
                "reward/mean": avg_reward,
                "reward/max": max_reward,
                "reward/min": min_reward,
                "loss/policy": total_policy_loss / max(n_updates, 1),
                "loss/value": total_value_loss / max(n_updates, 1),
                "loss/entropy": total_entropy / max(n_updates, 1),
                "perf/fps": fps,
                "contact/fingers_touching": fingers_touching,
                "tower/complete_rate": tower_complete_rate,
                "tower/avg_blocks_correct": avg_blocks_correct,
                "tower/checks_this_iter": tower_checks,
                "reward/lift_component": reward_info_accum.get("lift_reward", 0),
                "reward/contact_component": reward_info_accum.get("contact_reward", 0),
                "reward/place_component": reward_info_accum.get("place_reward", 0),
                "reward/approach_component": reward_info_accum.get("approach_reward", 0),
                "reward/release_component": reward_info_accum.get("release_reward", 0),
                "episode/blocks_lifted": reward_info_accum.get("blocks_lifted", 0),
                "episode/blocks_placed": reward_info_accum.get("blocks_placed", 0),
                "placement/xy_error_at_release": reward_info_accum.get("xy_error_at_release", 0),
                "placement/z_error_at_retreat": reward_info_accum.get("z_error_at_retreat", 0),
                "iteration": iteration,
            }
            for i in range(CFG.num_blocks):
                log_dict[f"block/z_{i}"] = block_zs[i]
            for p in range(8):
                log_dict[f"phase/{Phase(p).name}"] = phase_counts[p].item()
            for b in range(3):
                log_dict[f"block_dist/{b}"] = block_counts[b].item()
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
                "num_blocks": CFG.num_blocks,
                "stack_heights": CFG.stack_heights,
            }, os.path.join(save_dir, "best_model.pt"))

        if (iteration + 1) % 500 == 0 or iteration == max_iterations - 1:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": avg_reward,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "num_blocks": CFG.num_blocks,
                "stack_heights": CFG.stack_heights,
            }, os.path.join(save_dir, f"checkpoint_{iteration+1}.pt"))
            print(f"  -> Saved checkpoint at iteration {iteration+1}")

    print(f"\nTraining complete. Best reward: {best_reward:.3f}")
    if wandb is not None:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

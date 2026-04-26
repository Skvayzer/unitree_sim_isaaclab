#!/usr/bin/env python3
"""
Fine-tune v4 grasp checkpoint for position invariance.

Key insight: v4 already grasps at 82% at center. The ParameterizedArmTrajectory
adapts the arm to any position, so the finger policy SHOULD be position-invariant
by construction. This script validates that by fine-tuning with position randomization.

Differences from train_position_invariant.py:
- Loads v4 actor checkpoint (fresh asymmetric critic)
- Starts with ±2cm range immediately (no warmup needed — policy already works)
- Lower learning rate (1e-4 for fine-tuning)
- LR cosine decay schedule

Usage:
    cd ~/unitree_sim_isaaclab
    nohup python -u experiments/system0_skills/train_pos_inv_finetune.py \
        --num_envs 1024 --max_iterations 3000 --headless \
        --load_checkpoint logs/system0_block_stack/checkpoint_3000_v4.pt \
        > ~/train_finetune.log 2>&1 &
"""

import os
import sys
import time
import math
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fine-tune Position-Invariant Grasp")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--load_checkpoint", type=str, required=True)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv

from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.block_stack_rewards import compute_block_stack_reward
from experiments.system0_skills.policy import System0Actor, System0AsymmetricCritic

CFG = BlockStackConfig()

RIGHT_PALM_BODY_IDX = 40
BLOCK_FRICTION = CFG.block_friction
BLOCK_MASS = CFG.block_mass


def build_privileged_obs(env, force_mags, device):
    robot = env.scene["robot"]
    block = env.scene["block"]
    env_origins = env.scene.env_origins
    num_envs = env.num_envs

    palm_pos_w = robot.data.body_pos_w[:, RIGHT_PALM_BODY_IDX]
    block_state = block.data.root_state_w
    block_pos_w = block_state[:, :3]
    block_quat = block_state[:, 3:7]
    block_vel = block_state[:, 7:10]

    block_pos_rel = block_pos_w - palm_pos_w
    block_pos_local = block_pos_w - env_origins
    target_3d = torch.tensor(CFG.target_pos, device=device).unsqueeze(0)
    dist_to_target = block_pos_local - target_3d

    friction = torch.full((num_envs, 1), BLOCK_FRICTION, device=device)
    mass = torch.full((num_envs, 1), BLOCK_MASS, device=device)

    priv_obs = torch.cat([
        block_pos_rel, block_quat, block_vel,
        force_mags, dist_to_target, friction, mass,
    ], dim=-1)
    return priv_obs.nan_to_num(0.0).clamp(-10.0, 10.0)


def get_y_range(iteration):
    """Curriculum for position-invariant fine-tuning.

    Must reach ±4cm to cover multi-block positions:
      Block 1 at y=-0.220 (offset -4cm from center y=-0.180)
      Block 2 at y=-0.140 (offset +4cm from center y=-0.180)
    Extended to ±5cm for margin beyond block positions.
    """
    if iteration < 500:
        return (-0.02, 0.02)   # ±2cm (easy warmup)
    elif iteration < 1500:
        return (-0.03, 0.03)   # ±3cm
    elif iteration < 2500:
        return (-0.04, 0.04)   # ±4cm (covers multi-block positions)
    else:
        return (-0.05, 0.05)   # ±5cm (margin for robustness)


def sample_block_y(num_envs, iteration, device):
    y_lo, y_hi = get_y_range(iteration)
    center_y = CFG.block_initial_pos[1]
    offsets = torch.empty(num_envs, device=device).uniform_(y_lo, y_hi)
    return center_y + offsets


def write_block_positions(env, block_y, env_mask=None):
    block = env.scene["block"]
    env_origins = env.scene.env_origins
    if env_mask is not None:
        n = env_mask.sum().item()
        if n == 0:
            return
        state = block.data.root_state_w[env_mask].clone()
        state[:, 0] = env_origins[env_mask, 0] + CFG.block_initial_pos[0]
        state[:, 1] = env_origins[env_mask, 1] + block_y[env_mask]
        state[:, 2] = env_origins[env_mask, 2] + CFG.block_initial_pos[2]
        state[:, 3] = 1.0
        state[:, 4:7] = 0.0
        state[:, 7:13] = 0.0
        block.write_root_state_to_sim(state, env_ids=torch.where(env_mask)[0])
    else:
        state = block.data.root_state_w.clone()
        state[:, 0] = env_origins[:, 0] + CFG.block_initial_pos[0]
        state[:, 1] = env_origins[:, 1] + block_y
        state[:, 2] = env_origins[:, 2] + CFG.block_initial_pos[2]
        state[:, 3] = 1.0
        state[:, 4:7] = 0.0
        state[:, 7:13] = 0.0
        block.write_root_state_to_sim(state)


def main():
    num_envs = args.num_envs
    max_iterations = args.max_iterations

    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    traj = ParameterizedArmTrajectory(CFG, device, num_envs)

    obs_dim = CFG.obs_dim
    act_dim = CFG.action_dim
    actor = System0Actor(obs_dim, act_dim, CFG.hidden_dim).to(device)
    critic = System0AsymmetricCritic(obs_dim, hidden=256).to(device)

    # Load v4 actor checkpoint
    ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(ckpt["actor"])
    print(f"[INFO] Loaded actor from {args.load_checkpoint} (fresh asymmetric critic)")
    print(f"[INFO] Checkpoint was at iter {ckpt.get('iteration', '?')}, reward {ckpt.get('reward', '?')}")

    # Fine-tuning hyperparameters — scale LR with batch size
    # With 16K envs, each mini-batch has 16x more data than 1K envs
    # Use linear scaling rule: LR ∝ sqrt(batch_size)
    BASE_LR = 3e-5     # base LR for 1024 envs fine-tuning
    lr_scale = max(1.0, (num_envs / 1024) ** 0.5)  # sqrt scaling
    INIT_LR = BASE_LR / lr_scale   # ~7.5e-6 for 16K envs
    MIN_LR = 1e-6      # decay target
    ENTROPY_COEFF = 0.02
    STD_FLOOR = 0.3
    CLIP_EPS = 0.2

    # Scale mini_batches with envs to keep mini_batch_size ~16K
    MINI_BATCHES = max(4, num_envs * CFG.steps_per_rollout // 16384)
    CFG.mini_batches = MINI_BATCHES

    # Reduce placement reward scale
    CFG.reward_block_placed = 10.0
    CFG.reward_approach_target = 3.0

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=INIT_LR,
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iterations, eta_min=MIN_LR
    )

    arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(CFG.right_hand_indices, device=device)
    env_origins = env.scene.env_origins
    target_xy = env_origins[:, :2] + torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device).unsqueeze(0)

    prev_action = torch.zeros(num_envs, act_dim, device=device)
    obs_dict, _ = env.reset()

    block_y = sample_block_y(num_envs, 0, device)
    write_block_positions(env, block_y)
    traj.set_block_positions(block_y)

    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_pos_inv_finetune")
    os.makedirs(save_dir, exist_ok=True)

    block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    current_y_range = get_y_range(0)

    print(f"\n{'='*70}")
    print(f"POSITION-INVARIANT FINE-TUNING (from v4 checkpoint)")
    print(f"  num_envs={num_envs}, lr={INIT_LR:.2e}→{MIN_LR:.2e} (cosine), mini_batches={MINI_BATCHES}")
    print(f"  std_floor={STD_FLOOR}, entropy_coeff={ENTROPY_COEFF}")
    print(f"  ASYMMETRIC AC: actor={obs_dim}D, critic={obs_dim}+18=46D")
    print(f"  curriculum: ±2cm(0-500) → ±3cm(500-1500) → ±4cm(1500-2500) → ±5cm(2500+)")
    print(f"{'='*70}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()
        sys.stdout.flush()

        # Cosine LR decay
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        new_y_range = get_y_range(iteration)
        if new_y_range != current_y_range:
            current_y_range = new_y_range
            print(f"\n  >>> CURRICULUM: y_range expanded to ±{new_y_range[1]*100:.1f}cm at iter {iteration}")

        rollout_obs = []
        rollout_act = []
        rollout_logp = []
        rollout_rew = []
        rollout_done = []
        rollout_val = []
        rollout_priv = []
        reward_info_accum = {}

        for step in range(CFG.steps_per_rollout):
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf

            arm_targets = traj.get_arm_targets(ep_step)
            robot.data.joint_pos_target[:, arm_idx] = arm_targets

            phase_ids = traj.get_phase_ids(ep_step)
            phase_onehot = traj.get_phase_onehot(ep_step)
            env._phase_onehot = phase_onehot

            finger_pos = robot.data.joint_pos[:, hand_idx]
            finger_vel = robot.data.joint_vel[:, hand_idx]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()

            obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)
            obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)

            priv_obs = build_privileged_obs(env, force_mags, device)

            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(STD_FLOOR, 10.0)
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().clamp(-1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = critic(obs, priv_obs)

            smoothed = CFG.action_ema_alpha * raw_action + (1 - CFG.action_ema_alpha) * prev_action
            prev_action = smoothed.clone()

            rollout_obs.append(obs)
            rollout_act.append(raw_action)
            rollout_logp.append(log_prob)
            rollout_val.append(value)
            rollout_priv.append(priv_obs)

            obs_dict, env_reward, terminated, truncated, info = env.step(smoothed)
            done = terminated | truncated

            block_pos = env.scene["block"].data.root_pos_w
            block_z = block_pos[:, 2]
            block_xy = block_pos[:, :2]
            block_z_local = block_z - env_origins[:, 2]

            is_lift_phase = (phase_ids == Phase.LIFT)
            just_lifted = is_lift_phase & (block_z_local > CFG.block_initial_z + 0.03)
            block_was_lifted = block_was_lifted | just_lifted

            reward, rew_info = compute_block_stack_reward(
                phase_ids=phase_ids, block_z=block_z, block_xy=block_xy,
                target_xy=target_xy, contact_forces=force_mags, action=raw_action,
                block_initial_z=CFG.block_initial_z, stack_height=CFG.block_initial_z,
                cfg=CFG, block_was_lifted=block_was_lifted,
            )

            reward = reward.clamp(-20.0, 20.0)
            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            for k, v in rew_info.items():
                if k not in reward_info_accum:
                    reward_info_accum[k] = 0.0
                reward_info_accum[k] += v

            if done.any():
                prev_action[done] = 0.0
                block_was_lifted[done] = False
                new_y = sample_block_y(done.sum().item(), iteration, device)
                block_y[done] = new_y
                write_block_positions(env, block_y, env_mask=done)
                traj.set_block_positions(new_y, env_mask=done)

        rollout_obs = torch.stack(rollout_obs)
        rollout_act = torch.stack(rollout_act)
        rollout_logp = torch.stack(rollout_logp)
        rollout_rew = torch.stack(rollout_rew)
        rollout_done = torch.stack(rollout_done)
        rollout_val = torch.stack(rollout_val)
        rollout_priv = torch.stack(rollout_priv)

        # GAE
        with torch.no_grad():
            robot = env.scene["robot"]
            ep_step = env.episode_length_buf
            phase_onehot = traj.get_phase_onehot(ep_step)
            env._phase_onehot = phase_onehot
            finger_pos = robot.data.joint_pos[:, hand_idx]
            finger_vel = robot.data.joint_vel[:, hand_idx]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()
            last_obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)
            last_obs = last_obs.nan_to_num(0.0).clamp(-10.0, 10.0)
            last_priv = build_privileged_obs(env, force_mags, device)
            last_value = critic(last_obs, last_priv)

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
        flat_priv = rollout_priv.reshape(T * N, System0AsymmetricCritic.PRIVILEGED_DIM)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

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
                mb_priv = flat_priv[idx]

                mean, std = actor(mb_obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.nan_to_num(1.0).clamp(STD_FLOOR, 10.0)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs, mb_priv)

                ratio = (new_logp - mb_old_logp).clamp(-20, 20).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - mb_ret).clamp(-50, 50).pow(2).mean()
                loss = policy_loss + CFG.value_coeff * value_loss - ENTROPY_COEFF * entropy

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

        avg_reward = rollout_rew.mean().item()
        max_reward = rollout_rew.max().item()
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        with torch.no_grad():
            ep_step = env.episode_length_buf
            block_z = env.scene["block"].data.root_pos_w[:, 2]
            block_z_local = block_z - env_origins[:, 2]
            blocks_above = (block_z_local > CFG.block_initial_z + 0.02).float().mean().item()

            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags_now = right_forces.norm(dim=-1)
            fingers_touching = (force_mags_now > 0.1).float().sum(dim=-1).mean().item()

        for k in reward_info_accum:
            reward_info_accum[k] /= CFG.steps_per_rollout

        if n_updates == 0:
            n_updates = 1

        if iteration % 10 == 0 or iteration < 5:
            y_lo, y_hi = current_y_range
            print(
                f"\n[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} | p_loss={total_policy_loss/n_updates:.4f} "
                f"v_loss={total_value_loss/n_updates:.1f} | ent={total_entropy/n_updates:.3f} "
                f"| lr={current_lr:.2e} | fps={fps:.0f}"
            )
            print(
                f"  above={blocks_above:.2f} touch={fingers_touching:.1f} | "
                f"lifted={reward_info_accum.get('blocks_lifted', 0):.3f} "
                f"placed={reward_info_accum.get('blocks_placed', 0):.3f} "
                f"was_lifted={block_was_lifted.float().mean():.3f} | "
                f"y=[{y_lo:+.3f},{y_hi:+.3f}]"
            )
            sys.stdout.flush()

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

    print(f"\nFine-tuning complete. Best reward: {best_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

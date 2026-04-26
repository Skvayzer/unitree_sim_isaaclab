#!/usr/bin/env python3
"""
Position-Invariant Grasp Training.

Trains a finger-only RL policy to grasp blocks at randomized y-positions,
using ParameterizedArmTrajectory for position-adaptive arm control.

Key differences from train_block_stack.py:
- ParameterizedArmTrajectory with piecewise-linear SR mapping
- Curriculum: gradually widens block y-position range
- block_was_lifted gate: prevents placement reward hacking
- Std floor (0.3): prevents entropy collapse with many envs
- Block position randomization on reset

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/train_position_invariant.py --num_envs 512 --max_iterations 10000 --headless

    # Background:
    nohup python -u experiments/system0_skills/train_position_invariant.py \
        --num_envs 512 --max_iterations 10000 --headless > train_pos_inv.log 2>&1 &
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

parser = argparse.ArgumentParser(description="Position-Invariant Grasp Training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=10000, help="Max PPO iterations")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Resume from checkpoint")
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

# Body index for right palm (for privileged obs)
RIGHT_PALM_BODY_IDX = 40

# Current block physics (constants until domain randomization is added)
BLOCK_FRICTION = CFG.block_friction  # 2.0
BLOCK_MASS = CFG.block_mass          # 0.05 kg


def build_privileged_obs(env, force_mags, target_xy_local, device):
    """Build 18D privileged observation for asymmetric critic.

    Privileged info the actor never sees but the critic uses for better value estimation:
      block_pos_relative_to_palm(3) — where is block relative to hand?
      block_quat(4)                  — block orientation
      block_linear_vel(3)            — is block moving/slipping?
      grip_force_per_finger(3)       — per-fingertip force magnitudes
      distance_to_target(3)          — vector from block to target
      friction_coefficient(1)        — surface friction
      block_mass(1)                  — object mass
    """
    robot = env.scene["robot"]
    block = env.scene["block"]
    env_origins = env.scene.env_origins
    num_envs = env.num_envs

    # Palm position in world
    palm_pos_w = robot.data.body_pos_w[:, RIGHT_PALM_BODY_IDX]  # [N, 3]

    # Block state
    block_state = block.data.root_state_w  # [N, 13]
    block_pos_w = block_state[:, :3]
    block_quat = block_state[:, 3:7]
    block_vel = block_state[:, 7:10]

    # Relative position: block pos in palm frame (simple subtraction, no rotation)
    block_pos_rel = block_pos_w - palm_pos_w  # [N, 3]

    # Distance to target (in local env coords)
    block_pos_local = block_pos_w - env_origins
    target_3d = torch.tensor(CFG.target_pos, device=device).unsqueeze(0)
    dist_to_target = block_pos_local - target_3d  # [N, 3]

    # Constants (will be randomized later with domain randomization)
    friction = torch.full((num_envs, 1), BLOCK_FRICTION, device=device)
    mass = torch.full((num_envs, 1), BLOCK_MASS, device=device)

    priv_obs = torch.cat([
        block_pos_rel,       # 3
        block_quat,          # 4
        block_vel,           # 3
        force_mags,          # 3 (per-fingertip)
        dist_to_target,      # 3
        friction,            # 1
        mass,                # 1
    ], dim=-1)  # total: 18

    return priv_obs.nan_to_num(0.0).clamp(-10.0, 10.0)


# --- Curriculum (success-gated, per DexPBT/UniDexGrasp++ literature) ---
CURRICULUM_STAGES = [
    (-0.01, 0.01),   # Stage 0: ±1cm
    (-0.02, 0.02),   # Stage 1: ±2cm (intermediate, smoother progression)
    (-0.03, 0.03),   # Stage 2: ±3cm
    (-0.04, 0.04),   # Stage 3: ±4cm
    (-0.05, 0.05),   # Stage 4: ±5cm (full validated SR range)
]
ADVANCE_LIFT_RATE = 0.50   # advance when lift rate exceeds this
RETREAT_LIFT_RATE = 0.20   # retreat when lift rate drops below this
MIN_ITERS_PER_STAGE = 200  # minimum iterations before advancing/retreating
# Fallback: force advance at these iterations if lift rate never reaches threshold
FORCE_ADVANCE_ITERS = [2000, 4000, 6000, 8000]


class CurriculumManager:
    """Success-gated curriculum: advance on lift_rate > 50%, retreat on < 20%."""

    def __init__(self):
        self.stage = 0
        self.iters_in_stage = 0
        self.lift_rate_ema = 0.0
        self.ema_alpha = 0.05  # smoothing factor for lift rate EMA

    def get_y_range(self):
        return CURRICULUM_STAGES[self.stage]

    def update(self, iteration, lift_rate):
        """Update curriculum stage based on lift rate. Returns True if stage changed."""
        self.iters_in_stage += 1
        self.lift_rate_ema = self.ema_alpha * lift_rate + (1 - self.ema_alpha) * self.lift_rate_ema

        old_stage = self.stage

        # Check forced advance (fallback for time-based guarantee)
        for force_iter in FORCE_ADVANCE_ITERS:
            if iteration == force_iter and self.stage < len(CURRICULUM_STAGES) - 1:
                self.stage += 1
                self.iters_in_stage = 0
                print(f"\n  >>> CURRICULUM: FORCED advance to stage {self.stage} "
                      f"(±{CURRICULUM_STAGES[self.stage][1]*100:.0f}cm) at iter {iteration} "
                      f"(lift_ema={self.lift_rate_ema:.3f})")
                return True

        if self.iters_in_stage < MIN_ITERS_PER_STAGE:
            return False

        # Advance if doing well
        if self.lift_rate_ema > ADVANCE_LIFT_RATE and self.stage < len(CURRICULUM_STAGES) - 1:
            self.stage += 1
            self.iters_in_stage = 0
            print(f"\n  >>> CURRICULUM: advance to stage {self.stage} "
                  f"(±{CURRICULUM_STAGES[self.stage][1]*100:.0f}cm) at iter {iteration} "
                  f"(lift_ema={self.lift_rate_ema:.3f} > {ADVANCE_LIFT_RATE})")
            return True

        # Retreat if struggling
        if self.lift_rate_ema < RETREAT_LIFT_RATE and self.stage > 0:
            self.stage -= 1
            self.iters_in_stage = 0
            print(f"\n  >>> CURRICULUM: RETREAT to stage {self.stage} "
                  f"(±{CURRICULUM_STAGES[self.stage][1]*100:.0f}cm) at iter {iteration} "
                  f"(lift_ema={self.lift_rate_ema:.3f} < {RETREAT_LIFT_RATE})")
            return True

        return old_stage != self.stage


def sample_block_y(num_envs, y_range, device):
    """Sample block y-positions from curriculum range."""
    y_lo, y_hi = y_range
    center_y = CFG.block_initial_pos[1]  # -0.180
    offsets = torch.empty(num_envs, device=device).uniform_(y_lo, y_hi)
    return center_y + offsets


def write_block_positions(env, block_y, env_mask=None):
    """Write new block positions to sim (y-axis only, keep x/z fixed)."""
    block = env.scene["block"]
    env_origins = env.scene.env_origins
    num_envs = env.num_envs
    device = env.device

    if env_mask is not None:
        # Partial update
        n = env_mask.sum().item()
        if n == 0:
            return
        state = block.data.root_state_w[env_mask].clone()
        # Set position in world coords: env_origin + local_pos
        state[:, 0] = env_origins[env_mask, 0] + CFG.block_initial_pos[0]  # x fixed
        state[:, 1] = env_origins[env_mask, 1] + block_y[env_mask]         # y randomized
        state[:, 2] = env_origins[env_mask, 2] + CFG.block_initial_pos[2]  # z fixed
        # Reset orientation to upright
        state[:, 3] = 1.0  # qw
        state[:, 4:7] = 0.0  # qx, qy, qz
        # Zero velocity
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

    print(f"\n[INFO] Environment created with {num_envs} envs on {device}")

    # Initialize parameterized arm trajectory
    traj = ParameterizedArmTrajectory(CFG, device, num_envs)
    print(f"[INFO] Arm trajectory: {traj.total_steps} steps per cycle")
    print(f"[INFO] Target SR: {traj.target_sr:.4f}")

    # Initialize policy with asymmetric actor-critic
    obs_dim = CFG.obs_dim
    act_dim = CFG.action_dim
    actor = System0Actor(obs_dim, act_dim, CFG.hidden_dim).to(device)
    critic = System0AsymmetricCritic(obs_dim, hidden=256).to(device)  # 46D input (28+18)

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        # Asymmetric critic has different architecture; don't load old critic
        print(f"[INFO] Loaded actor from {args.load_checkpoint} (fresh asymmetric critic)")
        # Reinitialize log_std to avoid collapsed std from old checkpoint
        # log(0.25) = -1.386 → std=0.25 (above floor=0.15, room to decrease)
        old_std = actor.log_std.data.exp().mean().item()
        actor.log_std.data.fill_(math.log(0.50))
        print(f"[INFO] Reset log_std: old_std={old_std:.4f} → new_std=0.50 (entropy≈5.1 for 7D)")

    # Annealing schedule (per research: DexPBT, UniDexGrasp++, OpenAI ShadowHand)
    ENTROPY_COEFF_START = 0.05  # high exploration early
    ENTROPY_COEFF_END = 0.01    # converge to literature-standard value
    ENTROPY_ANNEAL_END = 5000   # linear anneal over first 5K iters
    STD_FLOOR_EARLY = 0.40      # 7D: entropy≈3.4, prevents collapse
    STD_FLOOR_LATE = 0.25       # precision phase after lift_rate > 30%
    STD_FLOOR_SWITCH_ITER = 4000  # switch point (also gated on lift rate in curriculum)

    # Override reward scales for position-invariant training
    # Reduce placement reward to prevent v_loss explosion (30→10)
    CFG.reward_block_placed = 10.0
    CFG.reward_approach_target = 3.0  # reduce from 5.0
    CLIP_EPS = 0.2
    LR_START = 1e-4   # documented optimal for Phase 5
    LR_END = 5e-6
    VALUE_COEFF = 0.5  # reduce from 1.0 to prevent v_loss dominating gradients

    # Scale mini_batches with num_envs to keep mini-batch size ~16K
    # Default mini_batches=4 is for 512 envs. With 16K envs need 64 mini-batches.
    batch_size = num_envs * CFG.steps_per_rollout
    TARGET_MINI_BATCH = 16384
    CFG.mini_batches = max(4, batch_size // TARGET_MINI_BATCH)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LR_START,
    )
    # Cosine LR decay: 1e-4 → 5e-6 over max_iterations
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iterations, eta_min=LR_END
    )

    # WandB
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(
                project=CFG.wandb_project,
                entity=CFG.wandb_entity,
                name=f"pos_inv_{num_envs}envs",
                config={
                    "num_envs": num_envs,
                    "max_iterations": max_iterations,
                    "lr_start": LR_START,
                    "lr_end": LR_END,
                    "entropy_coeff_start": ENTROPY_COEFF_START,
                    "entropy_coeff_end": ENTROPY_COEFF_END,
                    "clip_eps": CLIP_EPS,
                    "std_floor_early": STD_FLOOR_EARLY,
                    "std_floor_late": STD_FLOOR_LATE,
                    "curriculum": "success-gated: advance@lift>0.5, retreat@lift<0.2",
                },
            )
            wandb = _wandb
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Setup tensors
    arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
    hand_idx = torch.tensor(CFG.right_hand_indices, device=device)
    env_origins = env.scene.env_origins
    target_xy_local = torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device)
    target_xy = env_origins[:, :2] + target_xy_local.unsqueeze(0)

    prev_action = torch.zeros(num_envs, act_dim, device=device)
    obs_dict, _ = env.reset()

    # Initialize block positions and trajectory
    curriculum = CurriculumManager()
    block_y = sample_block_y(num_envs, curriculum.get_y_range(), device)
    write_block_positions(env, block_y)
    traj.set_block_positions(block_y)

    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_pos_invariant")
    os.makedirs(save_dir, exist_ok=True)

    # Tracking
    block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_max_lift = torch.zeros(num_envs, device=device)
    current_y_range = curriculum.get_y_range()

    print(f"\n{'='*70}")
    print(f"POSITION-INVARIANT GRASP TRAINING")
    print(f"  num_envs={num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  iterations={max_iterations}, steps/rollout={CFG.steps_per_rollout}")
    print(f"  lr={LR_START}→{LR_END} (cosine), entropy_coeff={ENTROPY_COEFF_START}→{ENTROPY_COEFF_END} (anneal {ENTROPY_ANNEAL_END}), clip_eps={CLIP_EPS}")
    print(f"  std_floor={STD_FLOOR_EARLY}→{STD_FLOOR_LATE} (switch at iter {STD_FLOOR_SWITCH_ITER})")
    print(f"  ASYMMETRIC AC: actor={obs_dim}D, critic={obs_dim}+{System0AsymmetricCritic.PRIVILEGED_DIM}D={obs_dim+System0AsymmetricCritic.PRIVILEGED_DIM}D")
    print(f"  curriculum: SUCCESS-GATED (advance@lift>{ADVANCE_LIFT_RATE}, retreat@lift<{RETREAT_LIFT_RATE})")
    print(f"  stages: {[f'±{s[1]*100:.0f}cm' for s in CURRICULUM_STAGES]}")
    print(f"  block_was_lifted gate: ON")
    print(f"{'='*70}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()
        sys.stdout.flush()

        # Anneal std floor for this iteration (used in both rollout and PPO update)
        iter_std_floor = STD_FLOOR_LATE if iteration >= STD_FLOOR_SWITCH_ITER else STD_FLOOR_EARLY

        # Check curriculum change (success-gated)
        current_y_range = curriculum.get_y_range()

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

            # 1. Arm targets from parameterized trajectory
            arm_targets = traj.get_arm_targets(ep_step)
            robot.data.joint_pos_target[:, arm_idx] = arm_targets

            # 2. Phase info
            phase_ids = traj.get_phase_ids(ep_step)
            phase_onehot = traj.get_phase_onehot(ep_step)
            env._phase_onehot = phase_onehot

            # 3. Build observation
            finger_pos = robot.data.joint_pos[:, hand_idx]
            finger_vel = robot.data.joint_vel[:, hand_idx]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
            contact_binary = (force_mags > 0.1).float()

            obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)
            obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)

            # 4. Build privileged obs for asymmetric critic
            priv_obs = build_privileged_obs(env, force_mags, target_xy_local, device)

            # 5. Policy forward (actor sees 28D, critic sees 28+18=46D)
            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(iter_std_floor, 10.0)  # MUST match PPO update floor
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().clamp(-1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = critic(obs, priv_obs)

            # 6. Smooth action
            smoothed = CFG.action_ema_alpha * raw_action + (1 - CFG.action_ema_alpha) * prev_action
            prev_action = smoothed.clone()

            rollout_obs.append(obs)
            rollout_act.append(raw_action)
            rollout_logp.append(log_prob)
            rollout_val.append(value)
            rollout_priv.append(priv_obs)

            # 6. Step
            obs_dict, env_reward, terminated, truncated, info = env.step(smoothed)
            done = terminated | truncated

            # 7. Track block_was_lifted (sticky flag)
            block_pos = env.scene["block"].data.root_pos_w
            block_z = block_pos[:, 2]
            block_xy = block_pos[:, :2]
            block_z_local = block_z - env_origins[:, 2]

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
                stack_height=CFG.block_initial_z,
                cfg=CFG,
                block_was_lifted=block_was_lifted,
            )

            # Track metrics
            lift = (block_z_local - CFG.block_initial_z).clamp(min=0.0)
            episode_max_lift = torch.max(episode_max_lift, lift)

            reward = reward.clamp(-20.0, 20.0)
            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            for k, v in rew_info.items():
                if k not in reward_info_accum:
                    reward_info_accum[k] = 0.0
                reward_info_accum[k] += v

            # 9. On reset: randomize block positions
            if done.any():
                prev_action[done] = 0.0
                block_was_lifted[done] = False
                episode_max_lift[done] = 0.0

                # Sample new block positions for reset envs
                new_y = sample_block_y(done.sum().item(), curriculum.get_y_range(), device)
                # Update block_y for reset envs
                block_y[done] = new_y
                write_block_positions(env, block_y, env_mask=done)
                traj.set_block_positions(new_y, env_mask=done)

        # Stack rollout
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
            last_priv = build_privileged_obs(env, force_mags, target_xy_local, device)
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

        # Anneal entropy coeff (std floor already computed as iter_std_floor)
        ent_frac = min(1.0, iteration / max(1, ENTROPY_ANNEAL_END))
        entropy_coeff = ENTROPY_COEFF_START + (ENTROPY_COEFF_END - ENTROPY_COEFF_START) * ent_frac

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
                mb_priv = flat_priv[idx]

                mean, std = actor(mb_obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                # STD FLOOR: must match rollout floor for consistent importance ratios
                std = std.nan_to_num(1.0).clamp(iter_std_floor, 10.0)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs, mb_priv)

                ratio = (new_logp - mb_old_logp).clamp(-20, 20).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - mb_ret).clamp(-50, 50).pow(2).mean()
                loss = policy_loss + VALUE_COEFF * value_loss - entropy_coeff * entropy

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
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        with torch.no_grad():
            ep_step = env.episode_length_buf
            phase_ids = traj.get_phase_ids(ep_step)
            phase_counts = torch.zeros(8, device=device)
            for p in range(8):
                phase_counts[p] = (phase_ids == p).sum().item()

            block_z = env.scene["block"].data.root_pos_w[:, 2]
            block_z_local = block_z - env_origins[:, 2]
            blocks_above = (block_z_local > CFG.block_initial_z + 0.02).float().mean().item()

            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags_now = right_forces.norm(dim=-1)
            fingers_touching = (force_mags_now > 0.1).float().sum(dim=-1).mean().item()
            mean_force = force_mags_now.mean().item()

        for k in reward_info_accum:
            reward_info_accum[k] /= CFG.steps_per_rollout

        # Success-gated curriculum update
        cur_lift_rate = reward_info_accum.get("blocks_lifted", 0)
        curriculum.update(iteration, cur_lift_rate)
        current_y_range = curriculum.get_y_range()

        if n_updates == 0:
            n_updates = 1  # avoid div by zero

        # Step LR scheduler
        lr_scheduler.step()

        if iteration % 10 == 0 or iteration < 5:
            y_lo, y_hi = current_y_range
            phase_str = " ".join(f"{int(c):3d}" for c in phase_counts.tolist())
            print(
                f"\n[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} (max={max_reward:+.3f}) | "
                f"p_loss={total_policy_loss/n_updates:.4f} v_loss={total_value_loss/n_updates:.4f} | "
                f"ent={total_entropy/n_updates:.3f} | lr={optimizer.param_groups[0]['lr']:.1e} | fps={fps:.0f}"
            )
            print(
                f"  phases=[{phase_str}] | "
                f"above_table={blocks_above:.2f} | "
                f"touch={fingers_touching:.1f} force={mean_force:.2f} | "
                f"y_range=[{y_lo:+.3f},{y_hi:+.3f}]"
            )
            print(
                f"  lift_rew={reward_info_accum.get('lift_reward', 0):.3f} | "
                f"contact={reward_info_accum.get('contact_reward', 0):.3f} | "
                f"lifted={reward_info_accum.get('blocks_lifted', 0):.3f} | "
                f"placed={reward_info_accum.get('blocks_placed', 0):.3f} | "
                f"was_lifted={block_was_lifted.float().mean():.3f}"
            )
            sys.stdout.flush()

        if wandb is not None:
            log_dict = {
                "reward/mean": avg_reward,
                "reward/max": max_reward,
                "loss/policy": total_policy_loss / n_updates,
                "loss/value": total_value_loss / n_updates,
                "loss/entropy": total_entropy / n_updates,
                "perf/fps": fps,
                "block/above_table_frac": blocks_above,
                "contact/fingers_touching": fingers_touching,
                "contact/mean_force": mean_force,
                "episode/blocks_lifted": reward_info_accum.get("blocks_lifted", 0),
                "episode/blocks_placed": reward_info_accum.get("blocks_placed", 0),
                "episode/was_lifted_frac": block_was_lifted.float().mean().item(),
                "curriculum/y_range": current_y_range[1],
                "curriculum/stage": curriculum.stage,
                "curriculum/lift_rate_ema": curriculum.lift_rate_ema,
                "hyperparams/entropy_coeff": entropy_coeff,
                "hyperparams/std_floor": iter_std_floor,
                "iteration": iteration,
            }
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

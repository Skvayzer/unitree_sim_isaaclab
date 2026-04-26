#!/usr/bin/env python3
"""
System 0 — Blind Tactile Grasping RL Training.

The robot is blind (no cameras). The arm is pre-positioned at hover pose
by high-stiffness actuators in BlockStackEnvCfg. Only the 7 right-hand
finger joints are controlled. The policy learns to search for the block
with gentle touch and then grasp it.

Architecture
────────────
  obs  (93-D): right_finger_qpos(7) + qvel(7) + tactile_ext(72) + torques(7)
  action (7-D): offset from default hand pose, bounded by tanh × delta_max
  intent(128-D): one-hot curriculum stage ([:4]), rest zero

Curriculum (block XY randomization beyond env's built-in ±8 mm)
────────────────────────────────────────────────────────────────
  Stage 0: fixed  (only env's ±8 mm)
  Stage 1: +±2 cm extra offset
  Stage 2: +±5 cm extra offset
  Stage 3: +±5 cm + domain randomisation (mass/friction)
  Advance when lift_rate ≥ 75 % over last 1000 episodes.

Usage
─────
  python experiments/system0_rl/train.py --num_envs 512 --headless
  python experiments/system0_rl/train.py --num_envs 1 --total_timesteps 5000
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import deque

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PROJECT_ROOT"] = project_root
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="System 0 Blind Grasping RL Training")
parser.add_argument("--num_envs",         type=int,   default=512)
parser.add_argument("--total_timesteps",  type=int,   default=10_000_000)
parser.add_argument("--checkpoint",       type=str,   default=None)
parser.add_argument("--curriculum_stage", type=int,   default=0,
                    help="Starting curriculum stage (0-3).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher    = AppLauncher(args)
simulation_app  = app_launcher.app

import torch
import numpy as np

from experiments.system0_rl.config    import TrainConfig
from experiments.system0_rl.system0_moe import System0Config, System0PPOWrapper
from experiments.system0_rl.ppo       import RolloutBuffer, ppo_update
from experiments.system0_rl.rewards   import compute_reward_blind, is_lift_success, set_contact_baseline, BLOCK_INIT_Z
from tasks.common_observations.tactile_state import (
    get_tactile_obs_extended,
    reset_tactile_state,
)

# ── Joint name lists ─────────────────────────────────────────────────────────

RIGHT_HAND_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

LEFT_HAND_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]


# ── OU Noise ─────────────────────────────────────────────────────────────────

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration.

    Gives the policy smooth, mean-reverting random perturbations in action
    space, which is important for tactile search (abrupt random steps would
    mask the tactile gradient signal).
    """

    def __init__(self, shape: tuple, theta: float, sigma: float, device):
        self.theta  = theta
        self.sigma  = sigma
        self.device = device
        self.state  = torch.zeros(shape, device=device)

    def reset(self, env_ids: "torch.Tensor | None" = None) -> None:
        if env_ids is None:
            self.state.zero_()
        else:
            self.state[env_ids] = 0.0

    def sample(self) -> torch.Tensor:
        dx = -self.theta * self.state + self.sigma * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state.clone()


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_joint_index_maps(env):
    """Return list of indices in robot.data.joint_* for the 7 right-hand joints."""
    joint_names = env.scene["robot"].data.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    sim_hand = [name_to_idx[n] for n in RIGHT_HAND_NAMES if n in name_to_idx]
    if len(sim_hand) != 7:
        missing = [n for n in RIGHT_HAND_NAMES if n not in name_to_idx]
        raise RuntimeError(f"Could not find right-hand joints: {missing}")

    print(f"\n[IndexMap] right hand indices ({len(sim_hand)}): {sim_hand}")
    for idx in sim_hand:
        pos = env.scene["robot"].data.joint_pos[0, idx].item()
        print(f"  [{idx:2d}] {joint_names[idx]:40s}  pos={pos:+.4f}")
    return sim_hand


def build_obs_batch(env, sim_hand: list, device) -> torch.Tensor:
    """Build (num_envs, 93) observation tensor.

    Layout: right_finger_qpos(7) | qvel(7) | tactile_ext(72) | torques(7)
    """
    robot = env.scene["robot"]
    hand_pos = robot.data.joint_pos[:, sim_hand].to(device)   # (N, 7)
    hand_vel = robot.data.joint_vel[:, sim_hand].to(device)   # (N, 7)

    try:
        tactile = get_tactile_obs_extended(env).to(device)    # (N, 72)
    except Exception:
        tactile = torch.zeros(hand_pos.shape[0], 72, device=device)

    try:
        torques = robot.data.applied_torque[:, sim_hand].to(device)  # (N, 7)
    except (AttributeError, IndexError):
        torques = torch.zeros_like(hand_pos)

    obs = torch.cat([hand_pos, hand_vel, tactile, torques], dim=1)  # (N, 93)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def build_left_joint_index_map(env) -> list:
    """Return indices in robot.data.joint_* for the 7 left-hand joints (zero-filled if absent)."""
    joint_names = env.scene["robot"].data.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    indices = [name_to_idx[n] for n in LEFT_HAND_NAMES if n in name_to_idx]
    if len(indices) != 7:
        print(f"[IndexMap] WARNING: only {len(indices)}/7 left-hand joints found — zero-filling obs")
    return indices


def build_rl_system0_obs(env, sim_right: list, sim_left: list, device) -> torch.Tensor:
    """Build (num_envs, 100) observation tensor for RLSystem0Policy.

    Layout: tactile_ext(72) | right_torques(7) | right_qpos(7) | left_torques(7) | left_qpos(7)
    Left hand is zero-filled if sim_left is empty (G1 left hand not in scene).
    """
    robot = env.scene["robot"]
    N = robot.data.joint_pos.shape[0]

    try:
        tactile = get_tactile_obs_extended(env).to(device)        # (N, 72)
    except Exception:
        tactile = torch.zeros(N, 72, device=device)

    r_qpos = robot.data.joint_pos[:, sim_right].to(device)        # (N, 7)
    try:
        r_torques = robot.data.applied_torque[:, sim_right].to(device)
    except (AttributeError, IndexError):
        r_torques = torch.zeros_like(r_qpos)

    if sim_left:
        l_qpos    = robot.data.joint_pos[:, sim_left].to(device)  # (N, 7)
        try:
            l_torques = robot.data.applied_torque[:, sim_left].to(device)
        except (AttributeError, IndexError):
            l_torques = torch.zeros_like(l_qpos)
    else:
        l_qpos    = torch.zeros(N, 7, device=device)
        l_torques = torch.zeros(N, 7, device=device)

    obs = torch.cat([tactile, r_torques, r_qpos, l_torques, l_qpos], dim=1)  # (N, 100)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def apply_curriculum(env, just_reset: torch.Tensor, stage: int, device) -> None:
    """Teleport block with extra XY randomisation for curriculum stages 1-3.

    The env's EventCfg already applies ±8 mm on reset. Here we add an
    additional offset on top of that for stages 1+.
    """
    if stage == 0 or not just_reset.any():
        return

    xy_ranges = (0.00, 0.02, 0.05, 0.05)
    extra = xy_ranges[min(stage, 3)]
    if extra == 0.0:
        return

    block       = env.scene["block"]
    states      = block.data.root_state_w.clone()          # (N, 13)
    reset_ids   = just_reset.nonzero(as_tuple=False).squeeze(1)
    n_reset     = len(reset_ids)

    offsets = (torch.rand(n_reset, 2, device=device) * 2.0 - 1.0) * extra
    states[reset_ids, 0] += offsets[:, 0]
    states[reset_ids, 1] += offsets[:, 1]

    block.write_root_state_to_sim(states)
    env.scene.write_data_to_sim()


def encode_curriculum_intent(stage: int, N: int, intent_dim: int, device) -> torch.Tensor:
    """One-hot curriculum stage in the first 4 dims of a (N, intent_dim) tensor."""
    intent = torch.zeros(N, intent_dim, device=device)
    intent[:, min(stage, 3)] = 1.0
    return intent


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = TrainConfig(
        num_envs=args.num_envs,
        headless=getattr(args, "headless", True),
        total_timesteps=args.total_timesteps,
    )
    device = torch.device(getattr(args, "device", "cuda:0"))

    print("=" * 60)
    print("System 0 — Blind Tactile Grasping RL")
    print(f"  Envs:           {config.num_envs}")
    print(f"  Device:         {device}")
    print(f"  Timesteps:      {config.total_timesteps:,}")
    print(f"  Rollout steps:  {config.rollout_steps} per env")
    print(f"  Obs dim:        {config.joint_dim + config.vel_dim + config.tactile_dim + config.torque_dim}")
    print(f"  Action dim:     {config.target_dim}")
    print("=" * 60)

    # ── Environment ──────────────────────────────────────────────────────────
    from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = config.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    N = config.num_envs
    print(f"Environment created: {N} envs  |  action_dim={env.action_manager.total_action_dim}")

    sim_hand = build_joint_index_maps(env)

    # ── Policy ───────────────────────────────────────────────────────────────
    moe_cfg = System0Config(
        joint_dim   = config.joint_dim,
        vel_dim     = config.vel_dim,
        tactile_dim = config.tactile_dim,
        torque_dim  = config.torque_dim,
        target_dim  = config.target_dim,
        intent_dim  = config.intent_dim,
        hidden_dim  = config.hidden_dim,
        n_experts   = config.n_experts,
        top_k       = config.top_k,
        action_dim  = config.joint_dim,  # 7 right-hand joints = action output dim
    )
    policy    = System0PPOWrapper(moe_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    print(f"Policy: {sum(p.numel() for p in policy.parameters())/1e6:.2f} M params")

    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("total_steps", 0)
        print(f"Resumed from step {start_step:,}")

    # ── Exploration noise ─────────────────────────────────────────────────────
    ou_noise = OUNoise((N, config.target_dim), config.ou_theta, config.ou_sigma, device)

    # ── Rollout buffer ────────────────────────────────────────────────────────
    obs_dim = config.joint_dim + config.vel_dim + config.tactile_dim + config.torque_dim
    buffer  = RolloutBuffer(
        rollout_steps = config.rollout_steps,
        num_envs      = N,
        obs_dim       = obs_dim + config.target_dim,   # obs_with_targets input to MoE
        intent_dim    = config.intent_dim,
        action_dim    = config.joint_dim,              # action dim stored in buffer
        device        = device,
    )

    # ── Episode tracking ──────────────────────────────────────────────────────
    ep_rewards         = torch.zeros(N, device=device)
    ep_lengths         = torch.zeros(N, dtype=torch.long, device=device)
    # Track whether the block was lifted at ANY point in the current episode.
    # Checked before reset — after env.step() resets the env, block is already
    # back to init position, so we must accumulate during the rollout.
    lifted_this_ep     = torch.zeros(N, dtype=torch.bool, device=device)
    recent_rewards     = deque(maxlen=200)
    recent_lengths     = deque(maxlen=200)
    recent_lift        = deque(maxlen=1000)  # bool — was block lifted this ep?
    total_episodes     = 0
    curriculum_stage   = args.curriculum_stage
    stage_episodes     = 0
    prev_hand_vel      = torch.zeros(N, 7, device=device)
    # Per-env block initial Z captured on each reset (fix for hardcoded 0.819 magic number)
    block_init_z       = torch.full((N,), BLOCK_INIT_Z, device=device)

    # ── Wandb ─────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project="System0_Blind",
        entity="skvayzer",
        config={
            "num_envs": N, "total_timesteps": config.total_timesteps,
            "lr": config.lr, "gamma": config.gamma, "clip_eps": config.clip_eps,
            "rollout_steps": config.rollout_steps, "ppo_epochs": config.ppo_epochs,
            "n_experts": config.n_experts, "top_k": config.top_k,
            "ou_theta": config.ou_theta, "ou_sigma": config.ou_sigma,
            "delta_max": config.delta_max,
        },
        name=f"s0_blind_{N}envs_{time.strftime('%m%d_%H%M')}",
    )
    print("[wandb] Initialized")

    # ── Reset ─────────────────────────────────────────────────────────────────
    env.reset()
    reset_tactile_state()
    ou_noise.reset()
    simulation_app.update()
    # Tare the contact sensor: capture idle table-contact forces at hover pose
    # so rewards use differential (block-contact-only) forces.
    n_act = env.action_manager.total_action_dim
    _zero_act = torch.zeros(config.num_envs, n_act, device=device)
    for _ in range(5):   # settle for 5 steps before reading baseline
        env.step(_zero_act)
    set_contact_baseline(env, device)

    print("\nTraining started...")
    t_start    = time.time()
    total_steps = start_step
    n_updates  = 0
    coarse_targets = torch.zeros(N, config.target_dim, device=device)  # open-hand baseline

    while total_steps < config.total_timesteps:
        buffer.reset()
        intent = encode_curriculum_intent(curriculum_stage, N, config.intent_dim, device)

        # ── Rollout collection ────────────────────────────────────────────────
        for step in range(config.rollout_steps):
            obs_batch        = build_obs_batch(env, sim_hand, device)            # (N, 93)
            obs_with_targets = torch.cat([obs_batch, coarse_targets], dim=1)    # (N, 100)

            with torch.no_grad():
                raw_delta, log_probs, values = policy.act(obs_with_targets, intent)

            # Bound action; add OU noise for exploration
            action = torch.tanh(raw_delta) * config.delta_max + ou_noise.sample()
            action = action.clamp(-config.delta_max, config.delta_max)

            _, _, terminated, truncated, _ = env.step(action)

            if step % 10 == 0:
                simulation_app.update()

            cur_hand_vel  = env.scene["robot"].data.joint_vel[:, sim_hand].to(device)
            rewards       = compute_reward_blind(env, prev_hand_vel, cur_hand_vel, device,
                                                 block_init_z=block_init_z)
            prev_hand_vel = cur_hand_vel.detach().clone()

            just_reset = (terminated | truncated).to(device)

            # Accumulate lift success BEFORE reset clears the block position
            lifted_this_ep |= is_lift_success(env, device, block_init_z=block_init_z)

            # Apply curriculum block offsets for reset envs; capture new block Z baseline
            if just_reset.any():
                reset_ids = just_reset.nonzero(as_tuple=False).squeeze(1)
                apply_curriculum(env, just_reset, curriculum_stage, device)
                ou_noise.reset(reset_ids)
                reset_tactile_state(reset_ids, N, device)
                prev_hand_vel[reset_ids] = 0.0
                # Update per-env block initial Z after block teleport
                try:
                    block_init_z[reset_ids] = env.scene["block"].data.root_pos_w[reset_ids, 2].to(device)
                except (KeyError, AttributeError):
                    pass

            # Store step in buffer (all envs at once — correct for per-env GAE)
            buffer.add_step(obs_with_targets, intent, action, log_probs,
                            rewards, just_reset.float(), values)

            ep_rewards += rewards
            ep_lengths += 1
            total_steps += N

            # Episode bookkeeping (after accumulating lifted_this_ep above)
            if just_reset.any():
                for i in just_reset.nonzero(as_tuple=False).squeeze(1).tolist():
                    recent_rewards.append(ep_rewards[i].item())
                    recent_lengths.append(ep_lengths[i].item())
                    recent_lift.append(bool(lifted_this_ep[i].item()))
                    ep_rewards[i]     = 0.0
                    ep_lengths[i]     = 0
                    lifted_this_ep[i] = False
                    total_episodes   += 1
                    stage_episodes   += 1

        # ── PPO update ────────────────────────────────────────────────────────
        with torch.no_grad():
            last_obs = build_obs_batch(env, sim_hand, device)
            last_obs_full = torch.cat([last_obs, coarse_targets], dim=1)
            _, _, last_values = policy.act(last_obs_full, intent)

        buffer.compute_advantages(last_values, just_reset.float(),
                                  config.gamma, config.gae_lambda)
        metrics = ppo_update(policy, optimizer, buffer, config)
        n_updates += 1

        # ── Curriculum advancement ────────────────────────────────────────────
        if (stage_episodes >= config.curriculum_min_episodes
                and len(recent_lift) >= config.curriculum_min_episodes
                and curriculum_stage < 3):
            lift_rate = sum(recent_lift) / len(recent_lift)
            if lift_rate >= config.curriculum_success_threshold:
                curriculum_stage += 1
                stage_episodes    = 0
                print(f"\n>>> Curriculum advanced to stage {curriculum_stage}  "
                      f"(lift_rate={lift_rate:.1%})\n")
                wandb.log({"curriculum/stage": curriculum_stage}, step=total_steps)

        # ── Logging ───────────────────────────────────────────────────────────
        if n_updates % config.log_interval == 0:
            elapsed   = time.time() - t_start
            fps       = total_steps / max(elapsed, 1)
            avg_r     = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            avg_len   = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            lift_rate = float(np.mean(list(recent_lift)[-200:])) if recent_lift else 0.0
            print(
                f"Step {total_steps:>10,} | "
                f"Ep {total_episodes:>6,} | "
                f"R {avg_r:>8.1f} | "
                f"Len {avg_len:>5.0f} | "
                f"Lift {lift_rate:.1%} | "
                f"Curric {curriculum_stage} | "
                f"PL {metrics['policy_loss']:.4f} | "
                f"VL {metrics['value_loss']:.4f} | "
                f"Ent {metrics['entropy']:.3f} | "
                f"FPS {fps:.0f}"
            )
            wandb.log({
                "reward/mean":           avg_r,
                "reward/episode_length": avg_len,
                "reward/lift_rate":      lift_rate,
                "reward/total_episodes": total_episodes,
                "loss/policy":           metrics["policy_loss"],
                "loss/value":            metrics["value_loss"],
                "loss/entropy":          metrics["entropy"],
                "curriculum/stage":      curriculum_stage,
                "perf/fps":              fps,
                "perf/total_steps":      total_steps,
            }, step=total_steps)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if n_updates % config.save_interval == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"step_{total_steps:010d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy":           policy.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "total_steps":      total_steps,
                "n_updates":        n_updates,
                "curriculum_stage": curriculum_stage,
                "avg_reward":       avg_r if "avg_r" in dir() else 0.0,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = Path(config.checkpoint_dir) / "final.pt"
    torch.save({
        "policy":           policy.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "total_steps":      total_steps,
        "curriculum_stage": curriculum_stage,
    }, final_path)
    print(f"\nDone.  Final checkpoint: {final_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

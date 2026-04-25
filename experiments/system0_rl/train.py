#!/usr/bin/env python3
"""
System 0 MoE standalone RL training with Isaac Lab / Isaac Sim.
Supports parallel environments for fast training.

Usage:
    cd unitree_sim_isaaclab
    # Visual test (1 env):
    python experiments/system0_rl/train.py --num_envs 1 --total_timesteps 5000 --enable_cameras
    # Test controller only:
    python experiments/system0_rl/train.py --num_envs 1 --test_controller --total_timesteps 500 --enable_cameras
    # Full training (parallel, headless):
    python experiments/system0_rl/train.py --num_envs 64 --total_timesteps 2000000 --headless
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

parser = argparse.ArgumentParser(description="System 0 MoE RL Training")
parser.add_argument("--task", type=str, default="Isaac-Stack-RgyBlock-G129-Dex3-Joint")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--total_timesteps", type=int, default=2_000_000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--test_controller", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

try:
    import tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3  # noqa
except ImportError:
    import tasks  # noqa

from experiments.system0_rl.config import TrainConfig
from experiments.system0_rl.system0_moe import System0Config, System0PPOWrapper
from experiments.system0_rl.scripted_controller import ScriptedController, Phase
from experiments.system0_rl.rewards import compute_reward
from experiments.system0_rl.ppo import RolloutBuffer, ppo_update

ARM_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]


def build_joint_index_maps(env):
    robot = env.scene["robot"]
    joint_names = robot.data.joint_names
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    print(f"\n{'='*70}")
    print(f"JOINT DIAGNOSTICS — {len(joint_names)} joints")
    print(f"{'='*70}")
    for i, name in enumerate(joint_names):
        pos = robot.data.joint_pos[0, i].item()
        try:
            lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
            hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
            print(f"  [{i:2d}] {name:40s}  pos={pos:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        except (IndexError, RuntimeError):
            print(f"  [{i:2d}] {name:40s}  pos={pos:+.4f}")
    print(f"{'='*70}\n")

    sim_arm = [name_to_idx[n] for n in ARM_NAMES if n in name_to_idx]
    sim_hand = [name_to_idx[n] for n in HAND_NAMES if n in name_to_idx]
    print(f"[IndexMap] arm ({len(sim_arm)}): {sim_arm}")
    print(f"[IndexMap] hand ({len(sim_hand)}): {sim_hand}")
    assert len(sim_arm) == 14 and len(sim_hand) == 14
    return sim_arm, sim_hand


def build_obs_batch(env, sim_arm, sim_hand, device):
    """Build observations for ALL envs at once. Returns (N, obs_dim)."""
    robot = env.scene["robot"]
    N = robot.data.joint_pos.shape[0]
    arm_pos = robot.data.joint_pos[:, sim_arm].to(device)       # (N, 14)
    hand_pos = robot.data.joint_pos[:, sim_hand].to(device)     # (N, 14)
    arm_vel = robot.data.joint_vel[:, sim_arm].to(device)       # (N, 14)
    hand_vel = robot.data.joint_vel[:, sim_hand].to(device)     # (N, 14)

    try:
        forces = env.scene["fingertip_contacts"].data.net_forces_w  # (N, 6, 3)
        tactile = forces.reshape(N, -1).to(device)[:, :18]
    except (KeyError, AttributeError):
        tactile = torch.zeros(N, 18, device=device)

    try:
        arm_t = robot.data.applied_torque[:, sim_arm].to(device)
        hand_t = robot.data.applied_torque[:, sim_hand].to(device)
        torques = torch.cat([arm_t, hand_t], dim=1)
    except (AttributeError, IndexError):
        torques = torch.zeros(N, 28, device=device)

    obs = torch.cat([arm_pos, hand_pos, arm_vel, hand_vel, tactile, torques], dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def map_to_sim_batch(refined, env, sim_arm, sim_hand):
    """Map (N, 28) refined targets to (N, n_joints) sim action."""
    N = refined.shape[0]
    full = env.scene["robot"].data.default_joint_pos[:N].clone()
    full[:, sim_arm] = refined[:, :14]
    full[:, sim_hand] = refined[:, 14:28]
    return full


def main():
    config = TrainConfig(
        num_envs=args.num_envs,
        headless=args.headless if hasattr(args, 'headless') else False,
        total_timesteps=args.total_timesteps,
    )
    device = torch.device(getattr(args, "device", "cuda:0"))

    print("=" * 60)
    print("System 0 MoE RL Training")
    print(f"  Envs: {config.num_envs}")
    print(f"  Device: {device}")
    print(f"  Timesteps: {config.total_timesteps}")
    print(f"  Rollout: {config.rollout_steps}")
    if args.test_controller:
        print("  MODE: --test_controller (delta_q=0)")
    print("=" * 60)

    from experiments.system0_rl.env_cfg import System0TrainEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    env_cfg = System0TrainEnvCfg()
    env_cfg.scene.num_envs = config.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    N = config.num_envs
    print(f"Environment created: {N} envs")

    sim_arm, sim_hand = build_joint_index_maps(env)

    moe_cfg = System0Config(
        joint_dim=config.joint_dim, vel_dim=config.vel_dim,
        tactile_dim=config.tactile_dim, torque_dim=config.torque_dim,
        target_dim=config.target_dim, intent_dim=config.intent_dim,
        hidden_dim=config.hidden_dim, n_experts=config.n_experts,
        top_k=config.top_k,
    )
    policy = System0PPOWrapper(moe_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    print(f"Policy: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M params")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("total_steps", 0)
        print(f"Resumed from step {start_step}")
    else:
        start_step = 0

    # Per-env controllers
    controllers = [
        ScriptedController(config, device=device,
                           sim_arm_indices=sim_arm, sim_hand_indices=sim_hand,
                           env_idx=i)
        for i in range(N)
    ]

    obs_dim = config.joint_dim + config.vel_dim + config.tactile_dim + config.torque_dim
    buffer = RolloutBuffer(
        config.rollout_steps * N, obs_dim + config.target_dim,
        config.intent_dim, config.joint_dim, device=device,
    )

    # Episode tracking
    ep_rewards = torch.zeros(N, device=device)
    ep_lengths = torch.zeros(N, dtype=torch.long, device=device)
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_phases = deque(maxlen=100)  # track max phase reached per episode

    total_steps = start_step
    n_updates = 0
    total_episodes = 0

    env.reset()
    for c in controllers:
        c.reset()

    # Wandb logging
    import wandb
    if not args.test_controller:
        wandb.init(
            project="System0_MoE",
            entity="skvayzer",
            config={
                "num_envs": N, "total_timesteps": config.total_timesteps,
                "lr": config.lr, "gamma": config.gamma, "clip_eps": config.clip_eps,
                "rollout_steps": config.rollout_steps, "ppo_epochs": config.ppo_epochs,
                "n_experts": config.n_experts, "top_k": config.top_k,
                "approach_dist": config.approach_dist,
            },
            name=f"s0_moe_{N}envs_{time.strftime('%m%d_%H%M')}",
        )
        print("[wandb] Initialized")
    else:
        wandb_run = None

    print("\nTraining started...")
    t_start = time.time()

    while total_steps < config.total_timesteps:
        buffer.reset()

        for step in range(config.rollout_steps):
            obs_batch = build_obs_batch(env, sim_arm, sim_hand, device)

            # Run per-env controllers (sequential for now — IK is CPU-bound)
            coarse_list = []
            intent_list = []
            for i in range(N):
                ct, pi = controllers[i].step(env)
                coarse_list.append(ct)
                intent_list.append(pi)
            coarse_targets = torch.stack(coarse_list).to(device)  # (N, 28)
            phase_intents = torch.stack(intent_list).to(device)   # (N, 128)

            obs_with_targets = torch.cat([obs_batch, coarse_targets], dim=1)  # (N, obs+28)

            if args.test_controller:
                delta_q = torch.zeros(N, config.joint_dim, device=device)
                log_probs = torch.zeros(N, device=device)
                values = torch.zeros(N, device=device)
            else:
                with torch.no_grad():
                    delta_q, log_probs, values = policy.act(obs_with_targets, phase_intents)

            refined = coarse_targets + delta_q
            sim_action = map_to_sim_batch(refined, env, sim_arm, sim_hand)

            env.step(sim_action)

            if step % 10 == 0:
                simulation_app.update()

            # Per-env rewards and block OOB check
            rewards = torch.zeros(N, device=device)
            dones = torch.zeros(N, dtype=torch.bool, device=device)

            for i in range(N):
                block_idx = min(controllers[i].current_block_idx, len(config.stacking_order) - 1)
                r = compute_reward(
                    env, controllers[i].phase, controllers[i].hand,
                    config.stacking_order[block_idx],
                    controllers[i].current_block_idx,
                    coarse_targets[i], refined[i], config, device,
                    env_idx=i,
                )
                rewards[i] = r

                # Block OOB penalty (env auto-resets via termination function)
                for bname in config.stacking_order:
                    try:
                        bpos = env.scene[bname].data.root_pos_w[i]
                        if bpos[2].item() < 0.75 or ((bpos[:2] - env.scene["robot"].data.root_pos_w[i, :2]) ** 2).sum().sqrt().item() > 0.5:
                            rewards[i] += -50.0
                            break
                    except (KeyError, AttributeError):
                        pass

            # Store in buffer (flatten across envs)
            for i in range(N):
                buffer.add(
                    obs_with_targets[i], phase_intents[i], delta_q[i],
                    log_probs[i], rewards[i].item(), float(dones[i]), values[i],
                )

            ep_rewards += rewards
            ep_lengths += 1
            total_steps += N

            # Handle resets
            reset_ids = []
            for i in range(N):
                if dones[i]:
                    recent_rewards.append(ep_rewards[i].item())
                    recent_lengths.append(ep_lengths[i].item())
                    recent_phases.append(controllers[i].phase.value)
                    total_episodes += 1
                    ep_rewards[i] = 0
                    ep_lengths[i] = 0
                    controllers[i].reset()
                    reset_ids.append(i)
            # Env auto-resets terminated envs via block_oob termination function

        # PPO update
        with torch.no_grad():
            last_obs = build_obs_batch(env, sim_arm, sim_hand, device)
            last_ct = []
            last_pi = []
            for i in range(N):
                ct, pi = controllers[i].step(env)
                last_ct.append(ct)
                last_pi.append(pi)
            last_obs_full = torch.cat([last_obs, torch.stack(last_ct).to(device)], dim=1)
            if args.test_controller:
                last_values = torch.zeros(N, device=device)
            else:
                _, _, last_values = policy.act(last_obs_full, torch.stack(last_pi).to(device))

        # Use mean last value for GAE
        buffer.compute_advantages(last_values.mean(), config.gamma, config.gae_lambda)

        if not args.test_controller:
            metrics = ppo_update(policy, optimizer, buffer, config)
        else:
            metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates += 1

        # Logging
        if n_updates % config.log_interval == 0:
            elapsed = time.time() - t_start
            fps = total_steps / max(elapsed, 1)
            avg_r = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_len = np.mean(recent_lengths) if recent_lengths else 0.0
            max_phase = max(recent_phases) if recent_phases else 0
            phase_names = {0:"APPROACH", 1:"PRE_GRASP", 2:"GRASP", 3:"LIFT",
                           4:"TRANSPORT", 5:"DESCEND", 6:"RELEASE", 7:"RETREAT", 8:"DONE"}
            print(
                f"Step {total_steps:>9d} | "
                f"Ep {total_episodes:>5d} | "
                f"R {avg_r:>8.1f} | "
                f"Len {avg_len:>6.0f} | "
                f"MaxPhase {phase_names.get(max_phase, '?'):>10s} | "
                f"PL {metrics['policy_loss']:.4f} | "
                f"VL {metrics['value_loss']:.4f} | "
                f"Ent {metrics['entropy']:.3f} | "
                f"FPS {fps:.0f}"
            )
            if not args.test_controller:
                wandb.log({
                    "reward/mean": avg_r,
                    "reward/episode_length": avg_len,
                    "reward/max_phase": max_phase,
                    "reward/total_episodes": total_episodes,
                    "loss/policy": metrics['policy_loss'],
                    "loss/value": metrics['value_loss'],
                    "loss/entropy": metrics['entropy'],
                    "perf/fps": fps,
                    "perf/total_steps": total_steps,
                }, step=total_steps)

        # Save
        if n_updates % config.save_interval == 0 and not args.test_controller:
            ckpt_path = Path(config.checkpoint_dir) / f"step_{total_steps:08d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_steps": total_steps,
                "n_updates": n_updates,
                "avg_reward": avg_r if 'avg_r' in dir() else 0,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    if not args.test_controller:
        final_path = Path(config.checkpoint_dir) / "final.pt"
        torch.save({
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "total_steps": total_steps,
        }, final_path)
        print(f"\nDone. Final: {final_path}")
    else:
        print(f"\nDone. Test controller, {total_steps} steps")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

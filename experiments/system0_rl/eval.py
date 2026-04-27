#!/usr/bin/env python3
"""
System 0 — Blind Tactile Grasping RL Evaluation (with rendering).

Runs a trained checkpoint (or fresh random policy) in simulation with
rendering enabled so you can visually inspect the learned behaviour.

Usage
─────
  # Trained checkpoint (copy from remote first)
  python experiments/system0_rl/eval.py --checkpoint experiments/system0_rl/checkpoints/step_XXXXXXXXXX.pt

  # Fresh random policy (observe scripted arm + random fingers)
  python experiments/system0_rl/eval.py --num_envs 1

  # More envs side by side
  python experiments/system0_rl/eval.py --checkpoint path/to/ckpt.pt --num_envs 4
"""

import os
import sys
import argparse
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PROJECT_ROOT"] = project_root
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="System 0 Eval with rendering")
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--checkpoint",   type=str, default=None)
parser.add_argument("--max_episodes", type=int, default=200,
                    help="Stop after this many completed episodes (per env).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from experiments.system0_rl.config      import TrainConfig
from experiments.system0_rl.system0_moe import System0Config, System0PPOWrapper
import experiments.system0_rl.rewards as _rewards_mod
from experiments.system0_rl.rewards     import (
    compute_reward_blind, is_lift_success, set_contact_baseline, BLOCK_INIT_Z,
    _R_ALL,
)
from tasks.common_observations.tactile_state import (
    get_tactile_obs, get_tactile_obs_extended, reset_tactile_state,
)

RIGHT_ARM_CONTROLLABLE_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

RIGHT_HAND_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]


def build_joint_index_maps(env):
    joint_names = env.scene["robot"].data.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    sim_arm = [name_to_idx[n] for n in RIGHT_ARM_CONTROLLABLE_NAMES if n in name_to_idx]
    if len(sim_arm) != 5:
        missing = [n for n in RIGHT_ARM_CONTROLLABLE_NAMES if n not in name_to_idx]
        raise RuntimeError(f"Missing arm joints: {missing}")
    sim_hand = [name_to_idx[n] for n in RIGHT_HAND_NAMES if n in name_to_idx]
    if len(sim_hand) != 7:
        missing = [n for n in RIGHT_HAND_NAMES if n not in name_to_idx]
        raise RuntimeError(f"Missing right-hand joints: {missing}")
    print(f"[IndexMap] arm indices: {sim_arm}")
    print(f"[IndexMap] right hand indices: {sim_hand}")
    return sim_arm, sim_hand


def build_obs_batch(env, sim_arm: list, sim_hand: list, device) -> torch.Tensor:
    robot    = env.scene["robot"]
    arm_pos  = robot.data.joint_pos[:, sim_arm].to(device)    # (N, 5)
    arm_vel  = robot.data.joint_vel[:, sim_arm].to(device)    # (N, 5)
    hand_pos = robot.data.joint_pos[:, sim_hand].to(device)   # (N, 7)
    hand_vel = robot.data.joint_vel[:, sim_hand].to(device)   # (N, 7)
    try:
        tactile = get_tactile_obs_extended(env).to(device)
    except Exception:
        tactile = torch.zeros(hand_pos.shape[0], 72, device=device)
    try:
        torques = robot.data.applied_torque[:, sim_hand].to(device)
    except (AttributeError, IndexError):
        torques = torch.zeros_like(hand_pos)
    obs = torch.cat([arm_pos, arm_vel, hand_pos, hand_vel, tactile, torques], dim=1)  # (N, 103)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def encode_intent(stage: int, N: int, intent_dim: int, device) -> torch.Tensor:
    intent = torch.zeros(N, intent_dim, device=device)
    intent[:, min(stage, 3)] = 1.0
    return intent


def main():
    config = TrainConfig(num_envs=args.num_envs)
    device = torch.device(getattr(args, "device", "cuda:0"))
    N = args.num_envs

    from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = N
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"Env: {N} envs | action_dim={env.action_manager.total_action_dim}")

    sim_arm, sim_hand = build_joint_index_maps(env)

    action_dim = config.arm_dim + config.joint_dim  # 12
    moe_cfg = System0Config(
        arm_dim     = config.arm_dim,
        joint_dim   = config.joint_dim,
        vel_dim     = config.vel_dim,
        tactile_dim = config.tactile_dim,
        torque_dim  = config.torque_dim,
        target_dim  = config.target_dim,
        intent_dim  = config.intent_dim,
        hidden_dim  = config.hidden_dim,
        n_experts   = config.n_experts,
        top_k       = config.top_k,
        action_dim  = action_dim,
    )
    policy = System0PPOWrapper(moe_cfg).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        try:
            policy.load_state_dict(ckpt["policy"])
            print(f"Loaded checkpoint: step {ckpt.get('total_steps', 0):,}")
        except RuntimeError as e:
            # Shape mismatch (e.g. tactile dim change 64→72 between runs)
            print(f"WARNING: checkpoint shape mismatch — running fresh random policy")
            print(f"  {str(e).splitlines()[0]}")
    else:
        if args.checkpoint:
            print(f"WARNING: checkpoint not found: {args.checkpoint}")
        print("Running fresh random policy (no checkpoint)")

    policy.eval()

    # ── Init ────────────────────────────────────────────────────────────────
    env.reset()
    reset_tactile_state()
    simulation_app.update()

    n_act     = env.action_manager.total_action_dim
    zero_act  = torch.zeros(N, n_act, device=device)
    for _ in range(5):
        env.step(zero_act)
    set_contact_baseline(env, device)
    print("Contact baseline tared.")

    intent         = encode_intent(0, N, config.intent_dim, device)
    coarse_targets = torch.zeros(N, config.target_dim, device=device)   # 12D
    block_init_z   = torch.full((N,), BLOCK_INIT_Z, device=device)
    prev_hand_vel  = torch.zeros(N, 7, device=device)   # finger vels for smoothness
    ep_lifts       = torch.zeros(N, dtype=torch.bool, device=device)
    ep_rewards     = torch.zeros(N, device=device)
    ep_count       = 0
    step_count     = 0
    total_lifts    = 0

    print("\nEval running. Diagnostics every 20 steps.\n")
    print(f"{'step':>6}  {'block_z':>8}  {'thumb_f':>8}  {'opp_f':>8}  {'r':>7}  {'ep':>5}")
    print("-" * 55)

    while simulation_app.is_running():
        obs_batch        = build_obs_batch(env, sim_arm, sim_hand, device)   # (N, 103)
        obs_with_targets = torch.cat([obs_batch, coarse_targets], dim=1)     # (N, 115)

        with torch.no_grad():
            raw_delta, _, _ = policy.act(obs_with_targets, intent)

        # Deterministic eval: tanh-bounded 12D action, no OU noise
        action = torch.tanh(raw_delta) * config.delta_max
        _, _, terminated, truncated, _ = env.step(action)
        simulation_app.update()

        cur_hand_vel  = env.scene["robot"].data.joint_vel[:, sim_hand].to(device)
        rewards       = compute_reward_blind(env, prev_hand_vel, cur_hand_vel, device,
                                             block_init_z=block_init_z)
        prev_hand_vel = cur_hand_vel.detach().clone()
        ep_rewards   += rewards

        ep_lifts |= is_lift_success(env, device, block_init_z=block_init_z)

        if step_count % 20 == 0:
            # Use same force computation as rewards.py: differential (raw - baseline)
            raw = get_tactile_obs(env).to(device)          # (N, 18) scalar forces per pad
            baseline = _rewards_mod._CONTACT_BASELINE
            f = (raw - baseline).clamp(min=0.0) if baseline is not None else raw
            f_right  = f[:, _R_ALL]                        # (N, 9) right-hand pads
            # [3:5]=thumb_0/1  [5:9]=middle_0/1+index_0/1
            thumb_f  = f_right[:, 3:5].sum(dim=1).mean().item()
            opp_f    = f_right[:, 5:9].sum(dim=1).mean().item()
            block_z  = env.scene["block"].data.root_pos_w[:, 2].mean().item()
            r_mean   = rewards.mean().item()
            print(f"{step_count:6d}  {block_z:8.4f}  {thumb_f:8.3f}  {opp_f:8.3f}  {r_mean:7.3f}  {ep_count:5d}")

        just_reset = (terminated | truncated).to(device)
        if just_reset.any():
            reset_ids = just_reset.nonzero(as_tuple=False).squeeze(1)
            for i in reset_ids.tolist():
                lifted = ep_lifts[i].item()
                if lifted:
                    total_lifts += 1
                ep_count += 1
                r_ep = ep_rewards[i].item()
                print(f"  [Episode {ep_count:4d}] env={i}  reward={r_ep:7.1f}  "
                      f"LIFTED={'YES ✓' if lifted else 'no'}  "
                      f"lift_rate={total_lifts/ep_count:.1%}")
                ep_rewards[i] = 0.0
                ep_lifts[i] = False
            reset_tactile_state(reset_ids, N, device)
            prev_hand_vel[reset_ids] = 0.0

            if ep_count >= args.max_episodes * N:
                break

        step_count += 1

    print(f"\nDone. {ep_count} episodes | lift_rate={total_lifts/max(ep_count,1):.1%}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

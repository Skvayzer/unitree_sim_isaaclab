#!/usr/bin/env python3
"""
System 0 Skill Training Script.

Supports Skill 1 (grasp) and Skill 2 (hold while moving).
Uses RSL-RL PPO via Isaac Lab wrapper, falls back to custom PPO.

Usage:
    cd ~/unitree_sim_isaaclab

    # Skill 1 — Grasp (RSL-RL):
    python experiments/system0_skills/train.py --skill grasp --num_envs 512 --headless

    # Skill 2 — Hold (custom PPO, initialized from Skill 1):
    python experiments/system0_skills/train.py --skill hold --num_envs 512 --headless \
        --load_checkpoint logs/system0_grasp/model_999.pt

    # On remote PC (headless, 1024 envs):
    nohup python experiments/system0_skills/train.py --skill hold --num_envs 1024 --headless \
        --load_checkpoint logs/system0_grasp/model_999.pt > train_hold.log 2>&1 &
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

parser = argparse.ArgumentParser(description="System 0 Skill Training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--skill", type=str, default="grasp",
                    choices=["grasp", "grasp_v2", "hold", "release", "moe"],
                    help="Which skill to train (grasp_v2 = table grasp with visible finger motion)")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max PPO iterations")
parser.add_argument("--load_checkpoint", type=str, default=None,
                    help="Path to checkpoint to initialize from (supports RSL-RL and custom formats)")
parser.add_argument("--use_custom_ppo", action="store_true", help="Force custom PPO instead of RSL-RL")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
parser.add_argument("--grasp_ckpt", type=str, default="logs/system0_grasp/model_999.pt",
                    help="Grasp skill checkpoint (for MoE or hold/release init)")
parser.add_argument("--hold_ckpt", type=str, default="logs/system0_hold/best_model.pt",
                    help="Hold skill checkpoint (for MoE)")
parser.add_argument("--release_ckpt", type=str, default="logs/system0_release/best_model.pt",
                    help="Release skill checkpoint (for MoE)")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import everything else ---
import torch
import torch.nn as nn
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from experiments.system0_skills.config import System0Config
from experiments.system0_skills.grasp_env import GraspEnvCfg
from experiments.system0_skills.policy import System0Actor, System0Critic

CFG = System0Config()

# Skill-specific obs dims
SKILL_OBS_DIM = {
    "grasp": 21,      # finger_pos(7) + finger_vel(7) + contact(3) + target(3) + grasped(1)
    "grasp_v2": 21,   # same obs, different env (table + open fingers + large action_scale)
    "hold": 28,       # grasp(21) + arm_vel(7)
    "release": 22,    # grasp(21) + height(1)
    "moe": 28,        # unified obs space
}


def load_checkpoint_into_actor_critic(ckpt_path: str, actor: System0Actor, critic: System0Critic,
                                      device: torch.device):
    """Load weights from either RSL-RL or custom PPO checkpoint.

    Handles dimension mismatch gracefully (Skill 1 → Skill 2: obs_dim 21 → 28).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        # RSL-RL format: keys like actor.0.weight, critic.0.weight, std
        sd = ckpt["model_state_dict"]
        actor_sd = {}
        critic_sd = {}
        for k, v in sd.items():
            if k == "std":
                actor_sd["log_std"] = v.log()  # RSL-RL stores std, we store log_std
            elif k.startswith("actor."):
                # actor.0.weight → net.0.weight
                new_k = k.replace("actor.", "net.", 1)
                actor_sd[new_k] = v
            elif k.startswith("critic."):
                new_k = k.replace("critic.", "net.", 1)
                critic_sd[new_k] = v
    elif "actor" in ckpt:
        # Custom PPO format
        actor_sd = ckpt["actor"]
        critic_sd = ckpt["critic"]
    else:
        print(f"[WARN] Unknown checkpoint format: {list(ckpt.keys())}. Skipping load.")
        return 0

    # Load actor weights, handling dimension mismatch for first layer
    loaded_actor = 0
    current_actor_sd = actor.state_dict()
    for k in current_actor_sd:
        if k in actor_sd:
            src = actor_sd[k]
            dst = current_actor_sd[k]
            if src.shape == dst.shape:
                current_actor_sd[k] = src
                loaded_actor += 1
            elif k == "net.0.weight" and src.shape[1] < dst.shape[1]:
                # Input dim grew (e.g., 21→28). Copy old weights, zero-init new dims
                current_actor_sd[k][:, :src.shape[1]] = src
                current_actor_sd[k][:, src.shape[1]:] = 0.0
                loaded_actor += 1
                print(f"  [LOAD] {k}: expanded input {src.shape[1]}→{dst.shape[1]}")
            else:
                print(f"  [SKIP] {k}: shape mismatch {src.shape} vs {dst.shape}")
    actor.load_state_dict(current_actor_sd)

    # Load critic weights similarly
    loaded_critic = 0
    current_critic_sd = critic.state_dict()
    for k in current_critic_sd:
        if k in critic_sd:
            src = critic_sd[k]
            dst = current_critic_sd[k]
            if src.shape == dst.shape:
                current_critic_sd[k] = src
                loaded_critic += 1
            elif k == "net.0.weight" and src.shape[1] < dst.shape[1]:
                current_critic_sd[k][:, :src.shape[1]] = src
                current_critic_sd[k][:, src.shape[1]:] = 0.0
                loaded_critic += 1
                print(f"  [LOAD] {k}: expanded input {src.shape[1]}→{dst.shape[1]}")
            else:
                print(f"  [SKIP] {k}: shape mismatch {src.shape} vs {dst.shape}")
    critic.load_state_dict(current_critic_sd)

    total = loaded_actor + loaded_critic
    print(f"  Loaded {loaded_actor} actor + {loaded_critic} critic params from {ckpt_path}")
    return total


def try_rsl_rl_training(env_cfg, num_envs: int, max_iterations: int, skill: str):
    """Attempt training with RSL-RL library via Isaac Lab wrapper."""
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
                                     RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
    from rsl_rl.runners import OnPolicyRunner

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    log_dir = f"logs/system0_{skill}"
    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda:0",
        num_steps_per_env=CFG.steps_per_rollout,
        max_iterations=max_iterations,
        empirical_normalization=False,
        save_interval=500,
        experiment_name=f"system0_{skill}",
        logger="wandb" if not args.no_wandb else "tensorboard",
        wandb_project=CFG.wandb_project,
        policy=RslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=[CFG.hidden_dim, CFG.hidden_dim],
            critic_hidden_dims=[CFG.hidden_dim, CFG.hidden_dim],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=CFG.value_coeff,
            use_clipped_value_loss=True,
            clip_param=CFG.clip_eps,
            entropy_coef=CFG.entropy_coeff,
            num_learning_epochs=CFG.ppo_epochs,
            num_mini_batches=CFG.mini_batches,
            learning_rate=CFG.lr,
            schedule="adaptive",
            gamma=CFG.gamma,
            lam=CFG.gae_lambda,
            desired_kl=0.01,
            max_grad_norm=CFG.max_grad_norm,
        ),
    )

    runner = OnPolicyRunner(wrapped_env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args.load_checkpoint:
        runner.load(args.load_checkpoint)
        print(f"Loaded RSL-RL checkpoint: {args.load_checkpoint}")

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    env.close()
    return True


def custom_ppo_training(env_cfg, num_envs: int, max_iterations: int, skill: str,
                        arm_perturbation=None, arm_lift_targets=None):
    """Custom PPO training loop. Supports arm perturbation for Skill 2."""
    # Optional wandb
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(
                project=CFG.wandb_project,
                entity=CFG.wandb_entity,
                name=f"skill_{skill}_{num_envs}envs",
                config={
                    "skill": skill,
                    "num_envs": num_envs,
                    "lr": CFG.lr,
                    "hidden_dim": CFG.hidden_dim,
                    "action_dim": CFG.action_dim,
                    "obs_dim": SKILL_OBS_DIM[skill],
                    "max_iterations": max_iterations,
                    "steps_per_rollout": CFG.steps_per_rollout,
                    "gamma": CFG.gamma,
                    "gae_lambda": CFG.gae_lambda,
                    "clip_eps": CFG.clip_eps,
                    "entropy_coeff": CFG.entropy_coeff,
                    "force_penalty_weight": CFG.force_penalty_weight,
                    "load_checkpoint": args.load_checkpoint or "none",
                },
            )
            wandb = _wandb
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}. Continuing without wandb.")

    env = ManagerBasedRLEnv(cfg=env_cfg)

    device = env.device
    obs_dim = SKILL_OBS_DIM[skill]
    act_dim = CFG.action_dim

    actor = System0Actor(obs_dim, act_dim, CFG.hidden_dim).to(device)
    critic = System0Critic(obs_dim, CFG.hidden_dim).to(device)

    # Load checkpoint if provided
    if args.load_checkpoint:
        ckpt_path = args.load_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
        load_checkpoint_into_actor_critic(ckpt_path, actor, critic, device)
        # Reset log_std to reasonable exploration level after loading
        # (RSL-RL uses very high std ~9.0; we want ~0.5 for fine-tuning)
        with torch.no_grad():
            actor.log_std.fill_(-0.7)  # exp(-0.7) ≈ 0.5
        print(f"  Reset log_std to {actor.log_std.data.exp().mean().item():.3f}")

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=CFG.lr,
    )

    # Arm perturbation for Skill 2
    if arm_perturbation is not None:
        arm_perturbation = arm_perturbation.__class__(num_envs, device)

    prev_action = torch.zeros(num_envs, act_dim, device=device)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].nan_to_num(0.0).clamp(-10.0, 10.0)

    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", f"system0_{skill}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"System 0 {skill.upper()} Training — Custom PPO")
    print(f"  num_envs={num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  iterations={max_iterations}, steps/rollout={CFG.steps_per_rollout}")
    print(f"  force_penalty_weight={CFG.force_penalty_weight}")
    print(f"  checkpoint={'loaded' if args.load_checkpoint else 'none'}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()

        rollout_obs = []
        rollout_act = []
        rollout_logp = []
        rollout_rew = []
        rollout_done = []
        rollout_val = []

        for step in range(CFG.steps_per_rollout):
            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(1e-6, 10.0)
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().clamp(-1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = critic(obs)

            smoothed_action = 0.7 * raw_action + 0.3 * prev_action
            prev_action = smoothed_action.clone()

            rollout_obs.append(obs)
            rollout_act.append(raw_action)
            rollout_logp.append(log_prob)
            rollout_val.append(value)

            # Apply arm perturbation for Skill 2 (hold)
            if arm_perturbation is not None:
                arm_offsets = arm_perturbation.get_targets()  # [num_envs, 7]
                robot = env.scene["robot"]
                arm_idx_tensor = torch.tensor(CFG.right_arm_indices, device=device)
                base_pos = robot.data.default_joint_pos[:, arm_idx_tensor]
                target_pos = base_pos + arm_offsets
                robot.data.joint_pos_target[:, arm_idx_tensor] = target_pos

            # Arm lift for grasp_v2: lift arm mid-episode so block must come with hand
            if arm_lift_targets is not None:
                robot = env.scene["robot"]
                joint_names = list(robot.data.joint_names)
                ep_step = env.episode_length_buf  # [num_envs]
                lift_start = 12   # start lifting at step 12 (of 24-step rollout)
                for jname, reach_val, lift_val in arm_lift_targets:
                    jidx = joint_names.index(jname)
                    # t=0 at lift_start, t=1 at end of rollout
                    t = ((ep_step.float() - lift_start) / max(CFG.steps_per_rollout - lift_start, 1)).clamp(0, 1)
                    t = t * t * (3 - 2 * t)  # smoothstep
                    target = reach_val + (lift_val - reach_val) * t
                    robot.data.joint_pos_target[:, jidx] = target

            obs_dict, reward, terminated, truncated, info = env.step(smoothed_action)
            obs = obs_dict["policy"]
            # Clamp obs to prevent NaN propagation
            obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)
            done = terminated | truncated

            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            if done.any():
                prev_action[done] = 0.0
                if arm_perturbation is not None:
                    arm_perturbation.reset(done.nonzero(as_tuple=False).squeeze(-1))

        # Stack rollout
        rollout_obs = torch.stack(rollout_obs)
        rollout_act = torch.stack(rollout_act)
        rollout_logp = torch.stack(rollout_logp)
        rollout_rew = torch.stack(rollout_rew)
        rollout_done = torch.stack(rollout_done)
        rollout_val = torch.stack(rollout_val)

        # GAE
        with torch.no_grad():
            last_value = critic(obs)

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
                mb_obs = flat_obs[idx].nan_to_num(0.0).clamp(-10.0, 10.0)
                mb_act = flat_act[idx]
                mb_old_logp = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                mean, std = actor(mb_obs)
                # Clamp mean/std to prevent NaN in distribution
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(1e-6, 10.0)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CFG.clip_eps, 1 + CFG.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
                loss = policy_loss + CFG.value_coeff * value_loss - CFG.entropy_coeff * entropy

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
        avg_ep_len = (~rollout_done.bool()).float().sum(dim=0).mean().item()
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        with torch.no_grad():
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
            force_mags = right_forces.norm(dim=-1)
            mean_force = force_mags.mean().item()
            fingers_touching = (force_mags > 0.1).float().sum(dim=-1).mean().item()
            block_z = env.scene["block"].data.root_pos_w[:, 2]
            grasped_frac = ((block_z > CFG.block_initial_z - 0.05) &
                           ((force_mags > 0.1).sum(dim=-1) >= 2)).float().mean().item()

        if iteration % 10 == 0 or iteration < 5:
            print(
                f"[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} (min={min_reward:+.3f} max={max_reward:+.3f}) | "
                f"p_loss={total_policy_loss/n_updates:.4f} v_loss={total_value_loss/n_updates:.4f} | "
                f"ent={total_entropy/n_updates:.3f} | "
                f"grasp={grasped_frac:.2f} touch={fingers_touching:.1f} force={mean_force:.2f} | "
                f"fps={fps:.0f}"
            )

        if wandb is not None:
            wandb.log({
                "reward/mean": avg_reward,
                "reward/min": min_reward,
                "reward/max": max_reward,
                "episode/length_mean": avg_ep_len,
                "episode/success_rate": grasped_frac,
                "loss/policy": total_policy_loss / n_updates,
                "loss/value": total_value_loss / n_updates,
                "loss/entropy": total_entropy / n_updates,
                "perf/fps": fps,
                "perf/num_envs": num_envs,
                "contact/mean_force": mean_force,
                "contact/num_fingers_touching": fingers_touching,
                "iteration": iteration,
            }, step=total_steps)

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": best_reward,
                "skill": skill,
                "obs_dim": obs_dim,
            }, os.path.join(save_dir, "best_model.pt"))

        if (iteration + 1) % 1000 == 0:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": avg_reward,
                "skill": skill,
                "obs_dim": obs_dim,
            }, os.path.join(save_dir, "latest_checkpoint.pt"))
            print(f"  -> Saved checkpoint at iteration {iteration+1}")

    print(f"\nTraining complete. Best reward: {best_reward:.3f}")
    if wandb is not None:
        wandb.finish()
    env.close()


def moe_training(num_envs: int, max_iterations: int):
    """Train the MoE router with frozen skill experts in a multi-task env."""
    from experiments.system0_skills.multitask_env import (
        MultiTaskEnvCfg, set_task_state, TASK_GRASP, TASK_HOLD, TASK_RELEASE
    )
    from experiments.system0_skills.moe_policy import System0MoEActor, System0MoECritic

    # Optional wandb
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(
                project=CFG.wandb_project,
                entity=CFG.wandb_entity,
                name=f"moe_router_{num_envs}envs",
                config={
                    "skill": "moe",
                    "num_envs": num_envs,
                    "lr": CFG.lr,
                    "max_iterations": max_iterations,
                    "grasp_ckpt": args.grasp_ckpt,
                    "hold_ckpt": args.hold_ckpt,
                    "release_ckpt": args.release_ckpt,
                },
            )
            wandb = _wandb
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}. Continuing without wandb.")

    env_cfg = MultiTaskEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # Create MoE actor + critic
    actor = System0MoEActor(hidden=CFG.hidden_dim).to(device)
    critic = System0MoECritic(obs_dim=28, hidden=CFG.hidden_dim).to(device)

    # Load skill experts
    def resolve_path(p):
        if not os.path.isabs(p):
            return os.path.join(_PROJECT_ROOT, p)
        return p

    print("\nLoading skill experts into MoE...")
    actor.load_skill_experts(
        grasp_ckpt=resolve_path(args.grasp_ckpt),
        hold_ckpt=resolve_path(args.hold_ckpt),
        release_ckpt=resolve_path(args.release_ckpt),
        device=device,
    )

    # Freeze experts — only train router + encoder
    actor.freeze_experts()

    # Only optimize trainable params
    trainable_params = [p for p in actor.parameters() if p.requires_grad] + list(critic.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=CFG.lr)

    # Task assignment per env (random, reassigned on reset)
    task_ids = torch.randint(0, 3, (num_envs,), device=device)
    arm_vel_noise = torch.zeros(num_envs, 7, device=device)

    def randomize_tasks(env_ids: torch.Tensor):
        """Assign random tasks to reset environments."""
        n = len(env_ids)
        task_ids[env_ids] = torch.randint(0, 3, (n,), device=device)
        # Generate arm vel noise for hold tasks
        hold_mask = task_ids[env_ids] == TASK_HOLD
        arm_vel_noise[env_ids] = 0.0
        if hold_mask.any():
            hold_ids = env_ids[hold_mask]
            arm_vel_noise[hold_ids] = torch.randn(len(hold_ids), 7, device=device) * 0.5

    # Initialize
    randomize_tasks(torch.arange(num_envs, device=device))
    set_task_state(task_ids, arm_vel_noise)

    prev_action = torch.zeros(num_envs, 7, device=device)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].nan_to_num(0.0).clamp(-10.0, 10.0)

    total_steps = 0
    best_reward = float("-inf")
    save_dir = os.path.join(_PROJECT_ROOT, "logs", "system0_moe")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"System 0 MoE Router Training")
    print(f"  num_envs={num_envs}, iterations={max_iterations}")
    print(f"  trainable params: {sum(p.numel() for p in trainable_params)}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        iter_start = time.time()

        rollout_obs = []
        rollout_act = []
        rollout_logp = []
        rollout_rew = []
        rollout_done = []
        rollout_val = []

        for step in range(CFG.steps_per_rollout):
            with torch.no_grad():
                mean, std = actor(obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(1e-6, 10.0)
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().clamp(-1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = critic(obs)

            smoothed_action = 0.7 * raw_action + 0.3 * prev_action
            prev_action = smoothed_action.clone()

            rollout_obs.append(obs)
            rollout_act.append(raw_action)
            rollout_logp.append(log_prob)
            rollout_val.append(value)

            # Update arm vel noise for hold tasks (slowly varying)
            hold_mask = task_ids == TASK_HOLD
            if hold_mask.any():
                arm_vel_noise[hold_mask] += torch.randn_like(arm_vel_noise[hold_mask]) * 0.05
                arm_vel_noise.clamp_(-2.0, 2.0)
            set_task_state(task_ids, arm_vel_noise)

            obs_dict, reward, terminated, truncated, info = env.step(smoothed_action)
            obs = obs_dict["policy"].nan_to_num(0.0).clamp(-10.0, 10.0)
            done = terminated | truncated

            rollout_rew.append(reward)
            rollout_done.append(done.float())
            total_steps += num_envs

            if done.any():
                prev_action[done] = 0.0
                randomize_tasks(done.nonzero(as_tuple=False).squeeze(-1))
                set_task_state(task_ids, arm_vel_noise)

        # Stack rollout
        rollout_obs = torch.stack(rollout_obs)
        rollout_act = torch.stack(rollout_act)
        rollout_logp = torch.stack(rollout_logp)
        rollout_rew = torch.stack(rollout_rew)
        rollout_done = torch.stack(rollout_done)
        rollout_val = torch.stack(rollout_val)

        # GAE
        with torch.no_grad():
            last_value = critic(obs)
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
        obs_dim = 28
        act_dim = 7
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
                mb_obs = flat_obs[idx].nan_to_num(0.0).clamp(-10.0, 10.0)
                mb_act = flat_act[idx]
                mb_old_logp = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                mean, std = actor(mb_obs)
                mean = mean.nan_to_num(0.0).clamp(-5.0, 5.0)
                std = std.clamp(1e-6, 10.0)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = critic(mb_obs)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CFG.clip_eps, 1 + CFG.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
                loss = policy_loss + CFG.value_coeff * value_loss - CFG.entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, CFG.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Logging
        avg_reward = rollout_rew.mean().item()
        fps = (CFG.steps_per_rollout * num_envs) / (time.time() - iter_start)

        # Router weight analysis
        with torch.no_grad():
            router_w = actor.get_router_weights(obs)
            mean_router = router_w.mean(dim=0)  # [3]
            # Per-task router weights
            task_names = ["grasp", "hold", "release"]
            per_task_router = {}
            for t_id, t_name in enumerate(task_names):
                mask = task_ids == t_id
                if mask.any():
                    per_task_router[t_name] = router_w[mask].mean(dim=0)

        if iteration % 10 == 0 or iteration < 5:
            router_str = " ".join(f"{w:.2f}" for w in mean_router.tolist())
            print(
                f"[{iteration:4d}/{max_iterations}] "
                f"rew={avg_reward:+.3f} | "
                f"p_loss={total_policy_loss/n_updates:.4f} v_loss={total_value_loss/n_updates:.4f} | "
                f"router=[{router_str}] | "
                f"fps={fps:.0f}"
            )
            # Print per-task router weights
            for t_name, rw in per_task_router.items():
                rw_str = " ".join(f"{w:.2f}" for w in rw.tolist())
                print(f"        {t_name:>8s} → [{rw_str}]")

        if wandb is not None:
            log_dict = {
                "reward/mean": avg_reward,
                "loss/policy": total_policy_loss / n_updates,
                "loss/value": total_value_loss / n_updates,
                "loss/entropy": total_entropy / n_updates,
                "perf/fps": fps,
                "router/grasp_weight": mean_router[0].item(),
                "router/hold_weight": mean_router[1].item(),
                "router/release_weight": mean_router[2].item(),
                "iteration": iteration,
            }
            for t_name, rw in per_task_router.items():
                for i, expert_name in enumerate(task_names):
                    log_dict[f"router/{t_name}_to_{expert_name}"] = rw[i].item()
            wandb.log(log_dict, step=total_steps)

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": best_reward,
            }, os.path.join(save_dir, "best_model.pt"))

        if (iteration + 1) % 500 == 0:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "iteration": iteration,
                "reward": avg_reward,
            }, os.path.join(save_dir, "latest_checkpoint.pt"))
            print(f"  -> Saved checkpoint at iteration {iteration+1}")

    print(f"\nMoE training complete. Best reward: {best_reward:.3f}")
    if wandb is not None:
        wandb.finish()
    env.close()


def main():
    skill = args.skill
    num_envs = args.num_envs
    max_iterations = args.max_iterations

    arm_lift_data = None

    if skill == "grasp":
        env_cfg = GraspEnvCfg()
        env_cfg.scene.num_envs = num_envs
        arm_perturbation = None

    elif skill == "grasp_v2":
        from experiments.system0_skills.grasp_v2_env import (
            GraspV2EnvCfg, ARM_REACH_JOINTS, ARM_LIFT_JOINTS
        )
        env_cfg = GraspV2EnvCfg()
        env_cfg.scene.num_envs = num_envs
        arm_perturbation = None
        # Set up arm lift targets: [(joint_name, reach_val, lift_val), ...]
        arm_lift_data = [
            (jname, ARM_REACH_JOINTS[jname], ARM_LIFT_JOINTS[jname])
            for jname in ARM_REACH_JOINTS
        ]

    elif skill == "hold":
        from experiments.system0_skills.hold_env import HoldEnvCfg, ArmPerturbation
        env_cfg = HoldEnvCfg()
        env_cfg.scene.num_envs = num_envs
        arm_perturbation = ArmPerturbation(1, torch.device("cpu"))  # placeholder

    elif skill == "release":
        from experiments.system0_skills.release_env import ReleaseEnvCfg
        env_cfg = ReleaseEnvCfg()
        env_cfg.scene.num_envs = num_envs
        arm_perturbation = None

    elif skill == "moe":
        moe_training(num_envs, max_iterations)
        return

    # For grasp, try RSL-RL first; for hold/release, use custom PPO
    if skill == "grasp" and not args.use_custom_ppo:
        try:
            print("Attempting RSL-RL training...")
            try_rsl_rl_training(env_cfg, num_envs, max_iterations, skill)
            return
        except Exception as e:
            print(f"RSL-RL failed: {e}")
            print("Falling back to custom PPO...")

    custom_ppo_training(env_cfg, num_envs, max_iterations, skill,
                        arm_perturbation=arm_perturbation if skill == "hold" else None,
                        arm_lift_targets=arm_lift_data if skill == "grasp_v2" else None)


if __name__ == "__main__":
    main()
    simulation_app.close()

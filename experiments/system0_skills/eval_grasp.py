#!/usr/bin/env python3
"""
System 0 Grasp Evaluation — Visual Demo.

Loads the trained grasp policy (RSL-RL checkpoint) and runs it in the
grasp environment WITH the viewer enabled so you can see the robot
grasping the block.

Usage:
    cd ~/unitree_sim_isaaclab
    python experiments/system0_skills/eval_grasp.py --num_envs 4

    # With MoE policy instead of standalone grasp:
    python experiments/system0_skills/eval_grasp.py --num_envs 4 --moe
"""

import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="System 0 Grasp Eval")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--moe", action="store_true", help="Use MoE policy instead of standalone grasp")
parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
parser.add_argument("--grasp_ckpt", type=str, default="logs/system0_grasp/model_999.pt")
parser.add_argument("--moe_ckpt", type=str, default="logs/system0_moe/best_model.pt")
parser.add_argument("--hold_ckpt", type=str, default="logs/system0_hold/best_model.pt")
parser.add_argument("--release_ckpt", type=str, default="logs/system0_release/best_model.pt")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.grasp_env import GraspEnvCfg
from experiments.system0_skills.config import System0Config

CFG = System0Config()


def load_standalone_grasp(ckpt_path: str, device: torch.device):
    """Load the RSL-RL trained grasp actor."""

    class Actor(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ELU(),
                nn.Linear(hidden, hidden), nn.ELU(),
                nn.Linear(hidden, act_dim),
            )

        def forward(self, obs):
            return self.net(obs)

    actor = Actor(21, 7).to(device)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        actor_sd = {}
        for k, v in sd.items():
            if k.startswith("actor."):
                actor_sd[k.replace("actor.", "net.", 1)] = v
        actor.load_state_dict(actor_sd)
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])

    actor.eval()
    print(f"Loaded grasp actor from {ckpt_path}")
    return actor


def load_moe_policy(moe_ckpt: str, grasp_ckpt: str, hold_ckpt: str, release_ckpt: str,
                     device: torch.device):
    """Load the MoE policy with frozen experts."""
    from experiments.system0_skills.moe_policy import System0MoEActor

    actor = System0MoEActor(hidden=128, temperature=0.3).to(device)

    # Load skill experts
    def resolve(p):
        return os.path.join(_PROJECT_ROOT, p) if not os.path.isabs(p) else p

    actor.load_skill_experts(resolve(grasp_ckpt), resolve(hold_ckpt), resolve(release_ckpt), device)

    # Load trained router weights
    moe_path = resolve(moe_ckpt)
    moe_state = torch.load(moe_path, map_location=device, weights_only=True)
    if "actor" in moe_state:
        # Only load router + log_std (experts are already loaded from skill checkpoints)
        saved_sd = moe_state["actor"]
        current_sd = actor.state_dict()
        for k in current_sd:
            if k in saved_sd and ("router" in k or k == "log_std"):
                current_sd[k] = saved_sd[k]
        actor.load_state_dict(current_sd)
        print(f"Loaded MoE router from {moe_path}")

    actor.eval()
    return actor


def main():
    num_envs = args.num_envs
    max_steps = args.steps

    env_cfg = GraspEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # Load policy
    if args.moe:
        actor = load_moe_policy(args.moe_ckpt, args.grasp_ckpt, args.hold_ckpt,
                                args.release_ckpt, device)
        obs_dim = 28
        print("Using MoE policy")
    else:
        actor = load_standalone_grasp(args.grasp_ckpt, device)
        obs_dim = 21
        print("Using standalone grasp policy")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # For MoE, pad obs to 28D
    if args.moe and obs.shape[-1] < 28:
        pad = torch.zeros(num_envs, 28 - obs.shape[-1], device=device)
        obs = torch.cat([obs, pad], dim=-1)

    prev_action = torch.zeros(num_envs, 7, device=device)

    print(f"\n{'='*60}")
    print(f"System 0 Grasp Evaluation — {'MoE' if args.moe else 'Standalone'}")
    print(f"  num_envs={num_envs}, steps={max_steps}")
    print(f"  Close the viewer window to stop")
    print(f"{'='*60}\n")

    for step in range(max_steps):
        with torch.no_grad():
            if args.moe:
                mean, std = actor(obs)
                raw_action = mean.clamp(-1.0, 1.0)
            else:
                raw_action = actor(obs).clamp(-1.0, 1.0)

        action = 0.7 * raw_action + 0.3 * prev_action
        prev_action = action.clone()

        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        if args.moe and obs.shape[-1] < 28:
            pad = torch.zeros(num_envs, 28 - obs.shape[-1], device=device)
            obs = torch.cat([obs, pad], dim=-1)

        done = terminated | truncated
        if done.any():
            prev_action[done] = 0.0

        if step % 50 == 0:
            block_z = env.scene["block"].data.root_pos_w[:, 2]
            forces = env.scene["fingertip_contacts"].data.net_forces_w
            rf = forces[:, CFG.right_fingertip_contact_indices, :]
            fm = rf.norm(dim=-1)
            cc = (fm > 0.1).float().sum(dim=-1).mean().item()
            mean_force = fm.mean().item()

            if args.moe:
                rw = actor.get_router_weights(obs)
                router_str = f" router=[{rw[0,0]:.2f} {rw[0,1]:.2f} {rw[0,2]:.2f}]"
            else:
                router_str = ""

            print(
                f"[{step:4d}] block_z={block_z.mean():.4f} "
                f"contacts={cc:.1f} force={mean_force:.1f} "
                f"rew={reward.mean():.4f}{router_str}"
            )

    print("\nEvaluation complete. Block held at z={:.4f} with {:.1f} contacts.".format(
        env.scene["block"].data.root_pos_w[:, 2].mean().item(),
        (env.scene["fingertip_contacts"].data.net_forces_w[:, CFG.right_fingertip_contact_indices, :].norm(dim=-1) > 0.1).float().sum(dim=-1).mean().item()
    ))

    # Keep viewer alive — continue stepping until user closes window
    if not args.headless:
        print("Viewer is open. Close the window or press Ctrl+C to exit.")
        step = max_steps
        try:
            while simulation_app.is_running():
                with torch.no_grad():
                    if args.moe:
                        mean, std = actor(obs)
                        raw_action = mean.clamp(-1.0, 1.0)
                    else:
                        raw_action = actor(obs).clamp(-1.0, 1.0)
                action = 0.7 * raw_action + 0.3 * prev_action
                prev_action = action.clone()

                obs_dict, reward, terminated, truncated, info = env.step(action)
                obs = obs_dict["policy"]
                if args.moe and obs.shape[-1] < 28:
                    pad = torch.zeros(num_envs, 28 - obs.shape[-1], device=device)
                    obs = torch.cat([obs, pad], dim=-1)

                done = terminated | truncated
                if done.any():
                    prev_action[done] = 0.0

                step += 1
                if step % 200 == 0:
                    bz = env.scene["block"].data.root_pos_w[:, 2].mean().item()
                    print(f"[{step:5d}] block_z={bz:.4f}", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nStopped by user.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

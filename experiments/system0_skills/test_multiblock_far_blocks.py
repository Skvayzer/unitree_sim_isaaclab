#!/usr/bin/env python3
"""Test: Multi-block ENV with blocks 1,2 moved far away (30cm).
If block 0 lift recovers to ~80%, the drop is caused by physical interference
from adjacent blocks. If still ~8%, it's something else (physics solver, etc)."""

import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--grasp_checkpoint", type=str, required=True)
parser.add_argument("--release_checkpoint", type=str, required=True)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_config import BlockStackConfig
import experiments.system0_skills.multi_block_env as mb_env
from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()
device = "cuda:0"
num_envs = 16

# Load actors
ckpt = torch.load(args.grasp_checkpoint, map_location=device, weights_only=False)
grasp_actor = System0Actor(28, 7, 128).to(device)
grasp_actor.load_state_dict(ckpt["actor"])
grasp_actor.eval()
rckpt = torch.load(args.release_checkpoint, map_location=device, weights_only=False)
release_actor = System0Actor(28, 7, 128).to(device)
release_actor.load_state_dict(rckpt["actor"])
release_actor.eval()

# Disable block_fallen termination
mb_env._DISABLE_BLOCK_FALLEN = True

# Create multi-block env with blocks 1,2 moved FAR away
env_cfg = MultiBlockEnvCfg()
env_cfg.scene.num_envs = num_envs
env_cfg.episode_length_s = 10.0

# Move blocks 1,2 far away — 30cm in y from block 0
env_cfg.scene.block_1.init_state.pos = (0.321, -0.480, 0.819)
env_cfg.scene.block_2.init_state.pos = (0.321,  0.120, 0.819)
print(f"Block 0 pos: {env_cfg.scene.block_0.init_state.pos}")
print(f"Block 1 pos: {env_cfg.scene.block_1.init_state.pos} (moved far)")
print(f"Block 2 pos: {env_cfg.scene.block_2.init_state.pos} (moved far)")

env = ManagerBasedRLEnv(cfg=env_cfg)

arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
hand_idx = torch.tensor(CFG.right_hand_indices, device=device)

traj = ParameterizedArmTrajectory(CFG, device, num_envs)

def build_obs(env, hand_idx, phase_onehot):
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

prev_action = torch.zeros(num_envs, 7, device=device)
block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
total_episodes = 0
total_lifted = 0
num_target = 256

print(f"\nMulti-block ENV + FAR blocks (30cm spacing), block_0 at y=-0.180")
print(f"Envs: {num_envs}, target episodes: {num_target}\n")

for step_i in range(num_target * 500):
    robot = env.scene["robot"]
    ep_step = env.episode_length_buf

    arm_targets = traj.get_arm_targets(ep_step)
    robot.data.joint_pos_target[:, arm_idx] = arm_targets

    phase_ids = traj.get_phase_ids(ep_step)
    phase_onehot = traj.get_phase_onehot(ep_step)

    obs = build_obs(env, hand_idx, phase_onehot)

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

    block_pos = env.scene["block_0"].data.root_pos_w
    env_origins = env.scene.env_origins
    block_local = block_pos - env_origins
    block_z = block_local[:, 2]

    is_lift = (phase_ids == Phase.LIFT)
    just_lifted = is_lift & (block_z > CFG.block_initial_z + 0.03)
    block_was_lifted |= just_lifted

    if done.any():
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            total_episodes += 1
            total_lifted += int(block_was_lifted[i].item())

        if total_episodes % 32 == 0:
            lr = total_lifted / total_episodes
            print(f"  [{total_episodes}/{num_target}] lift={lr:.1%}")

        block_was_lifted[done] = False
        prev_action[done] = 0

        if total_episodes >= num_target:
            break

lr = total_lifted / max(total_episodes, 1)
print(f"\nRESULTS: {total_episodes} episodes, block_0 lift={lr:.1%}")
env.close()

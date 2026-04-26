#!/usr/bin/env python3
"""
Diagnostic: Run single-block eval pattern on multi_block_env.

Tests whether the multi_block_env itself causes the lift rate drop,
or if the state machine is the culprit.

Does NOT use BlockStackingStateMachine — uses the same raw loop
as single-block eval, but on multi_block_env.
"""

import os, sys, time, math

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
parser.add_argument("--num_envs", type=int, default=64)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()

num_envs = args.num_envs

# Create multi-block env
env_cfg = MultiBlockEnvCfg()
env_cfg.scene.num_envs = num_envs
env_cfg.episode_length_s = 30.0
env = ManagerBasedRLEnv(cfg=env_cfg)
device = env.device

# Load specialists
grasp_ckpt = torch.load(args.grasp_checkpoint, map_location=device, weights_only=False)
grasp_actor = System0Actor(28, 7, 128).to(device)
grasp_actor.load_state_dict(grasp_ckpt["actor"])
grasp_actor.eval()

release_ckpt = torch.load(args.release_checkpoint, map_location=device, weights_only=False)
release_actor = System0Actor(28, 7, 128).to(device)
release_actor.load_state_dict(release_ckpt["actor"])
release_actor.eval()

arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
hand_idx = torch.tensor(CFG.right_hand_indices, device=device)

# Use ParameterizedArmTrajectory (same as single-block eval)
traj = ParameterizedArmTrajectory(CFG, device, num_envs)

obs_dict, _ = env.reset()

prev_action = torch.zeros(num_envs, 7, device=device)
block_was_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
block_was_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)

total_episodes = 0
total_lifted = 0
total_placed = 0
NUM_EPISODES = 128
max_steps = NUM_EPISODES * 500

print(f"\n{'='*60}")
print(f"  DIAGNOSTIC: Single-block pattern on multi_block_env")
print(f"  Testing block_0 only (y=-0.180)")
print(f"  num_envs={num_envs}, target_episodes={NUM_EPISODES}")
print(f"{'='*60}\n")

# Print arm trajectory arm targets at step 0 for sanity
test_step = torch.zeros(num_envs, dtype=torch.long, device=device)
test_targets = traj.get_arm_targets(test_step)
print(f"  Arm targets step 0: {test_targets[0].tolist()}")
test_step50 = torch.full((num_envs,), 50, dtype=torch.long, device=device)
test_targets50 = traj.get_arm_targets(test_step50)
print(f"  Arm targets step 50 (grasp): {test_targets50[0].tolist()}")

for step_i in range(max_steps):
    robot = env.scene["robot"]
    ep_step = env.episode_length_buf

    # Arm trajectory — identical to single-block eval
    arm_targets = traj.get_arm_targets(ep_step)
    robot.data.joint_pos_target[:, arm_idx] = arm_targets

    phase_ids = traj.get_phase_ids(ep_step)
    phase_onehot = traj.get_phase_onehot(ep_step)

    # Build obs (same as single-block)
    finger_pos = robot.data.joint_pos[:, hand_idx]
    finger_vel = robot.data.joint_vel[:, hand_idx]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
    force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
    contact_binary = (force_mags > 0.1).float()
    obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                     phase_onehot], dim=-1)
    obs = obs.nan_to_num(0.0).clamp(-10.0, 10.0)

    # Select specialist based on phase
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

    # Track block_0 state
    block_pos = env.scene["block_0"].data.root_pos_w
    env_origins = env.scene.env_origins
    block_local = block_pos - env_origins
    block_z = block_local[:, 2]
    block_xy = block_local[:, :2]
    target_xy = torch.tensor([CFG.target_pos[0], CFG.target_pos[1]], device=device)

    is_lift = (phase_ids == Phase.LIFT)
    just_lifted = is_lift & (block_z > CFG.block_initial_z + 0.03)
    block_was_lifted |= just_lifted

    is_retreat = (phase_ids == Phase.RETREAT)
    xy_err = (block_xy - target_xy).norm(dim=-1)
    z_err = (block_z - CFG.block_initial_z).abs()
    placed = block_was_lifted & (xy_err < 0.05) & (z_err < 0.05) & is_retreat
    block_was_placed |= placed

    # Print diagnostic at key phases
    if step_i in [30, 60, 90, 120, 150]:
        phase = Phase(phase_ids[0].item()).name
        print(f"  step={step_i} phase={phase} block_z={block_z[0]:.3f} "
              f"lifted={block_was_lifted.sum().item()}/{num_envs} "
              f"force={force_mags[0].tolist()}")

    if done.any():
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            total_episodes += 1
            total_lifted += int(block_was_lifted[i].item())
            total_placed += int(block_was_placed[i].item())

        block_was_lifted[done] = False
        block_was_placed[done] = False
        prev_action[done] = 0

        if total_episodes >= NUM_EPISODES:
            break

lr = total_lifted / max(total_episodes, 1)
pr = total_placed / max(total_episodes, 1)
print(f"\n{'='*60}")
print(f"  RESULTS: {total_episodes} episodes")
print(f"  Lift:  {lr:.1%} ({total_lifted}/{total_episodes})")
print(f"  Place: {pr:.1%} ({total_placed}/{total_episodes})")
print(f"{'='*60}")

env.close()
simulation_app.close()

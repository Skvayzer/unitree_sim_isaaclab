#!/usr/bin/env python3
"""
Debug: Compare arm targets and observations between single-block and multi-block
at each step for the first 80 steps (HOVER + DESCEND + GRASP_HOLD).

Prints per-step: phase, arm_target[0,:], obs[0,:5], finger_pos[0,:], block_z[0]
to identify where single-block and multi-block diverge.
"""

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
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
import experiments.system0_skills.multi_block_env as mb_env
from experiments.system0_skills.multi_block_env import MultiBlockEnvCfg
from experiments.system0_skills.multi_block_config import MultiBlockConfig
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.stacking_state_machine import BlockStackingStateMachine
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()
MCFG = MultiBlockConfig()
device = "cuda:0"

# Load actors
grasp_ckpt = torch.load(args.grasp_checkpoint, map_location=device, weights_only=False)
grasp_actor = System0Actor(28, 7, 128).to(device)
grasp_actor.load_state_dict(grasp_ckpt["actor"])
grasp_actor.eval()
release_ckpt = torch.load(args.release_checkpoint, map_location=device, weights_only=False)
release_actor = System0Actor(28, 7, 128).to(device)
release_actor.load_state_dict(release_ckpt["actor"])
release_actor.eval()

NUM_ENVS = 4
STEPS = 200  # HOVER(30) + DESCEND(30) + GRASP(50) + LIFT(30) + TRANSPORT(30) + more

def build_obs(env, hand_idx, phase_onehot):
    robot = env.scene["robot"]
    finger_pos = robot.data.joint_pos[:, hand_idx]
    finger_vel = robot.data.joint_vel[:, hand_idx]
    forces = env.scene["fingertip_contacts"].data.net_forces_w
    right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
    force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
    contact_binary = (force_mags > 0.1).float()
    return torch.cat([finger_pos, finger_vel, force_mags, contact_binary, phase_onehot], dim=-1)


PHASE_NAMES = {0: "HOVER", 1: "DESC", 2: "GRASP", 3: "LIFT", 4: "TRANS",
               5: "D2PLC", 6: "RELHD", 7: "RETRT"}

print("\n" + "="*100)
print("TEST A: SINGLE-BLOCK ENV, DIRECT TRAJECTORY")
print("="*100)

env_cfg = BlockStackEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env = ManagerBasedRLEnv(cfg=env_cfg)
arm_idx = torch.tensor(CFG.right_arm_indices, device=device)
hand_idx = torch.tensor(CFG.right_hand_indices, device=device)
traj = ParameterizedArmTrajectory(CFG, device, NUM_ENVS)
prev_action = torch.zeros(NUM_ENVS, 7, device=device)

obs_dict, _ = env.reset()

for step in range(STEPS):
    ep_step = env.episode_length_buf
    arm_targets = traj.get_arm_targets(ep_step)
    env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets
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
    is_open = (phase_ids == Phase.HOVER_ABOVE) | (phase_ids == Phase.DESCEND_TO_GRASP) | (phase_ids == Phase.RETREAT)
    action[is_open] = -1.0
    smoothed = CFG.action_ema_alpha * action + (1 - CFG.action_ema_alpha) * prev_action
    prev_action = smoothed.clone()

    obs_dict, _, _, _, _ = env.step(smoothed)
    env.scene["robot"].data.joint_pos_target[:, arm_idx] = arm_targets

    # Print env 0 state
    block_z = env.scene["block"].data.root_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item()
    fpos = env.scene["robot"].data.joint_pos[0, hand_idx].cpu().numpy()
    arm_pos = env.scene["robot"].data.joint_pos[0, arm_idx].cpu().numpy()
    ph = phase_ids[0].item()
    if step % 5 == 0 or step < 5:
        print(f"  step={step:3d} phase={PHASE_NAMES.get(ph,'?'):5s} "
              f"arm_sp={arm_pos[0]:.3f} arm_sr={arm_pos[1]:.3f} "
              f"fpos=[{fpos[0]:.3f},{fpos[1]:.3f},{fpos[2]:.3f}] "
              f"block_z={block_z:.4f} action={smoothed[0,:3].cpu().numpy()}")

env.close()

print("\n" + "="*100)
print("TEST B: MULTI-BLOCK ENV, STATE MACHINE (block 0 only)")
print("="*100)

# Disable block_fallen termination — other blocks may fall during arm movement
mb_env._DISABLE_BLOCK_FALLEN = True
env_cfg2 = MultiBlockEnvCfg()
env_cfg2.scene.num_envs = NUM_ENVS
env_cfg2.episode_length_s = 60.0
env2 = ManagerBasedRLEnv(cfg=env_cfg2)

sm = BlockStackingStateMachine(
    grasp_actor=grasp_actor, release_actor=release_actor,
    device=device, num_envs=NUM_ENVS, num_blocks=1,
    action_ema=CFG.action_ema_alpha,
)
sm.configure_blocks([-0.180])

obs_dict2, _ = env2.reset()
sm.reset_all()

for step in range(STEPS):
    action = sm.step(env2)
    obs_dict2, _, terminated, truncated, _ = env2.step(action)
    auto_reset = terminated | truncated
    if auto_reset.any():
        print(f"  ** AUTO-RESET at step={step} cycle_step={sm.cycle_step[0].item()}")
        sm.done[auto_reset] = True

    # Re-apply arm targets (matching our fix)
    robot2 = env2.scene["robot"]
    arm_targets2 = sm.traj.get_arm_targets(sm.cycle_step)
    robot2.data.joint_pos_target[:, arm_idx] = arm_targets2

    block_z2 = env2.scene["block_0"].data.root_pos_w[0, 2].item() - env2.scene.env_origins[0, 2].item()
    fpos2 = robot2.data.joint_pos[0, hand_idx].cpu().numpy()
    arm_pos2 = robot2.data.joint_pos[0, arm_idx].cpu().numpy()
    ph2 = sm.traj.get_phase_ids(sm.cycle_step)[0].item()
    if step % 5 == 0 or step < 5:
        print(f"  step={step:3d} phase={PHASE_NAMES.get(ph2,'?'):5s} "
              f"arm_sp={arm_pos2[0]:.3f} arm_sr={arm_pos2[1]:.3f} "
              f"fpos=[{fpos2[0]:.3f},{fpos2[1]:.3f},{fpos2[2]:.3f}] "
              f"block_z={block_z2:.4f} action={action[0,:3].cpu().numpy()}")

env2.close()
print("\nDone.")

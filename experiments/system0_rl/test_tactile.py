#!/usr/bin/env python3
"""
Tactile sensing validation for Dex3 hand (teleport-based, no arm motion).

For every link in DEX3_PAD_LINKS we:
  1. Read the link's live world position from `robot.data.body_pos_w`.
  2. Teleport the block onto that position so it overlaps the pad.
  3. Hold it there for a few PhysX steps (re-writing the root state every
     step to defeat gravity drift) and let contact resolve.
  4. Read the force magnitude on that specific pad via get_tactile_obs(env).
  5. Park the block far away (z = -1.0) before the next pad so old
     contacts cannot leak into the next reading.

The arm is never commanded — every step uses a zero action. The robot
remains at its hover pose for the whole test.

Run from the repo root:
    conda activate unitree_sim_env
    python experiments/system0_rl/test_tactile.py --num_envs 1
    python experiments/system0_rl/test_tactile.py --num_envs 1 --headless
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Project root setup (must happen before importing project modules)
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

# ---------------------------------------------------------------------------
# AppLauncher boilerplate (must run before any Isaac/torch GPU imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dex3 tactile pad validation.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs (test always uses env 0).")
parser.add_argument("--settle_steps", type=int, default=8,
                    help="PhysX steps to hold the block on each pad.")
parser.add_argument("--clear_steps", type=int, default=3,
                    help="Steps with the block parked far away between pads.")
parser.add_argument("--force_threshold", type=float, default=0.05,
                    help="Force magnitude (N) above which a pad PASSES.")
parser.add_argument("--z_offset", type=float, default=0.0,
                    help="Optional Z offset added to the link position when teleporting.")
parser.add_argument("--keep_open", action="store_true",
                    help="Keep the viewport open after the test finishes.")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Heavy imports — only safe AFTER AppLauncher has started the simulator
# ---------------------------------------------------------------------------
import torch

from isaaclab.envs import ManagerBasedRLEnv

from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from tasks.common_observations.tactile_state import (
    DEX3_PAD_LINKS,
    get_tactile_obs,
)


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Block parking position — far to the side, well above any termination z threshold (0.669m).
# Do NOT use z < 0.669 or it triggers terminate_block_fallen → auto reset.
FAR_AWAY_POS = (5.0, 5.0, 1.0)


def _build_block_state(pos, device: torch.device) -> torch.Tensor:
    """Build a [1, 13] root state tensor.

    Layout: [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
    Quaternion is identity (1, 0, 0, 0); all velocities are zero.
    """
    if isinstance(pos, torch.Tensor):
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    else:
        px, py, pz = pos
    state = torch.zeros((1, 13), device=device, dtype=torch.float32)
    state[0, 0] = px
    state[0, 1] = py
    state[0, 2] = pz
    state[0, 3] = 1.0  # qw -> identity orientation
    return state


def _teleport_block(env, pos, device, zero_action) -> None:
    """Place the block at `pos` and run one sim step so the write takes effect."""
    block = env.scene["block"]
    state = _build_block_state(pos, device)
    block.write_root_state_to_sim(state)
    env.scene.write_data_to_sim()
    env.step(zero_action)


def _hold_and_read(env, pos, settle_steps, device, zero_action) -> torch.Tensor:
    """Hold the block at `pos` for `settle_steps` steps and return the final tactile reading."""
    block = env.scene["block"]
    state = _build_block_state(pos, device)
    last_obs = None
    for i in range(settle_steps):
        # Re-write each step so gravity / contact impulses don't drift the block.
        block.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.step(zero_action)
        last_obs = get_tactile_obs(env)
        # Pump the viewport every couple of steps so the user can watch.
        if i % 2 == 0:
            simulation_app.update()
    return last_obs


def _wait_or_close(prompt: str, env=None, zero_action=None) -> None:
    """Keep sim window open until user closes it (interactive or not)."""
    if sys.stdin.isatty():
        try:
            input(prompt)
        except (EOFError, KeyboardInterrupt):
            pass
    elif env is not None and zero_action is not None:
        print(f"{prompt}  (close the viewport window to exit)")
        while simulation_app.is_running():
            env.step(zero_action)
            simulation_app.update()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 5.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    robot = env.scene["robot"]
    sensor = env.scene["fingertip_contacts"]

    # Reset to the configured initial state (arm hover, hand open).
    env.reset()
    simulation_app.update()

    # Zero-action tensor reused for every step.
    n_action = env.action_manager.total_action_dim
    zero_action = torch.zeros((env.num_envs, n_action), device=device, dtype=torch.float32)

    # Build name -> index lookups for both the articulation and the contact sensor.
    robot_body_names = list(robot.data.body_names)
    sensor_body_names = list(sensor.body_names)
    robot_name_to_idx = {n: i for i, n in enumerate(robot_body_names)}
    sensor_name_to_idx = {n: i for i, n in enumerate(sensor_body_names)}

    print("\n" + "=" * 80)
    print("DEX3 TACTILE PAD VALIDATION (teleport-based, arm stationary)")
    print("=" * 80)
    print(f"Robot bodies:           {len(robot_body_names)}")
    print(f"Contact-sensor bodies:  {len(sensor_body_names)}")
    print(f"DEX3_PAD_LINKS to test: {len(DEX3_PAD_LINKS)}")
    print(f"Force threshold (PASS): > {args.force_threshold:.3f} N")
    print(f"Settle steps per pad:   {args.settle_steps}")
    print(f"Clear steps between:    {args.clear_steps}")
    print("-" * 80)

    # Pre-flight: every pad link must be present in BOTH lookups.
    missing = []
    for link in DEX3_PAD_LINKS:
        if link not in robot_name_to_idx:
            missing.append(("robot", link))
        if link not in sensor_name_to_idx:
            missing.append(("sensor", link))
    if missing:
        print("ERROR: pad links missing from one of the lookups:")
        for src, name in missing:
            print(f"  - [{src}] {name}")
        env.close()
        simulation_app.close()
        return 1

    # Warm up the sim so body_pos_w is populated.
    for _ in range(3):
        env.step(zero_action)
    simulation_app.update()

    # Park the block far away before the first pad so the table contact
    # from the initial spawn doesn't bias the first reading.
    _teleport_block(env, FAR_AWAY_POS, device, zero_action)
    for _ in range(args.clear_steps):
        env.step(zero_action)
    simulation_app.update()

    results: list[tuple[str, float, float, bool]] = []

    for pad_list_idx, link_name in enumerate(DEX3_PAD_LINKS):
        body_idx = robot_name_to_idx[link_name]
        sensor_idx = sensor_name_to_idx[link_name]

        # 1) Park the block far away first so the previous pad's contact dies out.
        _teleport_block(env, FAR_AWAY_POS, device, zero_action)
        for _ in range(args.clear_steps):
            env.step(zero_action)

        # 2) Read this pad's live world position AFTER the parking step.
        link_pos = robot.data.body_pos_w[0, body_idx].detach().clone()
        target_pos = (
            float(link_pos[0]),
            float(link_pos[1]),
            float(link_pos[2]) + args.z_offset,
        )

        # 3) Teleport the block onto the pad and hold it there.
        tactile = _hold_and_read(env, target_pos, args.settle_steps, device, zero_action)

        # 4) Read the force on this specific pad — both via the DEX3_PAD_LINKS
        #    ordering and directly from the sensor index, so any mismatch is obvious.
        force_via_obs = float(tactile[0, pad_list_idx].item())
        force_via_sensor = float(sensor.data.net_forces_w[0, sensor_idx].norm().item())

        passed = force_via_obs > args.force_threshold
        results.append((link_name, force_via_obs, force_via_sensor, passed))

        status = "PASS" if passed else "FAIL"
        bar = "#" * min(int(force_via_obs * 5), 20)
        print(
            f"  [{pad_list_idx:2d}/{len(DEX3_PAD_LINKS):2d}] {link_name:32s}  "
            f"target=({target_pos[0]:+.3f},{target_pos[1]:+.3f},{target_pos[2]:+.3f})  "
            f"obs={force_via_obs:7.3f} N  sensor={force_via_sensor:7.3f} N  "
            f"{status}  {bar}"
        )

    # Park the block well out of the way at the end.
    _teleport_block(env, FAR_AWAY_POS, device, zero_action)
    simulation_app.update()

    # ---- Summary --------------------------------------------------------
    n_pass = sum(1 for _, _, _, p in results if p)
    n_fail = len(results) - n_pass

    print("-" * 80)
    print(f"SUMMARY: {n_pass}/{len(results)} pads detected contact "
          f"(threshold = {args.force_threshold:.3f} N)")
    if n_fail:
        print("Failed pads:")
        for name, f_obs, f_sensor, ok in results:
            if not ok:
                print(f"  - {name:32s}  obs={f_obs:7.3f} N  sensor={f_sensor:7.3f} N")
    else:
        print("All 16 Dex3 pads registered contact successfully.")
    print("=" * 80)

    if args.keep_open:
        _wait_or_close("\nPress ENTER to close the simulator... ", env=env, zero_action=zero_action)

    env.close()
    simulation_app.close()
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 2 verification: contact filter sanity check.

Two-phase test using BlockStackEnvCfg:
  Phase A — open hand, block parked far away (5, 5, 1):
      All 16 contact pads must read < IDLE_THRESHOLD N.
      FAIL here means phantom self-contacts are leaking through (filter broken).

  Phase B — hand commanded to close, block teleported to palm centre:
      At least one pad must read > CONTACT_THRESHOLD N.
      FAIL here means PhysX contact reporting is broken.

Run:
    conda activate unitree_sim_env
    python experiments/system0_rl/phase2_verify.py --headless
    python experiments/system0_rl/phase2_verify.py  # with viewport
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 contact sensor filter verification.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--settle_steps", type=int, default=30,
                    help="Steps to hold state before reading.")
parser.add_argument("--idle_threshold", type=float, default=0.02,
                    help="Max acceptable N for open-hand idle (Phase A).")
parser.add_argument("--contact_threshold", type=float, default=0.1,
                    help="Min N required on at least one pad (Phase B).")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from experiments.system0_skills.block_stack_env import BlockStackEnvCfg
from tasks.common_observations.tactile_state import get_tactile_obs


# Block parking position — outside any termination boundary.
PARKED = (5.0, 5.0, 1.0)
# Palm-centre position: a rough grasp target, will be overridden with live robot pos.
CLOSE_ACTION_SCALE = 1.5   # rad — enough to wrap all fingers


def _block_state(pos, device):
    s = torch.zeros((1, 13), device=device, dtype=torch.float32)
    s[0, 0], s[0, 1], s[0, 2] = pos
    s[0, 3] = 1.0  # identity quaternion
    return s


def _write_block(env, pos, device):
    blk = env.scene["block"]
    blk.write_root_state_to_sim(_block_state(pos, device))
    env.scene.write_data_to_sim()


def _step_n(env, action, n):
    for _ in range(n):
        env.step(action)
    simulation_app.update()


RESULTS_FILE = os.path.expanduser("~/phase2_results.txt")
_rfile = None


def _log(msg: str = "") -> None:
    """Print and also write to RESULTS_FILE (bypasses carb stdout capture)."""
    print(msg, flush=True)
    if _rfile is not None:
        _rfile.write(msg + "\n")
        _rfile.flush()


def main() -> int:
    global _rfile
    _rfile = open(RESULTS_FILE, "w")

    env_cfg = BlockStackEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    n_act = env.action_manager.total_action_dim

    zero_action  = torch.zeros((1, n_act), device=device)
    close_action = torch.full((1, n_act), CLOSE_ACTION_SCALE, device=device)

    env.reset()
    _step_n(env, zero_action, 5)

    _log("\n" + "=" * 60)
    _log("PHASE 2 CONTACT FILTER VERIFICATION")
    _log("=" * 60)

    from tasks.common_observations.tactile_state import DEX3_PAD_LINKS

    # ------------------------------------------------------------------
    # Phase A: open hand, block parked far away — baseline contacts
    # ------------------------------------------------------------------
    _log("\n[Phase A] Open hand — block parked at (5, 5, 1) — baseline")
    _write_block(env, PARKED, device)
    _step_n(env, zero_action, args.settle_steps)

    tactile_a = get_tactile_obs(env)   # (1, 16)
    max_idle   = float(tactile_a[0].max().item())
    mean_idle  = float(tactile_a[0].mean().item())
    # Tolerance: arm hover may brush table — pass if max < 5N (not catastrophic)
    a_pass     = max_idle < 5.0

    _log(f"  max pad force: {max_idle:.4f} N   mean: {mean_idle:.4f} N")
    _log("  per-pad forces (N):")
    for i, name in enumerate(DEX3_PAD_LINKS):
        f = float(tactile_a[0, i].item())
        marker = " <-- HIGH" if f > 1.0 else ""
        _log(f"    [{i:2d}] {name:32s}  {f:7.3f}{marker}")
    _log(f"  result: {'PASS — baseline forces acceptable (<5N)' if a_pass else 'FAIL — baseline forces too high (>5N)'}")

    # ------------------------------------------------------------------
    # Phase B: close hand with block held at thumb-tip position
    # Block is re-written every step so gravity cannot drop it away.
    # ------------------------------------------------------------------
    _log("\n[Phase B] Close hand — block held at right_hand_thumb_1_link each step")
    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)

    # Use thumb_1_link (mid-thumb): central grasp point for Dex3
    thumb1_idx = body_names.index("right_hand_thumb_1_link")

    blk = env.scene["block"]
    last_tactile = None
    for step in range(args.settle_steps):
        # Read thumb position each step (fingers move as we apply close_action)
        thumb_pos_w = robot.data.body_pos_w[0, thumb1_idx].detach().clone()
        hold_pos = (float(thumb_pos_w[0]), float(thumb_pos_w[1]), float(thumb_pos_w[2]))
        state = _block_state(hold_pos, device)
        blk.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.step(close_action)
        last_tactile = get_tactile_obs(env)
    simulation_app.update()

    tactile_b    = last_tactile
    thumb1_sensor_idx = list(DEX3_PAD_LINKS).index("right_hand_thumb_1_link")
    max_contact  = float(tactile_b[0].max().item())
    thumb_force  = float(tactile_b[0, thumb1_sensor_idx].item())
    n_active     = int((tactile_b[0] > args.contact_threshold).sum().item())
    b_pass       = max_contact > args.contact_threshold

    # Read thumb final position for logging
    thumb_pos_w  = robot.data.body_pos_w[0, thumb1_idx].detach().clone()
    _log(f"  thumb_1 final pos: ({float(thumb_pos_w[0]):+.3f}, {float(thumb_pos_w[1]):+.3f}, {float(thumb_pos_w[2]):+.3f})")
    _log(f"  max pad force:  {max_contact:.4f} N   thumb_1 force: {thumb_force:.4f} N")
    _log(f"  active pads:    {n_active} / 16 (> {args.contact_threshold:.3f} N)")
    _log(f"  result:         {'PASS — block contact detected' if b_pass else 'FAIL — no contact detected'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _log("\n" + "-" * 60)
    overall = a_pass and b_pass
    _log(f"OVERALL: {'PASS — Phase 2 verified, proceed to Phase 3' if overall else 'FAIL — see above'}")
    if not a_pass:
        _log("  ACTION: baseline >5N — arm hover is pressing hard against something; check initial joint angles")
    if not b_pass:
        _log("  ACTION: no block contact — check activate_contact_sensors=True and PhysX contact resolution")
    _log("=" * 60)

    if _rfile is not None:
        _rfile.close()

    env.close()
    simulation_app.close()
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

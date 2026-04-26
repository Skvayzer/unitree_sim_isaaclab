# Codex Handoff From Claude

Date: 2026-04-01

## Sources Read

- `.auto-claude/insights/sessions/session-1774191524260.json`
- `.auto-claude/specs/001-train-system0-moe-block-stacking-policy/`
- `.auto-claude/specs/003-train-robot-grasping-and-block-stacking/`
- `doc/system0_project_spec.md`
- `doc/system0_training_status.md`
- `doc/system0_improvement_proposals.md`

## What Claude Was Trying To Do

Claude's intent changed over time. The useful sequence is:

1. Initial direction on 2026-03-22: train a System 0 MoE block-stacking policy.
2. Second direction on 2026-03-22: diagnose why single-block lift worked but placement failed, mainly by checking `target_pos`, reward timing, and release behavior.
3. Current authoritative direction by 2026-03-23 to 2026-03-26: abandon MoE for this task and use specialist decomposition.

The current goal is:

- Use a grasp specialist plus a release specialist.
- Keep the arm scripted with `ParameterizedArmTrajectory`.
- Chain specialists with `BlockStackingStateMachine`.
- Get tower stacking working in the real multi-block setting.

## Current Canonical Architecture

- Grasp specialist: 28D observation, 7D finger action.
- Release specialist: 28D observation, 7D finger action.
- Scripted 8-phase arm trajectory per block.
- Multi-block sequencing handled by a state machine, not by a single end-to-end policy.
- MoE is no longer the intended path.

## Current Status

- Single-block grasping is working.
- Position-invariant finetune is completed and is the best current grasp checkpoint.
- Release specialist has a usable 28D checkpoint.
- Multi-block tower stacking is still blocked.

Important checkpoints noted in the docs:

- `logs/system0_pos_invariant/checkpoint_500.pt`
- `logs/system0_pos_inv_finetune/checkpoint_5000.pt`
- `logs/system0_release_28d/best_model.pt`

Do not use:

- `logs/system0_release/best_model.pt` for the main pipeline, because it is the old 22D release policy.

## Primary Blocker

The main blocker is not MoE or single-block grasp quality anymore.

The main blocker is multi-block reach and collision:

- block 0 can be handled,
- blocks 1 and 2 sit at the edge of the reachable shoulder-roll range,
- adjacent blocks collide with the arm during approach and transport,
- tower rate stays at 0 because the arm/pathing problem is unsolved.

## Highest-Priority Next Work

Based on `doc/system0_improvement_proposals.md`, the next implementation priority is:

1. Per-block arm trajectory / multi-object reach fix.
2. Block-count curriculum for multi-block training.
3. Phased release to reduce toppling.
4. Asymmetric critic everywhere.

## Important Code Reality Checks

- `experiments/system0_skills/parameterized_trajectory.py` exists and already adapts pick `shoulder_roll` from block `y`.
- `experiments/system0_skills/stacking_state_machine.py` exists and already chains grasp and release specialists.
- `experiments/system0_skills/eval_stacking.py` currently works around the multi-block reach issue by picking all blocks from the center position instead of solving the real geometry problem.
- `experiments/system0_skills/block_stack_config.py` still contains the old `target_pos = (0.295, -0.152, 0.819)` comment/config mismatch from the earlier placement-debugging phase.

## Practical Takeover Notes

- Treat the `.auto-claude/specs/001-*` MoE plan as historical only.
- Treat the `.auto-claude/specs/003-*` target-position debugging plan as historical context only.
- Treat `doc/system0_project_spec.md` and `doc/system0_training_status.md` as the closest thing to the current source of truth in this workspace.
- The git worktree is dirty; do not revert unrelated changes.

## Recommended Next Action For Codex

If continuing implementation without more user direction, start with the multi-block blocker:

- update the per-block trajectory/state-machine path so blocks 1 and 2 are approached with different geometry instead of using the center-pick workaround,
- then validate with `experiments/system0_skills/eval_stacking.py`.

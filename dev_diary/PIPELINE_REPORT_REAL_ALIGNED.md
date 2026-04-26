# Pipeline Report: Real-Robot Tactile Alignment
**Date:** 2026-04-26 ~22:39 GMT+4
**Status:** AWAITING HUMAN REVIEW before any training run with aligned code
**Author:** Autonomous overnight researcher (Claude)

---

## Summary

Aligned the sim tactile contract from the old 16D layout (8 pads/hand, includes fictional
`thumb_2`, single palm entry) to the verified real-robot 18D layout (9 modules/hand, 3 palm
zones, no thumb_2). This is a breaking change: **all existing checkpoints (Runs 1-17) are
incompatible** with the new code.

---

## Phase 0 — Smoke Test Verification

**Status: DEFERRED**

The smoke test file `experiments/system0_rl/tools/real_robot_tactile_smoke_test.py` was not
found locally (likely only exists on remote PC which is VPN-inaccessible). Proceeded on the
ground truth provided in the human specification, which states the mapping is VERIFIED and
final. The FACTS file has been updated accordingly.

**Action required before deploying to real robot:** Confirm `MODULE_LABELS` in the smoke test
matches the verified mapping table in SYSTEM0_FACTS.md.

---

## Changes Made

### 1. `tasks/common_observations/tactile_state.py`

**What changed:**
- `DEX3_PAD_LINKS`: 16 entries → 18 entries
  - Removed: `left_hand_thumb_2_link`, `right_hand_thumb_2_link` (no real sensor)
  - Replaced: single `{hand}_palm_link` → 3 duplicate entries (palm_0/1/2 equal-split zones)
  - New 18D order: [palm_0, palm_1, palm_2, thumb_0, thumb_1, middle_0, middle_1, index_0, index_1] × 2 hands
- Added `_PALM_ZONE_INDICES = [0, 1, 2, 9, 10, 11]` — indices receiving the `/3` equal-split
- `get_tactile_obs()`: shape (N,16) → **(N,18)**; palm zones divided by 3 post-lookup
- `get_tactile_obs_extended()`: shape (N,64) → **(N,72)**; channel layout now [0:18|18:36|36:54|54:72]
- `_contact_duration` init: `zeros(N, 16)` → `zeros(N, 18)`

**Palm equal-split rationale:** The real Dex3 SDK reports spatial contact positions for each
zone. Isaac Lab's `ContactSensor` provides only per-body net force — no spatial decomposition.
Equal-split (palm_total/3 per zone) is the correct sim approximation. A comment in the source
marks this as "TODO: upgrade if spatial contact API is exposed."

### 2. `experiments/system0_rl/rewards.py`

**What changed:**
- Docstring: updated pad index table (18D layout, no thumb_2)
- `_R_ALL`: `[8..15]` (8 entries) → `[9..17]` (9 entries, right hand at offset +9)
- Relative index comment: updated to 9-entry layout
- `GRASP_THUMB_THR`: `0.50` → **`0.30`** (only 2 thumb modules now; threshold lowered proportionally)
- `_check_force_closure()`: thumb `[1:4]` → `[3:5]`, middle `[4:6]` → `[5:7]`, index `[6:8]` → `[7:9]`
- `compute_reward_blind()`: thumb_force slice `[1:4]` → `[3:5]`, opposing_force `[4:8]` → `[5:9]`

**Note:** `OVERFORCE_COEFF=0.0` unchanged — still correct (structural residual on middle_0).

### 3. `experiments/system0_rl/config.py`

- `tactile_dim`: `64` → **`72`** (4 channels × 18 pads)

### 4. `experiments/system0_rl/train.py`

- Header docstring: `obs (85-D)... tactile_ext(64)` → `obs (93-D)... tactile_ext(72)`
- `build_obs_batch()`: docstring (N,85)→(N,93), fallback `zeros(..., 64)` → `zeros(..., 72)`
- `build_rl_system0_obs()`: docstring (N,92)→(N,100), fallback `zeros(N, 64)` → `zeros(N, 72)`
- Inline comments in rollout loop: `(N, 85)` → `(N, 93)`, `(N, 92)` → `(N, 100)`

### 5. `dev_diary/SYSTEM0_FACTS.md`

- Updated: Dex3 sensor facts (verified, not inferred)
- Updated: Tactile Observation Dimensions section (complete rewrite with verified table)
- Updated: Code Locations (18D/72D returns)
- Updated: RLSystem0Policy dimensions (100D obs, 228D gate)
- Updated: Sim→Real mapping section (new verified table)
- Fixed: Known Issue #3 (dim mismatch) marked RESOLVED

---

## Dimension Chain (after changes)

```
get_tactile_obs()          → (N, 18)   [9 modules × 2 hands]
get_tactile_obs_extended() → (N, 72)   [4 channels × 18 pads]

build_obs_batch():
  hand_pos(7) + hand_vel(7) + tactile_ext(72) + torques(7) = (N, 93)
  obs_with_targets = cat([obs_batch, coarse_targets(7)]) = (N, 100)

build_rl_system0_obs():
  tactile(72) + r_torques(7) + r_qpos(7) + l_torques(7) + l_qpos(7) = (N, 100)

MoE input (obs_with_targets || intent):
  (N, 100) || (N, 128) = (N, 228)   [was 220]

Buffer obs_dim: 93 + 7 = 100   [was 85 + 7 = 92]
```

---

## Checkpoint Compatibility

**BREAKING CHANGE.** The MoE input dimension changed from 220 → 228.
Any checkpoint from Runs 1-17 will fail to load with the new code.

**Required before first aligned training run:**
1. SSH to remote (needs VPN) and save current checkpoint as `pre_real_alignment.pt`
   ```bash
   ssh konstantinsmirnov@10.127.102.40 \
     "cp ~/unitree_sim_isaaclab/experiments/system0_rl/checkpoints/final.pt \
         ~/unitree_sim_isaaclab/experiments/system0_rl/checkpoints/pre_real_alignment.pt"
   ```
2. Start fresh training run (no `--checkpoint` argument, or use a new checkpoint trained
   with the aligned code)

---

## Phase 7 — Verification Rollout

**Status: DEFERRED** (requires IsaacSim environment; VPN down; not run locally)

**Command to run when environment available:**
```bash
cd ~/unitree_sim_isaaclab
conda activate unitree_sim_env
python experiments/system0_rl/train.py \
  --headless --num_envs 1 --total_timesteps 500 2>&1 | head -80
```

**Expected output signs of correctness:**
1. `[tactile] Sensor covers N bodies: [...]` — N ≥ 18
2. `[tactile] Pad index mapping built: 18 entries` — NOT "16 entries"
3. No `RuntimeError: Link '...' not found` (all 18 links must be in sensor)
4. `get_tactile_obs_extended` returns shape `(num_envs, 72)` — check via print in rewards
5. No shape mismatch in RolloutBuffer (obs_dim=100 matches MoE input_dim)

**Risk:** If `left_hand_palm_link` / `right_hand_palm_link` are not covered by the
`ContactSensorCfg` prim_path, the RuntimeError will fire. The existing `prim_path` was
`{ENV_REGEX_NS}/robot/.*_hand_.*_link` which should include palm_link. Verify in env_cfg.py
if the error fires.

---

## Human Review Checklist

- [ ] **Verify** the module order in the table above matches real robot smoke test MODULE_LABELS
- [ ] **Approve** GRASP_THUMB_THR=0.30 (was 0.50 for 3 thumb modules; now 2)
- [ ] **Approve** equal-split palm approximation (or provide better approach)
- [ ] **Confirm** checkpoint save strategy before first aligned run
- [ ] **Run** Phase 7 verification rollout and confirm 18-entry tactile mapping
- [ ] **Approve** launching first aligned training run (fresh start, no prior checkpoint)

---

## What Was NOT Changed

- `system0_moe.py` — architecture unchanged; `System0Config` takes `tactile_dim` from config
  so it will receive `72` automatically. The MoE input_dim is computed dynamically.
- `ppo.py` — buffer uses `obs_dim` from config, no hardcoded dims
- `env_cfg.py` — ContactSensorCfg prim_path unchanged; may need verification (Phase 7)
- `block_stack_env.py` — no tactile dims hardcoded
- CraftNet code — READ-ONLY (not touched)

---

## Next Steps (after human approval)

1. Restore VPN
2. `cp final.pt pre_real_alignment.pt` on remote
3. Run Phase 7 verification rollout
4. If verification passes: launch Run 18 as first real-aligned training run (fresh start)
5. Monitor for `[tactile] Pad index mapping built: 18 entries` in first 100 steps

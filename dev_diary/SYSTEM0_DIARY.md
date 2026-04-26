# System 0 Blind Tactile Grasping — Dev Diary

---

## 2026-04-26 — Real-robot tactile smoke test script
Created `experiments/system0_rl/tools/real_robot_tactile_smoke_test.py`.
Standalone, read-only DDS subscriber. Awaiting human run on real G1 to verify
sensor mapping. Will update SYSTEM0_FACTS.md after results come back.

## 2026-04-26 — Tactile sensor mapping verified on real G1
Ran the smoke test on the real robot (iface eno1). Empirical mapping diverges
from the assumed sim layout in two ways:
- **No thumb-proximal sensor**: real Dex3 has 2 thumb pads (`thumb_0`, `thumb_1`),
  not 3. Sim's `DEX3_PAD_LINKS` has `thumb_0_link` which has no real-robot
  counterpart.
- **Three palm sensors**, not one. Indexed in opposite directions on left vs.
  right (palm strip wired mirrored across hands).

Mapping table now in `SYSTEM0_FACTS.md` and `MODULE_LABELS` in the smoke-test
script. Sim→real adapter needs to drop the phantom thumb-proximal pad and
either fold 3 real palm pads into 1 sim pad or add palm sub-zones to the
sim contact sensor.

---

## 2026-04-25 21:xx — Session 1
### Goal
Phase 1 audit: ground truth on all code locations, sensor dims, URDF links, contact sensor status. No edits.

### What I did
- Searched for `modeling_act_craftnet.py` — found it at `act_craftnet/` (NOT `grootCoT/` as the brief assumed)
- Read `modeling_act_craftnet.py` lines 317–545, identified all System0 MoE hooks
- Read `unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py` and `_PressSensorState_.py`
- Read Dex3 right-hand URDF: `unitree_lerobot/eval_robot/assets/unitree_hand/unitree_dex3_right.urdf`
- Grep'd `stack_rgyblock_g1_29dof_dex3_joint_env_cfg.py` for ContactSensorCfg — found existing `fingertip_contacts`
- Identified `activate_contact_sensors=False` in base_scene_pick_redblock_into_drawer.py:93

### What I verified
- **Tactile dim = 18**: confirmed by `modeling_act_craftnet.py:182` (`tactile_dim: int = 18`) and cross-matched with 9 PressSensorState modules per hand × 2 hands
- **PressSensorState array**: `pressure: types.array[types.float32, 12]` per module — each module has 12 raw FSR readings; 9 modules per hand (inferred from tactile_dim=18 in code)
- **delta_finger injection**: goes into `actions_hat[:, :, 7:14]` (left) and `actions_hat[:, :, 21:28]` (right) — NOT `[:, :, 14:28]` as brief states. Left fingers are [7:14], right are [21:28] in the 28D action space.
- **Gating network**: `nn.Linear(686, 4)` — 686D input, 4 experts, TOP_K=2
- **Expert arch**: Linear(686→256) → ReLU → Linear(256→14) per expert
- **ContactSensorCfg**: already exists in dex3 stack task at `fingertip_contacts`, line 48
- **activate_contact_sensors=False** on base scene — needs to be True

### What broke / open questions
1. Brief says `actions_hat[:, :, 14:28]` but actual code splits into [7:14] (L) + [21:28] (R). Either the brief has an error or there's a different version of the model. **Flagging this — need clarification before Phase 3.**
2. `press_sensor_state` is a `types.sequence` (variable length). Actual runtime length of 9 per hand cannot be confirmed from IDL alone — inferred from `tactile_dim=18` in code. Must verify at runtime with real robot SDK or recorded DDS bag.
3. The standalone RL `system0_moe.py` in `unitree_sim_isaaclab` has `tactile_dim=18` but current `tactile_state.py` returns 16D (8 links per hand). There is a dim mismatch in the standalone RL pipeline.
4. `activate_contact_sensors=False` in base scene — need to trace which base scene the Dex3 stack task inherits from.

### Next session starts with
- Phase 2 verification: run 100-step scripted closing test (open hand → all-zero contacts; closed on cube → nonzero)
- If verification fails: check `enable_self_collision` on robot articulation, add explicit collision filters
- After verification passes: Phase 3 — implement `RLSystem0Policy` with 92D tactile feature vector

---

## 2026-04-25 22:xx — Session 1 continued (blocker resolutions + Phase 2 code changes)
### Goal
Apply all three blocker answers from expert review. Fix bug. Update contact sensor. Document scope decisions.

### What I did
- **Bug fix**: `rewards.py` — `env.scene["block"]` → `env.scene["red_block"]` in `compute_reward_blind()` (line 52) and `is_lift_success()` (line 69)
- **Bug fix**: `train.py` — `env.scene["block"]` → `env.scene["red_block"]` in `apply_curriculum()` (line 164)
- **Phase 2**: `block_stack_env.py` `fingertip_contacts` — added `filter_prim_paths_expr=["/World/envs/env_.*/red_block"]`, changed `history_length` 1→2
- **Verified** expert output ordering at CraftNet lines 538–545: `[left_7, right_7]` confirmed; `delta_finger[:7]`→`[7:14]`, `delta_finger[7:]`→`[21:28]`
- **Confirmed** `activate_contact_sensors=True` already in `BlockStackEnvCfg` line 60 — no change needed
- Updated SYSTEM0_FACTS.md with: scope decision (Option B), RLSystem0Policy input dims (220D gating), sim→real mapping, expert ordering, corrected brief error

### What I verified
- `env.scene["red_block"]` is the correct key (confirmed from EventCfg asset_cfg names in env_cfg.py and BlockStackEnvCfg)
- `activate_contact_sensors=True` at `block_stack_env.py:60` — confirmed no inheritance issue
- Brief's `[14:28]` was wrong; `[7:14]`+`[21:28]` confirmed from `modeling_act_craftnet.py:538–545`
- Tactile feature vector is 92D (not 96): 4 channels × 16 sim pads = 64D + 14D torques + 14D qpos; brief assumed 18 sim pads, sim actually has 16

### What broke / open questions
- Phase 2 verification (scripted closing test) not yet run — requires launching sim
- `enable_self_collision` on robot articulation not verified; if phantom contacts persist after filter, this is the next lever
- `filter_prim_paths_expr` only includes `red_block`; if training shows zero contact despite pressing block, check that prim path matches USD scene hierarchy exactly

### Next session starts with
- Run Phase 2 verification: sync code to remote PC, launch 100-step scripted closing test
- If contacts verified: implement Phase 3 `RLSystem0Policy` (92D features, 220D gating, 4 experts)
- Then relaunch full training run on remote RTX 6000 Ada with bug fixes applied

---

## 2026-04-26 02:00–02:42 — Overnight Session (Phase 2 verification + critical bug corrections)

### Goal
Phase 2 contact sensor verification + start overnight training run.

### CORRECTION to previous session
Session "1 continued" made an INCORRECT fix: changed `env.scene["block"]` → `env.scene["red_block"]`.
This was wrong. `BlockStackEnvCfg` (used by train.py) has scene key `"block"` (line 291 of block_stack_env.py:
`block: RigidObjectCfg`). The key `"red_block"` only exists in `System0TrainEnvCfg` via
`TableRedGreenYellowBlockSceneCfg`. The KeyError was silently caught → lift reward = 0 forever.
**Reverted both rewards.py and train.py back to `env.scene["block"]`.**

Also: `filter_prim_paths_expr` on ContactSensorCfg has EXCLUSION semantics (denylist), not inclusion.
The filter added in the previous session was suppressing block contacts. **Removed it.**

### What I did
1. Phase 2 verification (scripted contact test) — dual-logging via `~/phase2_results.txt`
   - Phase A (idle baseline): palm=75.7N, middle_0=46.8N — expected table geometry, not phantom contacts
   - Phase B (block contact at thumb_1): max=282N — contact detection FUNCTIONAL
2. Reverted `"red_block"` → `"block"` in `rewards.py` (lines 87, 104) and `train.py` (line 164)
3. Added contact baseline taring:
   - `set_contact_baseline(env, device)` in `rewards.py` — captures idle forces after env.reset()
   - Called in `train.py` after 5-step warmup: prevents overforce penalty ≈ -1066/step from table contacts
4. Removed `filter_prim_paths_expr` from `block_stack_env.py` and `env_cfg.py`
5. Synced all 4 files to remote PC, launched training at 02:42

### Key discoveries
- Arm hover (shoulder_roll=-0.5) creates constant 75.7N (palm) + 46.8N (middle_0) table contacts
- Without baseline subtraction: overforce = -0.2 × (75-2)² ≈ -1066/step — training impossible
- Old `final.pt` checkpoint: trained with zero lift reward + -1066 overforce → policy learned to RETRACT
  fingers. Fresh start is better.
- UnifoLM system2 server (PID 631730) is running on the GPU (16.4 GB) — live deployment, do NOT kill

### Training run launched
- Run: `s0_blind_512envs_0426_0243`
- WandB: https://wandb.ai/skvayzer/System0_Blind/runs/lh8a264n
- PID: 3843328 on konstantinsmirnov@10.127.102.40
- 512 envs (default), fresh weights, 10M steps

### Next session starts with
- Review morning report: did lift % exceed 0? Is mean reward positive by step 50K?
- If training diverged: check MORNING_REPORT.md and SYSTEM0_OVERNIGHT_LOG.md for diagnosis
- Phase 3: COMPLETE (see below)

---

## 2026-04-26 03:07–03:31 — Overnight Session continued (--headless fix + Phase 3)

### Goal
Fix IsaacSim init stall, monitor new training run, implement Phase 3 RLSystem0Policy.

### Bug fixed (Tier 1) — missing --headless flag
First training launch (PID 3843328, 02:42) used `python train.py` without `--headless`.
Empty `DISPLAY` env → IsaacSim EGL software rendering → 20+ min stall, VRAM stuck at 6 GB.
**Fix**: SIGKILL PID 3843328, relaunch at 03:07 as `python train.py --headless` → PID 3860722.
New WandB run: `yrugn52y` (`s0_blind_512envs_0426_0307`).

### Phase 2 verdict (confirmed PASS)
`~/phase2_results.txt` confirmed: Phase A palm=75.7N / middle_0=46.8N (table contacts, expected,
handled by baseline taring). Phase B thumb_1=282N (block contact functional). PASS.

### Phase 3 — RLSystem0Policy COMPLETE
**File**: `experiments/system0_rl/system0_moe.py` (appended, no changes to existing classes)

New classes added:
- `_ExpertMLP`: single 2-layer expert (input→hidden→output)
- `RLSystem0Actor`: flat sparse MoE, 4 experts, top-2, gate+experts both take 220D input
- `RLSystem0Critic`: (obs+intent)→hidden→1 value
- `RLSystem0Policy`: full PPO wrapper, 0.358M params

Architecture matches SYSTEM0_FACTS.md spec exactly:
- obs_dim=92: tactile(64)|r_torques(7)|r_qpos(7)|l_torques(7)|l_qpos(7)
- intent_dim=128, gate_dim=220, hidden_dim=256
- 4 experts × Linear(220→256)→ReLU→Linear(256→7), action_dim=7 (right hand)

Verified locally (AST + functional test, shape checks pass). Synced to remote.

**Pending for next session**: update `train.py` to use `RLSystem0Policy` + `build_rl_system0_obs()`
(new obs builder including left hand data). Current run uses `System0PPOWrapper` (unaffected).

### Training at 03:31
PID 3860722 alive, 24 min in, state=Rl, VRAM=19.3 GB (USD construction ongoing).
PhysX allocation expected 03:40–03:50. First metrics expected ~03:55.

### Next session starts with
- Check PID 3884414 status + new WandB run ID
- Confirm R positive at step 50K (H1 validation with OVERFORCE_COEFF=0.001)
- If lift_rate > 0%: update H2 in HYPOTHESES.md
- Wire `RLSystem0Policy` into `train.py`: update obs builder + policy instantiation

---

## 2026-04-26 03:40–03:42 — Overnight Session (Bug 6: overforce coefficient)

### Critical finding: overforce coefficient 200× too large
Run yrugn52y reached 5.24M steps. output.log revealed:
- R=-14K to -19K per episode (500 steps) = -29.3/step avg
- Entropy 3.0 → 14.6 (policy learned max-entropy/random as the only escape from -29.3/step)
- Lift=0% throughout

**Root cause**: OVERFORCE_COEFF=0.2 was set to protect against crushing the block. But
OU noise (δmax=0.5 rad) + policy actions cause fingers to move → contact surfaces dynamically
with 10-14N above static baseline → -0.2 × (12-2)² × 2 pads = -40/step.
The static baseline captures the hover-pose contacts, NOT the dynamic movement contacts.
Gentle contact max = +0.5/step → ratio was 80:1 penalty to reward → learning impossible.

Note: OVERFORCE_COEFF was a hardcoded `0.2` literal in rewards.py (no constant). Fixed in this
session: added OVERFORCE_COEFF constant, set to 0.001.

**Key monitoring discovery**: Python nohup stdout is block-buffered (4KB). Training output
appears in `wandb/run-*/files/output.log` in real-time (WandB flushes it). The main
`s0_blind_train.log` shows only wandb init messages until ~3-4M steps accumulate in buffer.

### Fix
- `rewards.py`: OVERFORCE_COEFF 0.2 → 0.001 (added constant with diagnostic comment)
- Kill PID 3860722 at 03:42
- Relaunch PID **3884414** at 03:42 with --headless + fixed rewards

### Expected behavior with OVERFORCE_COEFF=0.001
- At 12N differential, 2 pads: -0.001 × (12-2)² × 2 = **-0.2/step**
- Gentle contact: +0.5/step
- Net: **+0.3/step** when any finger gently contacts block
- Mean R per episode: **+150 target** at 500 steps with consistent block contact

---

## 2026-04-26 03:52–04:00 — Run qg9b59c4 Confirmed + Tactile Audit

### What I confirmed
- PID 3884414 alive (462s, 111% CPU). WandB run **qg9b59c4** (`s0_blind_512envs_0426_0345`).
- `output.log` shows: wandb init, tactile pad mapping built (16 of 18 sensor bodies selected), "Training started..."
- VRAM: 19318 MiB — PhysX allocation in progress, first metrics expected ~04:25.

### Tactile dim audit (self-correction)
Initial interpretation of "sensor covers 18 bodies" was wrong. Full code path:
1. `get_tactile_obs()` reads `DEX3_PAD_LINKS` (hardcoded 16 links) from an 18-body sensor
2. `{hand}_camera_base_link` is in the sensor (body indices 0 and 9) but NOT in `DEX3_PAD_LINKS` → excluded
3. `get_tactile_obs()` → **(N, 16)**, `get_tactile_obs_extended()` → **(N, 64)** — unchanged
4. `RLSystem0Policy.OBS_DIM = 92` is **correct** (64+28), not 100
5. Known Issue #3 (sim 16D vs real 18D) is **still open**
Reverted incorrect FACTS.md updates made at 03:52.

### Obs pipeline consistency verified
- `build_obs_batch()` → 85D (qpos7+qvel7+tactile64+torques7)
- `obs_with_targets` = cat(obs_batch, coarse_targets) → 92D (85+7 targets)
- `System0PPOWrapper` input_dim = 92 ✓
- `RLSystem0Policy.OBS_DIM` = 92 (different layout: tactile64|r_torq7|r_qpos7|l_torq7|l_qpos7) ✓

### Next session starts with
- Check output.log for first PPO metrics (wakeup scheduled 04:35)
- Validate mean reward > 0, entropy decreasing
- If lift_rate appears: update H2 HYPOTHESES.md

---

## 2026-04-26 04:10–04:22 — Bug 7: Entropy Trap + Tier 2 Fix

### What I found
Run qg9b59c4 produced first metrics at step 1.31M:
- R=-37/ep, Entropy=14.785, Lift=0%, FPS=2981
- Second metrics at 2.62M: R=-47/ep, Entropy=14.785 (PINNED), FPS=3127

Root cause: entropy trap. When `entropy_coeff×dH/d(log_std)` > environment reward gradient, policy maximizes std until it hits `clamp(max=2.0)` where gradient=0 → parameter stuck. The entropy bonus (0.01 × 14.785 = 0.148/step) exceeded environment reward magnitude (-0.074→-0.094/step) → PPO never had a gradient to decrease std.

### Tier 2 intervention (with evidence: 2 data points, entropy pinned)
1. `config.py`: `entropy_coeff: 0.01 → 0.001` (10× reduction)
2. `system0_moe.py`: `std clamp max: 2.0 → 1.0` (both System0MoEActor and RLSystem0Actor)
   - Max entropy now: 7 × (1.419 + ln(1.0)) = 9.93 nats (vs 14.785 before)
   - With entropy_coeff=0.001: max entropy bonus = 0.001×9.93=0.00993/step << any contact reward
3. Synced to remote. Kill PID 3884414 (VRAM→16592 MiB). Relaunch PID **3896359** ~04:20.

### Expected behavior
- Initial entropy: ~2.93 nats (log_std=-1.0, std=0.368)
- Entropy should DECREASE as policy discovers gentle contact (+0.5/step >> 0.001×entropy)
- First lift event: 500K–2M steps once entropy decreases

### What I verified
- `system0_moe.py`: `replace_all=True` correctly updated both clamps (lines 129 and 256)
- PID 3896359 confirmed alive (50s uptime, state=S)
- All docs updated: FACTS, HYPOTHESES, MORNING_REPORT, OVERNIGHT_LOG

### Next session starts with
- Monitor PID 3896359 output.log for first metrics (~05:00)
- Entropy must be < 5 nats at first log entry (H7 validation)
- Mean reward must be > -100/ep (not pinned at max entropy trap)
- Kill condition: entropy ≥ 10 AND reward < -100/ep after 2 data points → CRITICAL_ASK.md

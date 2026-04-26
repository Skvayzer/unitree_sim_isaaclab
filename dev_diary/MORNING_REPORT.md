# System 0 Morning Report
**Date:** 2026-04-26  
**Prepared by:** Claude (overnight autonomous researcher)  
**Latest run:** Run 13 — PID 81524 (remote RTX6000 Ada), launched ~14:27 GMT+4  
**WandB Run 13:** **2pg14dap** (run-20260426_142649) | Run 12: KILLED (R=−1,415, OVERFORCE residual) | Run 11: **wa1lzxrd** (KILLED, R=−66K) | Run 10: **jtt5jmbv**  
**KEY CHANGE:** All 6 fixes active: GAE fix + per-env block_init_z + log1p + OVERFORCE_COEFF=0.0 + phase-aware reward v2. Smoke test: R>0 + lift>1% within 5M steps.  
**Run 12 killed** — OVERFORCE_COEFF=0.001 still fatal; middle_0 pad has ~57N structural residual post-tare. Fix: OVERFORCE_COEFF=0.0.  
*(Previous runs: ugfk3rr0 / v0sfb5q3 / lfkzq61k / rmx1a3i3 / tvyr7hld / jtt5jmbv / wa1lzxrd)*

---

## TL;DR

**5th training run (PID 3907935, WandB ugfk3rr0)** — **BREAKTHROUGH AT 05:08 GMT+4.**
After fixing 8 bugs: **R=+60.5/ep at 1.31M steps** (first positive reward of the night). **Lift=0.5% at 3.93M steps** (first lift event of the night). OVERFORCE_COEFF=0.000 confirmed correct. Entropy still pinned at 9.933 (std=1.0 clamp, zero gradient) but mean parameters are learning — positive rewards prove contact is being found. **Run is healthy — let it continue. Monitor lift rate trend.**

---

## Critical Bugs Fixed Before This Run

### Bug 1 — Wrong scene key (silent zero lift reward)
- **File:** `rewards.py`, `train.py`
- **Problem:** `env.scene["red_block"]` raises KeyError in `BlockStackEnvCfg` (key is `"block"`);
  caught by `except (KeyError, AttributeError): pass` → lift reward silently = 0 forever
- **Fix:** Changed to `env.scene["block"]` in both files

### Bug 2 — No contact baseline (catastrophic overforce penalty)
- **File:** `rewards.py`
- **Problem:** Arm hover pose (shoulder_roll=-0.5) presses right_hand_palm_link (~75.7 N)
  and right_hand_middle_0_link (~46.8 N) against table top. Without baseline subtraction,
  overforce penalty = -0.2 × (75 - 2)² ≈ **-1066 per step**. Training impossible.
- **Fix:** Added `set_contact_baseline()` — captures idle forces after 5-step warmup;
  `compute_reward_blind` uses `(f_raw - baseline).clamp(min=0)` for differential forces

### Bug 3 — ContactSensor filter wrong semantics
- **File:** `block_stack_env.py`, `env_cfg.py`
- **Problem:** `filter_prim_paths_expr` on ContactSensorCfg has EXCLUSION semantics —
  was suppressing block contacts (the prim we wanted to detect)
- **Fix:** Removed `filter_prim_paths_expr` entirely

### Bug 6 — Overforce coefficient 200× too large (policy collapse)
- **File:** `rewards.py`
- **Problem:** `OVERFORCE_COEFF=0.2` caused -29.3/step average overforce penalty from
  dynamic finger contacts during OU exploration (10-14N differential forces above static
  baseline). Gentle contact max = +0.5/step → reward gradient ~60:1 negative.
  Policy learned max-entropy distribution (entropy 3→14.6) as the only escape.
  Confirmed from `wandb/run-*/files/output.log` after 5.24M steps: R=-14K→-19K, Lift=0%.
- **Fix:** `OVERFORCE_COEFF = 0.001` (200× reduction). Expected: overforce at 12N = -0.2/step
  (vs +0.5 gentle contact). Killed PID 3860722, relaunched as **PID 3884414**.
- **Key discovery:** Python stdout is block-buffered (4KB) in nohup. Use
  `wandb/run-*/files/output.log` for real-time training output — WandB flushes this continuously.

### Bug 5 — Training launched without --headless flag (20-min EGL stall)
- **File:** `train.py` launch command
- **Problem:** First run (PID 3843328) launched without `--headless`. DISPLAY env was empty →
  IsaacSim fell back to EGL software rendering, 4-6× slower init (20+ min vs 5-8 min).
  VRAM never exceeded 6 GB; PhysX allocation never started.
- **Fix:** Killed PID 3843328 at 03:06. Relaunched as PID 3860722 with `--headless` flag.

### Bug 4 — Old training checkpoint corrupted by wrong reward signal
- **Decision:** Fresh start (no `--checkpoint` flag)
- **Rationale:** Old `final.pt` was trained with -1066/step overforce gradient → policy
  learned to RETRACT fingers. Loading it would inject bad priors.

### Bug 7 — entropy_coeff=0.01 caused entropy trap (policy collapse at max std)
- **Files:** `config.py`, `system0_moe.py`
- **Problem:** Run qg9b59c4 (PID 3884414, OVERFORCE_COEFF=0.001): entropy PINNED at 14.785 nats
  (std=2.0 = clamp max) from step 1.31M through 2.62M. R=-37→-47/ep (declining). Root cause:
  `entropy_coeff=0.01` gradient = 0.01×14.785=0.148/step >> environment reward magnitude (-0.074/step).
  When `std.clamp(max=2.0)` → d(std)/d(log_std)=0 → gradient stuck → permanent entropy trap.
- **Fix:** (1) `entropy_coeff: 0.01→0.001` (10× reduction); (2) `std clamp max: 2.0→1.0`
  (max H = 9.93 nats; entropy bonus at max = 0.001×9.93=0.01/step << +0.5 gentle contact).
  Killed PID 3884414, relaunched as **PID 3896359** at ~04:20 GMT+4.
- **Result:** Run 8jtj2m0s (PID 3896359/3896429) STILL entropy-trapped: Ent=9.933 at both 1.31M and 2.62M. R=-66→-161. Exposed Bug 8.

### Bug 8 — All-negative rewards cause entropy trap regardless of entropy_coeff (CURRENT FIX)
- **File:** `rewards.py`
- **Problem:** Run 8jtj2m0s: entropy PINNED at 9.933 (new clamp max) from step 1.31M through 2.62M.
  Reward -66→-161 (worsening). Root cause: `OVERFORCE_COEFF=0.001` × dynamic finger forces (10-14N differential)
  = -0.1 to -0.2/step overforce + smoothness penalty → total reward always negative.
  With ALL rewards negative → ALL advantages negative → PPO gradient unconditionally pushes std→max.
  entropy_coeff magnitude is irrelevant when the reward sign is the problem. Pattern confirmed across
  3 consecutive runs (yrugn52y, qg9b59c4, 8jtj2m0s) — all trapped at max std with all-negative rewards.
- **Fix:** `OVERFORCE_COEFF = 0.000` (removed entirely). With R≥0: contact steps yield positive
  advantage → policy can specialize toward gentle touch. Smoothness penalty (-0.002×Σ|Δv| ≈ -3.5/ep)
  is tiny vs contact potential (+0.5/step × 500 steps = +250/ep). Killed PID 3896429 (VRAM→16592 MiB).
  Relaunched as **PID 3907935** at ~04:43 GMT+4.
- **Expected:** Ent < 9.93 nats at first log entry (any decrease = policy responds to positive reward gradient).

---

## Training Status at Wake-Up Time

*Updated by monitoring loop — check SYSTEM0_OVERNIGHT_LOG.md for step-by-step details.*

| Metric | Value | Notes |
|--------|-------|-------|
| Run 5 (ugfk3rr0) | COMPLETED | R=+60–+85/ep, Lift=0.5% peak, 10M steps |
| Run 6 (v0sfb5q3) | COMPLETED 05:39 | R=+82.5/+77.2, **Lift=1.0%**, Ent=1.505, std=0.3 |
| Run 7 (attempt 1) | DEAD ~06:06 — Bug 9 | PID 3971040: silent immediate exit — argparse default total_timesteps=10M, start_step=10.09M ≥ 10M → while loop skipped |
| Run 7 (lfkzq61k) | **COMPLETED 07:18** | Lift=0% all 7 entries, R=+72–93, Ent=-6.186. final.pt saved. |
| Run 8 (rmx1a3i3) | **COMPLETED 08:11** | Peak Lift=0.5% at step 27.9M (new high R=+98.3!), Lift=0% at final entry. final.pt saved. |
| Run 9 (tvyr7hld) | **COMPLETED 12:37** | PID 4177174 — std_max=0.3, 7 entries, peak Lift=0.5% (step 35.3M), **peak R=+103.7** (new all-time record!). final.pt saved 12:37 |
| Run 10 | **RUNNING ~12:38 GMT+4** | PID **13723** — std_max=0.3, final.pt from Run 9, total_timesteps=50M |
| **Run 9 peak** | R=+103.7 new record | Step 39.2M: R=+103.7 — reward climbing as mean matures. Lift=0.5% transient at step 35.3M |
| Run 10 PID | **13723** | Remote PC konstantinsmirnov@10.127.102.40 |
| Run 10 WandB | **jtt5jmbv** (run-20260426_123853) | https://wandb.ai/skvayzer/System0_Blind |
| Run 10 checkpoint | final.pt (step ~40M, Run 9) | Peak R=103.7, evolving mean |
| Run 10 std | **0.3** | Stable productive regime — lift appears at ~5M steps |
| Run 10 steps | ~40M→50M (10M new) | ~25 min with warm cache |
| Run 10 target | **Lift ≥ 2%** | R trending up — mean maturing toward reliable grasp |
| **8jtj2m0s killed** | R=-161/ep, Ent=9.933 (pinned) | Bug 8: all-negative rewards → entropy trap |
| **qg9b59c4 killed** | R=-47/ep, Ent=14.785 (pinned) | Bug 7: entropy_coeff=0.01 → entropy trap |

---

## What to Watch

**Run ugfk3rr0 is HEALTHY as of 05:08. Breakthroughs confirmed:**
1. ✅ **Reward positive**: R=+60.5 → +47.1 → +67.6 — contact being found
2. ✅ **First lift event**: 0.5% at 3.93M steps
3. ⚠️ **Entropy pinned at 9.933** — std frozen at clamp max (1.0 rad, zero gradient). Mean is learning despite noisy actions. Acceptable — do NOT kill.
4. **Watch**: lift rate trend — target 1–5% over next 5–10M steps
5. **Watch**: reward stability — must stay ≥ 0/ep
6. **GPU OOM** — VRAM should stay under 40 GB (UnifoLM 16.4 + training ~20 GB)

### Kill criteria for PID 3907935
- Lift rate = 0% AND R < -50/ep sustained over 3 log entries → something regressed, diagnose
- R < -200/ep → catastrophic new bug, write CRITICAL_ASK.md

### Correct Monitoring Commands (run PID 3907935)
```bash
# Find new WandB run dir:
ssh konstantinsmirnov@10.127.102.40 "ls -lt ~/unitree_sim_isaaclab/wandb/ | head -4"

# Real-time training output (latest run dir):
ssh konstantinsmirnov@10.127.102.40 "grep 'Step\|Training started' ~/unitree_sim_isaaclab/wandb/run-*/files/output.log 2>/dev/null | tail -10"

# Process check:
ssh konstantinsmirnov@10.127.102.40 "ps -p 3907935 -o pid,etimes,stat --no-headers 2>/dev/null || echo DEAD"
```

---

## WandB Dashboard
- Current run (PID 3907935): https://wandb.ai/skvayzer/System0_Blind (find latest run after init)
- Previous runs: 8jtj2m0s (all-neg rewards trap), qg9b59c4 (entropy_coeff trap), yrugn52y (overforce collapse), lh8a264n (no --headless)

Log: `ssh konstantinsmirnov@10.127.102.40 "grep 'Step' ~/unitree_sim_isaaclab/wandb/run-*/files/output.log 2>/dev/null | tail -5"`

---

## Overnight Log
See `dev_diary/SYSTEM0_OVERNIGHT_LOG.md` for timestamped monitoring rows.

---

## Evening Update — 2026-04-26 ~22:39 GMT+4

### Run 16 — COMPLETED (all-time Lift record)
- **WandB:** yzz5f92o | **Steps completed:** ~150M
- **Peak Lift:** **2.0%** at Step 147.1M, R=+141.0 — new all-time record
- **final.pt saved:** confirmed 51MB at 15:15 (from task output)

### VPN Outage — ~18:20 GMT+4 (ongoing, 240+ min)
- GlobalProtect VPN disconnected; no route to 10.127 subnet
- Run 16 nohup training completed safely on remote PC
- Run 17 cannot launch until VPN restored
- 29 consecutive SSH failures; monitoring continues at 10-min intervals

### Real-Robot Tactile Alignment — COMPLETED (local code, awaiting human review)
- **Breaking change:** sim tactile contract updated 16D→18D to match verified real-robot mapping
- **Files changed:** tactile_state.py, rewards.py, config.py, train.py, SYSTEM0_FACTS.md
- **Key changes:**
  - DEX3_PAD_LINKS: 18 entries (removed thumb_2, palm split into 3 equal zones)
  - get_tactile_obs: (N,18); get_tactile_obs_extended: (N,72)
  - rewards.py: _R_ALL=[9..17], GRASP_THUMB_THR=0.30, slices updated
  - config.py: tactile_dim=72; train.py: obs 85→93, obs_with_targets 92→100
- **See:** `dev_diary/PIPELINE_REPORT_REAL_ALIGNED.md` for full change log and review checklist
- **Phase 7 (verification rollout) DEFERRED** — requires sim environment
- **CHECKPOINT INCOMPATIBILITY:** Runs 1-17 weights cannot be loaded after this change
- Human must approve + VPN restore + save pre_real_alignment.pt before first aligned run

### Pending When VPN Restores
1. `ssh konstantinsmirnov@10.127.102.40 "cp .../checkpoints/final.pt .../checkpoints/pre_real_alignment.pt"`
2. Launch Run 17: `nohup python experiments/system0_rl/train.py --headless --num_envs 2048 --checkpoint experiments/system0_rl/checkpoints/final.pt --total_timesteps 190000000 > /tmp/run17.log 2>&1 &`
   - Note: Run 17 uses OLD architecture (checkpoint from pre-alignment). This is correct — Run 17 is the GAE-fix smoke test per CRITICAL_ASK.md Option A.
3. After Run 17 completes: decide on real-alignment + MoE refactor (CRITICAL_ASK Issues 1+2+3 combined)

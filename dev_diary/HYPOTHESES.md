# System 0 Active Hypotheses
**Last updated:** 2026-04-26 04:20 GMT+4

This file tracks active hypotheses about training dynamics, reward shaping, and architecture choices.
Update as evidence is gathered from training runs.

---

## H1 — Contact baseline taring is sufficient to neutralize table overforce penalty
**Status:** PARTIALLY CONFIRMED + NEW ISSUE FOUND (2026-04-26 03:40)
**Hypothesis:** Baseline taring does subtract static hover forces — confirmed by output.log showing
R=-29.3/step (vs -1066/step before baseline = 36× improvement). Static forces ARE neutralized.
**BUT:** Dynamic forces from OU noise + policy movement are NOT baselined. Fingers move 0.3-0.5 rad
due to exploration → differential forces 10-14N → overforce penalty -29.3/step at OVERFORCE_COEFF=0.2.
**Fix applied:** OVERFORCE_COEFF 0.2 → 0.001. Expected: -0.128/step at 10N differential.
**New test:** With OVERFORCE_COEFF=0.001 and PID 3884414, check if mean reward turns positive at 50K steps.
**Kill condition:** If mean reward is still < -5/step (episode R < -2500) after 50K steps → further tuning needed.

---

## H2 — Policy will achieve first lift within 500K–2M steps
**Status:** UNTESTED  
**Hypothesis:** With correct reward signal (gentle contact +0.5, lift bonus +5.0, proportional +3×Δz),
the policy should discover the grasping behavior within 500K–2M steps.  
**Rationale:** 
- The block starts 2.2 cm above the table top (block_initial_z=0.819, table_top=0.797)
- The arm hover pose is already near the block — minimal exploration needed to contact it
- The gentle-contact channel (+0.5 per step) provides dense reward for any fingertip contact
- OU noise (σ=0.1, θ=0.15) provides exploration on top of the policy
**Evidence for:** Phase 2 showed 282N contact when hand closes at thumb_1_link — contact IS detectable.  
**Evidence against:** Block is on the table; fingers must approach from above without crushing it.  
**Test:** Monitor `reward/lift_rate` in WandB. First non-zero lift_rate is the key milestone.

---

## H3 — MoE gating will specialize across curriculum stages
**Status:** UNTESTED  
**Hypothesis:** The 8-expert MoE with curriculum intent encoding (one-hot in intent[:4]) will develop
expert specialization — different experts activate for different stages (gentle contact, firm grip, lift).  
**Evidence for:** Curriculum intent is passed as a separate 128D vector to the MoE router.  
**Evidence against:** With 512 envs and early-stage training (stage=0 for most of run), all experts
may collapse to the same behavior. Need to reach stage 1+ to see differentiation.  
**Test:** After reaching stage 1, check router entropy and load-balance metrics. If all experts have
similar activation frequency → no specialization yet.

---

## H4 — Table contacts do not interfere with block contact detection
**Status:** PARTIALLY VERIFIED  
**Hypothesis:** The 16 pads that show table contacts (pad 8=palm, pad 12=middle_0) will be suppressed
by the contact baseline, leaving clean signal from finger pads that contact the block.  
**Evidence for:** Phase 2 Phase B showed block contact on pad 15 (thumb_1), which is NOT one of the
table-contact pads. The 14 remaining pads are free to detect block contacts.  
**Evidence against:** If arm moves significantly during training, the baseline becomes stale.  
**Note:** The baseline is captured ONCE at episode start. If training causes the arm to hover at a
slightly different position over time, the baseline may drift. Monitor by checking if the overforce
penalty term grows during training.

---

## H5 — OU noise is sufficient for exploration without curriculum randomization
**Status:** UNTESTED  
**Hypothesis:** OUNoise (θ=0.15, σ=0.1) applied to the 7 right-hand joints provides enough
exploration to discover block contact and lift without explicit action randomization.  
**Evidence for:** OU noise is correlated (not i.i.d.) — it produces sweeping motions that are more
likely to explore the block's surface than pure Gaussian noise.  
**Evidence against:** δmax=0.5 rad limits exploration range. The block is small (~4 cm cube) and
must be contacted precisely.  
**Test:** If policy gets stuck at 0% lift after 2M steps, increase σ from 0.1 to 0.3 and rerun.

---

## H6 — 512 envs is sufficient sample efficiency for 10M step convergence
**Status:** UNTESTED  
**Hypothesis:** 512 parallel environments with 256 rollout steps gives 131K steps per update —
sufficient for PPO to converge within 10M total steps.  
**Evidence for:** Standard benchmark PPO on locomotion tasks converges in 5–20M steps with ~1K envs.
System 0 has a simpler task (single hand, 7 DOF, fixed arm).  
**Evidence against:** The sparse nature of the lift bonus (+5.0 once vs -1066/step before fix) may
require many episodes before the signal is seen.  
**Alternative:** If not converging, consider dense reward shaping: reward block height change per step
rather than just > LIFT_DELTA binary.

---

## H7 — entropy_coeff=0.01 causes entropy trap at sparse rewards
**Status:** CONFIRMED + FIXED (2026-04-26 04:20)
**Hypothesis:** When `entropy_coeff × dH/d(log_std)` > `reward gradient`, policy maximizes std until it hits `clamp(max)`, where gradient = 0 → stuck.
**Evidence:** Run qg9b59c4 (OVERFORCE_COEFF=0.001): entropy=14.785 PINNED at step 1.31M and 2.62M; std at clamp max 2.0; reward -37→-47/ep (declining).
**Fix applied:** `entropy_coeff: 0.01→0.001` (10× reduction); `std clamp max: 2.0→1.0` (prevents zero-gradient at max). PID 3896359 launched.
**Expected:** Initial entropy ≤ 3 nats (std=e^(-1)=0.368); reward gradient dominates; entropy decreases as policy learns contact.
**Kill condition:** If entropy ≥ 10 at first metrics AND reward < -100/ep → entropy trap persists → escalate to CRITICAL_ASK.

---

## Resolved Hypotheses

### RH1 — ContactSensor filter_prim_paths_expr has inclusion semantics [WRONG]
**Resolved:** 2026-04-26 02:30  
**Conclusion:** Semantics are EXCLUSION (denylist). Filter suppressed block contacts. Removed.

### RH2 — env.scene key for block is "red_block" [WRONG]
**Resolved:** 2026-04-26 02:35  
**Conclusion:** Key is `"block"` in `BlockStackEnvCfg`. `"red_block"` only exists in
`TableRedGreenYellowBlockSceneCfg` (unused by current train.py). 

### RH3 — Arm hover is stable — table contacts are constant [CORRECT]
**Resolved:** 2026-04-26 02:30  
**Conclusion:** Phase 2 Phase A confirmed palm=75.7N, middle_0=46.8N at idle — stable and constant,
suitable for one-time baseline taring.

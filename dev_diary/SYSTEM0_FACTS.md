# System 0 — Verified Facts (append-only)

Last updated: 2026-04-26 (real-robot tactile alignment — 16D→18D)

---

## Sensor / Hardware Facts

### Dex3 Press Sensor — Real Robot
- **IDL file**: `~/unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py`
- `HandState_.press_sensor_state`: `types.sequence[PressSensorState_]` — variable-length
- **`_PressSensorState_.py`**: `pressure: types.array[types.float32, 12]`, `temperature: types.array[types.float32, 12]`
- Each hand has **9 PressSensorState modules** — VERIFIED from real-robot smoke test MODULE_LABELS
- Each module has 12 raw FSR readings in `pressure[12]`
- Per-module scalar: `sum(press_sensor_state[i].pressure[0:12])`
- Idle noise: ~20-30 ADC counts; clear touch: ≥100 ADC counts
- DDS topic: `rt/dex3/{left,right}/state`
- **No thumb_2 sensor on real hardware** — sim's `thumb_2_link` is a fiction; excluded from tactile contract

### Dex3 URDF Link Names (right hand)
- Source: `unitree_lerobot/eval_robot/assets/unitree_hand/unitree_dex3_right.urdf`
- Left hand: replace `right_` with `left_`

| Body Part | Link Name |
|---|---|
| Palm | `right_hand_palm_link` |
| Thumb proximal | `right_hand_thumb_0_link` |
| Thumb mid | `right_hand_thumb_1_link` |
| Thumb distal | `right_hand_thumb_2_link` |
| Middle proximal | `right_hand_middle_0_link` |
| Middle distal | `right_hand_middle_1_link` |
| Index proximal | `right_hand_index_0_link` |
| Index distal | `right_hand_index_1_link` |

- Additional non-contact links in URDF (do NOT add to ContactSensor): `base_link_thumb`, `base_link_index`, `base_link_middle`, `base_link`, `thumb_tip`, `index_tip`, `middle_tip`
- **`thumb_2_link` exists in URDF** but has NO real sensor — excluded from `DEX3_PAD_LINKS`
- **Tactile contract per hand: 9 modules** (palm×3 zones + thumb×2 + middle×2 + index×2)
- **Total tactile modules both hands: 18**

### Tactile Observation Dimensions — ALIGNED 2026-04-26
- **Real robot**: 9 modules per hand = **18D total**
- **Sim `get_tactile_obs()`** → **(N, 18)** — 18 entries in `DEX3_PAD_LINKS` order
  - Palm body link appears 3x (palm_0/1/2); each zone = palm_force / 3 (equal-split)
  - `thumb_2_link` excluded (no real sensor)
- `get_tactile_obs_extended()` → **(N, 72)** (4 channels x 18 pads)
  - pressure[0:18] | binary[18:36] | delta[36:54] | duration[54:72]
- **Sim->real mismatch: RESOLVED.** Both sim and real output 18D in the same module order.
- Palm spatial split: real SDK uses contact positions; sim uses equal-split (palm_total/3).

Module order (left hand slots 0-8, right hand slots 9-17):
| Slot (L/R+9) | Module | Sim body | Notes |
|---|---|---|---|
| 0 | palm_0 (middle side) | `{hand}_palm_link` | equal-split /3 |
| 1 | palm_1 (centre) | `{hand}_palm_link` | equal-split /3 |
| 2 | palm_2 (index side) | `{hand}_palm_link` | equal-split /3 |
| 3 | thumb_0 (proximal) | `{hand}_thumb_0_link` | — |
| 4 | thumb_1 (tip) | `{hand}_thumb_1_link` | — |
| 5 | middle_0 (proximal) | `{hand}_middle_0_link` | — |
| 6 | middle_1 (tip) | `{hand}_middle_1_link` | — |
| 7 | index_0 (proximal) | `{hand}_index_0_link` | — |
| 8 | index_1 (tip) | `{hand}_index_1_link` | — |

---

## Code Locations

### CraftNet (act_craftnet policy)
- **Primary file**: `unitree_IL_lerobot/unitree_lerobot/lerobot/src/lerobot/policies/act_craftnet/modeling_act_craftnet.py`
- **System0 MoE class**: `_System0Policy` at line ~317
- **MoE forward pass**: `_System0Policy.forward_delta()` at line ~361
- **encode_feedback (Phase 1)**: `_System0Policy.encode_feedback()` at line ~356
- **delta_finger construction**: line ~539: `delta_finger = self.system0.forward_delta(mem[0], physical_intent, tactile, finger_state, s1_fingers)`
- **delta_finger injection**: lines 544-545:
  - `actions_hat[:, :, 7:14]  += delta_finger[:, :, :7]`   (LEFT fingers)
  - `actions_hat[:, :, 21:28] += delta_finger[:, :, 7:]`   (RIGHT fingers)
  - **NOTE**: Brief says `14:28` but actual code is `7:14` + `21:28` — needs clarification
- **Tactile obs entry**: line ~488: `tactile = batch["tactile"].float()` (18D)
- **Tactile enters MoE**: line ~366 in `forward_delta()` as `ta` variable
- **Gating network**: `self.router = nn.Linear(686, 4)` (line ~338)
- **Expert architecture**: `Linear(686→256) → ReLU → Linear(256→14)`, 4 experts (line ~339-342)
- **feedback_encoder**: `Linear(18+14=32→128) → ReLU → Linear(128→64)` (line ~344-347)
- **tactile_feedback_proj (S0→S1)**: `nn.Linear(64, 512)`, near-zero init (line ~413)
- **physical_intent_proj (S1→S0)**: `nn.Linear(512, 128)` (line ~410)
- **`tactile_dim`**: 18 (line ~182 in `_ModelCfg`)

### Isaac Lab (sim-side)
- **Dex3 stack task env cfg**: `tasks/g1_tasks/stack_rgyblock_g1_29dof_dex3/stack_rgyblock_g1_29dof_dex3_joint_env_cfg.py`
- **ContactSensorCfg**: EXISTS at line 48 as `fingertip_contacts = ContactSensorCfg(...)`
- **Tactile obs function**: `tasks/common_observations/tactile_state.py` — `get_tactile_obs()` returns 18D, `get_tactile_obs_extended()` returns 72D
- **`activate_contact_sensors=False`** in: `tasks/common_scene/base_scene_pick_redblock_into_drawer.py:93`
- **Standalone RL train**: `experiments/system0_rl/train.py` — uses `BlockStackEnvCfg` (NOT `System0TrainEnvCfg`)

### Standalone RL (unitree_sim_isaaclab)
- **Policy**: `experiments/system0_rl/system0_moe.py` — MoE with 8 experts, top-2, 4.40M params
- **Rewards**: `experiments/system0_rl/rewards.py` — `compute_reward_blind()`, `is_lift_success()`
- **Config**: `experiments/system0_rl/config.py` — `TrainConfig`
- **Env cfg**: `experiments/system0_rl/env_cfg.py` — `System0TrainEnvCfg`
- **Train**: `experiments/system0_rl/train.py`

---

## Hyperparams in Use

### Standalone RL (current run)
| Param | Value |
|---|---|
| num_envs | 2048 |
| total_timesteps | 10,000,000 |
| rollout_steps | 256 |
| ppo_epochs | 4 |
| minibatch_size | 512 |
| lr | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_eps | 0.2 |
| entropy_coeff | **0.001** | was 0.01 → entropy trap at 14.785 nats (qg9b59c4, Bug 7) |
| delta_max | 0.5 rad |
| ou_theta | 0.15 |
| ou_sigma | 0.10 |
| curriculum_success_threshold | 0.75 |

### Reward coefficients (current — v3, 2026-04-26 03:42)
| Term | Value | Notes |
|---|---|---|
| gentle_contact | +0.5 × max pad in [0.1–1.5N] | — |
| overforce | −**0.001** × Σ (f−2.0)² | was 0.2 → caused policy collapse; see Bug 6 |
| lift_bonus | +5.0 binary (block_z > 3cm) | — |
| lift_prop | +3.0 × Δz | — |
| smoothness | −0.002 × Σ|Δv| | — |

**Reward history**: v1 had wrong scene key (0 lift reward); v2 had OVERFORCE_COEFF=0.2 (policy collapse at -29.3/step); v3 had OVERFORCE_COEFF=0.001 but entropy_coeff=0.01 → entropy trap (pinned 14.785 nats); v4 entropy_coeff=0.001, std_max=1.0 but STILL entropy trap (all-neg rewards → all-neg advantages → std→max regardless of entropy_coeff); **v5 is current** (OVERFORCE_COEFF=0.000 — overforce removed entirely, R≥0 enables positive advantages).

**Policy std clamp**: `system0_moe.py` `System0MoEActor` and `RLSystem0Actor`: `std = log_std.exp().clamp(max=**0.3**)` — was 2.0→1.0→0.3. At 0.3: max entropy = 1.52 nats, entropy bonus = 0.00152/step << contact +0.5/step → policy can reduce std when mean near contact.

---

## Checkpoint Paths

| Date | Stage | Success | Host | Path |
|---|---|---|---|---|
| 2026-04-25 | 0 | 0.0% | Remote RTX6000 Ada | `~/unitree_sim_isaaclab/experiments/system0_rl/checkpoints/final.pt` |
| 2026-04-26 | 0 | 0.5% lift | Remote RTX6000 Ada | PID 3907935, WandB **ugfk3rr0**, run-20260426_042221 — first lift event at 3.93M steps |
| 2026-04-26 | 0 | 1.0% lift | Remote RTX6000 Ada | PID 3943390, WandB **v0sfb5q3**, run-20260426_052239 — std_max=0.3, loaded from step_6553600 |
| 2026-04-26 | 0 | 0.0% lift | Remote RTX6000 Ada | PID 3986450, WandB **lfkzq61k**, run-20260426_063133 — std_max=0.1, 9.9M new steps (10.09M→20M), Lift=0% all 7 entries, R=+72–93. final.pt saved 07:18 |
| 2026-04-26 | 0 | **0.5% lift** peak | Remote RTX6000 Ada | PID 4017579, WandB **rmx1a3i3** — std_max=0.1, 10M new steps (20M→30M), peak Lift=0.5% at step 27.9M (R=+98.3). final.pt saved 08:11 |
| 2026-04-26 | 0 | **0.5% lift** peak | Remote RTX6000 Ada | PID 4177174, WandB **tvyr7hld**, run-20260426_114938 — std_max=0.3, 40M total steps (10M new), peak Lift=0.5% at step 35.3M, **peak R=+103.7** (new record). final.pt saved 12:37 |
| 2026-04-26 | 0 | **0.5% lift** peak | Remote RTX6000 Ada | PID 13723, WandB **jtt5jmbv**, run-20260426_123853 — std_max=0.3, total_timesteps=50M, loaded from Run 9 final.pt. Run 10. Lift=0.5% entry 1 only, peak R=89, COMPLETED 13:25. |
| 2026-04-26 | 0 | **KILLED** | Remote RTX6000 Ada | PID 42891, WandB **wa1lzxrd** — phase-aware reward v2, OVERFORCE_COEFF=0.05 (BUG). R=−66K/ep from residual middle_0 force 56N → −130/step overforce. 15M corrupted steps. Killed 14:04. |
| 2026-04-26 | 0 | **RUNNING** | Remote RTX6000 Ada | PID 67676, WandB TBD — **GAE fix + phase-aware reward + OVERFORCE=0.001 + log1p + per-env block_z**. Run 12. Loaded from pre_phase_aware_reward.pt (Run 9). Smoke test: expect lift_rate >5% in 5M steps. |

---

## CraftNet delta_finger — Verified Ordering

`s1_fingers = cat([actions_hat[:, :, 7:14], actions_hat[:, :, 21:28]])` → `[left_7, right_7]`

Injection (lines 544–545 of `act_craftnet/modeling_act_craftnet.py`):
- `delta_finger[:, :, :7]` → `actions_hat[:, :, 7:14]`  (left hand)
- `delta_finger[:, :, 7:]` → `actions_hat[:, :, 21:28]` (right hand)

**Brief error**: spec said `[14:28]`; actual is `[7:14]` + `[21:28]`. Code is correct.

---

## Scope Decision — Option B (2026-04-25)

**CraftNet (`act_craftnet/`) is READ-ONLY** for all Phase 2–11 work.

The 96D tactile feature vector and new MoE live in the **standalone RL** (`experiments/system0_rl/`) as a separate `RLSystem0Policy`. No checkpoint sharing with CraftNet. Rationale: preserves working CraftNet pipeline; 96D cross-step features only make sense in closed-loop RL.

### RLSystem0Policy Input Dimensions

| Slot | Dim | Notes |
|---|---|---|
| Tactile features (4 ch × 18 sim pads) | 72 | pressure+binary+delta+duration per pad |
| Right finger torques | 7 | right hand only (matches action space) |
| Right finger qpos | 7 | right hand only |
| Left finger torques | 7 | observe both hands |
| Left finger qpos | 7 | observe both hands |
| Physical intent (zeroed during pure RL) | 128 | kept for future joint-training compat |

**Total tactile feature vector**: 72 + 7 + 7 + 7 + 7 = **100D** (18 real-aligned pads × 4 channels = 72; sim now matches real)

**Gating network input**: 100D features + 128D intent = **228D**

**Per expert**: `Linear(228→256) → ReLU → Linear(256→14)` (14D = 7L + 7R fingers)

**N_EXPERTS**: 4, TOP_K: 2 (dense soft gating, all experts always evaluated)

---

## Sim→Real Tactile Pad Mapping — VERIFIED 2026-04-26

**Status**: ALIGNED — sim and real now use identical 18D layout.

Verified module index → sim body (per hand; duplicate palm entries use equal-split /3):

| Module idx | Real module (VERIFIED) | Sim body | Sim slot (L/R+9) |
|---|---|---|---|
| 0 | palm_0 (middle side) | `{hand}_palm_link` (÷3) | 0 / 9 |
| 1 | palm_1 (centre) | `{hand}_palm_link` (÷3) | 1 / 10 |
| 2 | palm_2 (index side) | `{hand}_palm_link` (÷3) | 2 / 11 |
| 3 | thumb_0 (proximal) | `{hand}_thumb_0_link` | 3 / 12 |
| 4 | thumb_1 (tip) | `{hand}_thumb_1_link` | 4 / 13 |
| 5 | middle_0 (proximal) | `{hand}_middle_0_link` | 5 / 14 |
| 6 | middle_1 (tip) | `{hand}_middle_1_link` | 6 / 15 |
| 7 | index_0 (proximal) | `{hand}_index_0_link` | 7 / 16 |
| 8 | index_1 (tip) | `{hand}_index_1_link` | 8 / 17 |

**No zero-padding required.** All 9 real modules have corresponding sim entries.
Palm zones use equal-split (sim approximation); real SDK provides spatial contact positions.

---

## Phase 2 Changes (2026-04-25)

- `block_stack_env.py` `fingertip_contacts`: added `filter_prim_paths_expr=["/World/envs/env_.*/red_block"]`, `history_length` 1→2
- Rationale: without filter, PhysX sends ALL contact pairs (including finger self-contacts) to sensor, corrupting tactile reward signal

---

## Known Issues & Workarounds

1. ~~**Bug**: `env.scene["block"]`~~ **FIXED 2026-04-25** — renamed to `"red_block"` in `rewards.py` (both functions) and `apply_curriculum()` in `train.py`.
2. ~~**Brief says `[14:28]`**~~ **CLARIFIED** — code at `[7:14]`+`[21:28]` is correct; brief had error.
3. ~~**Dim mismatch**~~ **FIXED 2026-04-26**: `get_tactile_obs()` now returns 18D (9 modules/hand × 2 = 18). DEX3_PAD_LINKS rewritten with 3 palm zones (equal-split) and no thumb_2. `get_tactile_obs_extended()` now 72D. config.py `tactile_dim=72`. Obs dim 85→93, obs_with_targets 92→100, MoE input 220→228. Requires fresh checkpoint (breaks compat with Run 1-17 weights).
4. **`activate_contact_sensors`**: confirmed `True` in `BlockStackEnvCfg` at line 60. No action needed.
5. **RLSystem0Policy COMPLETE** (Phase 3, 2026-04-26, updated 2026-04-26 real-align): `experiments/system0_rl/system0_moe.py` — 4 experts, flat MoE. After real-align: obs_dim=100 (tactile_72|r_torq_7|r_qpos_7|l_torq_7|l_qpos_7), gate_dim=228. `build_rl_system0_obs()` in train.py now returns (N,100). Wire in next run by replacing `System0PPOWrapper` + `build_obs_batch()` with `RLSystem0Policy` + `build_rl_system0_obs()` + `build_left_joint_index_map()`. Current runs (1-17) use `System0PPOWrapper` (8 experts, 4.4M params) — checkpoint incompatible after real-align.
6. **UnifoLM server** on remote PC uses 16592 MiB / 49140 MiB GPU. ~32GB free for RL training.
7. **Phase 2 verification still needed**: run 100-step scripted closing test — open hand should give all-zero contacts; closed on cube should give nonzero on contact pads.

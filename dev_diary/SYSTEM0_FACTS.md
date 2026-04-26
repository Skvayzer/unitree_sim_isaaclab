# System 0 — Verified Facts (append-only)

Last updated: 2026-04-25

---

## Sensor / Hardware Facts

### Dex3 Press Sensor — Real Robot
- **IDL file**: `~/unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py`
- `HandState_.press_sensor_state`: `types.sequence[PressSensorState_]` — variable-length
- **`_PressSensorState_.py`**: `pressure: types.array[types.float32, 12]`, `temperature: types.array[types.float32, 12]`
- Each hand has **9 PressSensorState modules** (inferred from `tactile_dim=18` in code; 9×2=18)
- Each module has 12 raw FSR readings in `pressure[12]`
- DDS topics:
  - `rt/dex3/{left,right}/state` — main HF hand state (joints + tactile)
  - `rt/lf/dex3/{left,right}/state` — dedicated low-frequency tactile-only hand state. Same `HandState_` struct, same `press_sensor_state` field; preferred by `xr_teleoperate/.../robot_hand_unitree.py` for tactile reads. **Topics share the same module ordering** — switching topics does NOT change the mapping.
- **VERIFIED 2026-04-26 (real-robot smoke test)** — count = 9 modules per hand. **The real Dex3 has only 2 thumb sensors (no thumb-proximal sensor) and 3 palm sensors**, contrary to the prior assumption of "1 palm + 3 thumb + 2 middle + 2 index". The mapping is hand-specific for the palm because the firmware enumerates palm sensors in opposite directions on left vs. right (mirrored wiring).

| SDK index | LEFT hand pad | RIGHT hand pad |
|-----------|---------------|----------------|
| m0 | thumb_0 | thumb_0 |
| m1 | thumb_1 | thumb_1 |
| m2 | middle_0 | middle_0 |
| m3 | middle_1 | middle_1 |
| m4 | index_0 | index_0 |
| m5 | index_1 | index_1 |
| m6 | palm_2 (at index side) | palm_0 (next to middle finger) |
| m7 | palm_1 (between fingers, centre) | palm_1 (between fingers, centre) |
| m8 | palm_0 (next to middle finger) | palm_2 (at index side) |

Palm naming convention (same on both hands): `palm_0` = next to middle-finger side; `palm_1` = centre between fingers; `palm_2` = at index-finger side. The SDK assigns m6→m8 in the index→middle direction on LEFT, and middle→index on RIGHT.

**Sim/real mismatch implications**:
- Sim's `DEX3_PAD_LINKS` (in `tasks/common_observations/tactile_state.py`) includes `thumb_0_link` (proximal), but the real robot has no tactile sensor there — sim trains on a phantom pad. Either remove `thumb_0_link` from sim training or zero-fill m0 on the sim→real adapter.
- Sim has 1 palm pad (`palm_link`); real has 3 (m6/m7/m8). Sim is missing 2 dimensions vs. real. Either add 2 palm sub-zones to the URDF / contact sensor or fold the 3 real palm signals into 1 in the real→sim adapter.
- Authoritative mapping lives in `MODULE_LABELS` in `experiments/system0_rl/tools/real_robot_tactile_smoke_test.py` — keep the two in sync if the table changes.

#### Verified IDL imports (2026-04-26, read directly from installed SDK)
- `from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, PressSensorState_`
- `from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize`
- `ChannelFactoryInitialize(domainId: int = 0, networkInterface: str | None = None)` — call once before creating subscribers
- `ChannelSubscriber(name: str, type)` → `.Init(handler: Callable, queueLen: int = 0)` registers a callback; `.Read(timeout=None)` polls instead; `.Close()` to clean up
- `HandState_` full fields (in declared order): `motor_state: sequence[MotorState_]`, `press_sensor_state: sequence[PressSensorState_]`, `imu_state: IMUState_`, `power_v: float32`, `power_a: float32`, `system_v: float32`, `device_v: float32`, `error: uint32[2]`, `reserve: uint32[2]`
- `PressSensorState_` full fields: `pressure: float32[12]`, `temperature: float32[12]`, `lost: uint32`, `reserve: uint32` — `lost` is a packet-loss counter per module, useful for liveness checks

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
- **Total pad links per hand: 8** (palm + 3 thumb + 2 middle + 2 index)
- **Total both hands: 16 links**

### Tactile Observation Dimensions
- Real robot: 9 modules × 1 scalar each = **9D per hand, 18D total**
- Sim contact sensor: monitors **18 bodies** (confirmed run qg9b59c4 output.log), but `get_tactile_obs()` selects only the 16 `DEX3_PAD_LINKS` — `left_hand_camera_base_link` (sensor idx 0) and `right_hand_camera_base_link` (sensor idx 9) are intentionally excluded.
  - Actual `get_tactile_obs()` output: **(N, 16)** — 8 links/hand in `DEX3_PAD_LINKS` order
  - Sensor body layout: camera_base_link at idx 0/9; palm_link at 1/10; index_0 2/11; index_1 3/12; middle_0 4/13; middle_1 5/14; thumb_0 6/15; thumb_1 7/16; thumb_2 8/17
- `get_tactile_obs_extended()` → **(N, 64)** (4 channels × 16 pads)
- CraftNet code uses `tactile_dim=18` — sim outputs 16D. **Mismatch still present (see Known Issues #3).**
- Resolution path: add `{hand}_camera_base_link` to `DEX3_PAD_LINKS` to bring sim to 18D, OR zero-pad module 8 per hand on real-robot adapter.

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
- **Tactile obs function**: `tasks/common_observations/tactile_state.py` — `get_tactile_obs()` returns 16D, `get_tactile_obs_extended()` returns 64D
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
| Tactile features (4 ch × 16 sim pads) | 64 | binary+logp+edges+duration per pad |
| Right finger torques | 7 | right hand only (matches action space) |
| Right finger qpos | 7 | right hand only |
| Left finger torques | 7 | observe both hands |
| Left finger qpos | 7 | observe both hands |
| Physical intent (zeroed during pure RL) | 128 | kept for future joint-training compat |

**Total tactile feature vector**: 64 + 7 + 7 + 7 + 7 = **92D** (not 96 as in brief; brief formula counts 4×18=72 assuming 18 sim pads, but sim has 16 → 4×16=64; recomputed correctly)

**Gating network input**: 92D features + 128D intent = **220D**

**Per expert**: `Linear(220→256) → ReLU → Linear(256→14)` (14D = 7L + 7R fingers)

**N_EXPERTS**: 4, TOP_K: 2 (dense soft gating, all experts always evaluated)

---

## Sim→Real Tactile Pad Mapping

**Decision**: 8 sim pads per hand (Option B — no URDF surgery). Documented gap: real robot has 9 modules/hand.

Real-robot module index → sim link:

| Module idx | Real module (inferred) | Sim link |
|---|---|---|
| 0 | palm zone A | `{hand}_palm_link` |
| 1 | index proximal | `{hand}_index_0_link` |
| 2 | index distal | `{hand}_index_1_link` |
| 3 | middle proximal | `{hand}_middle_0_link` |
| 4 | middle distal | `{hand}_middle_1_link` |
| 5 | thumb proximal | `{hand}_thumb_0_link` |
| 6 | thumb mid | `{hand}_thumb_1_link` |
| 7 | thumb distal | `{hand}_thumb_2_link` |
| 8 | palm zone B | **NO SIM LINK** — zero-padded in sim→real adapter |

Real-robot deployment: 9-element vector per hand; index 8 is zeroed by sim-trained policy.
**TODO**: revisit if palm zone B turns out critical for real-robot grasps.

---

## Phase 2 Changes (2026-04-25)

- `block_stack_env.py` `fingertip_contacts`: added `filter_prim_paths_expr=["/World/envs/env_.*/red_block"]`, `history_length` 1→2
- Rationale: without filter, PhysX sends ALL contact pairs (including finger self-contacts) to sensor, corrupting tactile reward signal

---

## Known Issues & Workarounds

1. ~~**Bug**: `env.scene["block"]`~~ **FIXED 2026-04-25** — renamed to `"red_block"` in `rewards.py` (both functions) and `apply_curriculum()` in `train.py`.
2. ~~**Brief says `[14:28]`**~~ **CLARIFIED** — code at `[7:14]`+`[21:28]` is correct; brief had error.
3. **Dim mismatch (open)**: `get_tactile_obs()` returns 16D (8 links/hand × 2); real robot = 18D. Sensor covers 18 bodies but camera_base_link pads are excluded by `DEX3_PAD_LINKS`. Fix: add camera_base_link to `DEX3_PAD_LINKS` to reach 18D, OR zero-pad module 8 per hand in real-robot adapter. CraftNet stays at 18D — not our concern (read-only).
4. **`activate_contact_sensors`**: confirmed `True` in `BlockStackEnvCfg` at line 60. No action needed.
5. **RLSystem0Policy COMPLETE** (Phase 3, 2026-04-26): `experiments/system0_rl/system0_moe.py` — 4 experts, flat MoE, 0.358M params, obs_dim=92 (tactile_64|r_torq_7|r_qpos_7|l_torq_7|l_qpos_7), gate_dim=220. **obs_dim=92 is correct**: `get_tactile_obs_extended()` returns 64D (4 channels × 16 pads; camera_base_link excluded by `DEX3_PAD_LINKS`). `build_rl_system0_obs()` added to train.py. Wire in next run by replacing `System0PPOWrapper` + `build_obs_batch()` with `RLSystem0Policy` + `build_rl_system0_obs()` + `build_left_joint_index_map()`. Current run uses `System0PPOWrapper` (8 experts, 4.4M params).
6. **UnifoLM server** on remote PC uses 16592 MiB / 49140 MiB GPU. ~32GB free for RL training.
7. **Phase 2 verification still needed**: run 100-step scripted closing test — open hand should give all-zero contacts; closed on cube should give nonzero on contact pads.

# System 0 — Blind Tactile Grasping RL Pipeline
**For expert review. Written 2026-04-26. Prepared by autonomous researcher.**

---

## 1. Goal

Train a policy to pick up a 4×4×4 cm block from a table using only tactile feedback (no cameras, no proprioceptive arm state). The robot is a Unitree G1 humanoid with a Dex3 three-finger hand (thumb + index + middle, 7 DOF per hand). Only the 7 right-hand finger joints are controlled by the RL policy. The arm is held at a fixed hover pose by high-stiffness actuators in the simulation.

**Success criterion:** Block rises ≥3 cm from its initial resting height.

**Current state (Run 11, step ~1M as of writing):** 10 consecutive runs across ~100M total training steps have not achieved reliable lifting. Peak historic performance: 1.0% lift rate (Run 6/v0sfb5q3), with most runs stuck at 0–0.5%. The policy consistently learns to touch blocks gently but never forms a grasp. The phase-aware reward (Run 11 onwards) is a new attempt to break this local optimum.

---

## 2. Environment — `BlockStackEnvCfg`

**File:** `experiments/system0_skills/block_stack_env.py`

### Scene
- **Robot:** G1 + Dex3, `g1_29dof_with_dex3_base_fix.usd`. Base fixed to floor. 
- **Block:** Single `4×4×4 cm` rigid body on table. Key: `env.scene["block"]`.
- **Table:** Static kinematic cuboid.
- **Contact sensor:** `fingertip_contacts`, prim_path pattern `.*_hand_.*_link`. Covers **all** hand links (including camera_base_link joints). The policy uses only the 16 pad links via `DEX3_PAD_LINKS` mapping in `tactile_state.py`.

### Robot initial state
Arm joints are set to a fixed hover pose that positions the right hand directly above the block. High stiffness (PD) actuators hold the arm in place — the RL policy cannot move the arm, only the fingers.

```
Right-hand finger joint names (7 DOF, RIGHT_HAND_NAMES in train.py):
  right_hand_thumb_0_joint   (proximal)
  right_hand_thumb_1_joint   (mid)
  right_hand_thumb_2_joint   (tip)
  right_hand_middle_0_joint  (proximal)
  right_hand_middle_1_joint  (tip)
  right_hand_index_0_joint   (proximal)
  right_hand_index_1_joint   (tip)
```

### Block reset
On each episode reset, the block is teleported to a uniform random XY offset within `±5 cm` of its nominal position (declared in `EventCfg`). The curriculum can add additional XY offset on top (see Section 7).

### Termination
Episode ends when:
- Block falls below table (`z < 0.75 m`) or drifts beyond arm reach
- Episode length exceeds `20 s` (at `dt=0.005 s` × `decimation=2` → 100 Hz control, 2000 steps/episode)

### PhysX settings (relevant to grasping)
```
num_substeps            = 4
num_position_iterations = 16    (high, good for contact)
num_velocity_iterations = 4
contact_offset          = 0.01 m
rest_offset             = 0.001 m
friction_correlation    = 0.003 m
enable_ccd              = True
```

---

## 3. Observation Space — 85D (+ 7D coarse targets → 92D input to policy)

**File:** `train.py:build_obs_batch()`

| Slice | Dim | Content |
|---|---|---|
| `[0:7]`   | 7  | Right finger `joint_pos` (radians) |
| `[7:14]`  | 7  | Right finger `joint_vel` (rad/s) |
| `[14:78]` | 64 | Extended tactile observation (see below) |
| `[78:85]` | 7  | Right finger `applied_torque` (Nm) |

After building, `coarse_targets` (7D, zeros during training) is concatenated → 92D `obs_with_targets` passed to the policy.

### Extended Tactile Observation (64D) — `get_tactile_obs_extended()`

**File:** `tasks/common_observations/tactile_state.py`

4 channels × 16 pads = 64D:

| Channel | Slice | Content |
|---|---|---|
| `pressure`  | `[0:16]`  | Force magnitude / 10.0 N, clipped to [0,1] |
| `binary`    | `[16:32]` | 1 if force > 0.05 N else 0 |
| `delta`     | `[32:48]` | +1 new contact, -1 lost, 0 stable (compared to previous step) |
| `duration`  | `[48:64]` | Steps continuously in contact / 50, clipped to [0,1] |

**Pad index order in 16D** (`DEX3_PAD_LINKS` order):
```
[0-7]  Left hand:  palm, thumb0, thumb1, thumb2, middle0, middle1, index0, index1
[8-15] Right hand: palm, thumb0, thumb1, thumb2, middle0, middle1, index0, index1
```

**Right-hand only subset** used in rewards:
```python
_R_ALL = [8, 9, 10, 11, 12, 13, 14, 15]
# Within f_right (N×8):
# [0]=palm  [1]=thumb0  [2]=thumb1  [3]=thumb2
# [4]=middle0  [5]=middle1  [6]=index0  [7]=index1
```

**Contact baseline taring:** On startup, the arm is held at hover pose and 5 warmup steps are run. `set_contact_baseline()` captures the idle contact forces (palm and middle_0 press against the table/block at rest). All subsequent reward computations use `(f_raw - baseline).clamp(min=0)` to get differential forces. The baseline is captured **once at startup** and never updated per-episode.

**Known issue with tactile state:**
The `delta` and `duration` channels use global mutable tensors `_prev_binary` and `_contact_duration`. These are reset on episode termination via `reset_tactile_state(env_ids)`. However, the delta channel is computed as `binary - _prev_binary` BEFORE `_prev_binary` is updated — so the first step after a reset always shows `delta = binary_t0 - 0 = binary_t0`. This is a one-step artifact but probably harmless.

---

## 4. Action Space — 7D

**File:** `train.py:main()`, `block_stack_env.py:ActionsCfg`

```
action = tanh(policy_raw_delta) * delta_max + OU_noise
action = clamp(action, -delta_max, delta_max)
```

`delta_max = 0.5` rad. The env's `JointPositionActionCfg` has `scale=1.5, use_default_offset=True`, so effective joint displacement = `action × 1.5` radians added to default pose. Maximum finger displacement ≈ `0.5 × 1.5 = 0.75 rad` from default.

**OU exploration noise:** `theta=0.15, sigma=0.10`. Noise is reset to zero for each env on episode reset. This provides temporally correlated exploration.

---

## 5. Reward Function — Phase-Aware v2 (Run 11+)

**File:** `experiments/system0_rl/rewards.py`

Designed to break the "gentle contact" local optimum. Previous reward gave continuous positive signal just for light fingertip touch (~250/ep), making it easy to earn reward without grasping.

### Phase detection
```python
f_right = (f_raw[:, _R_ALL] - baseline[:, _R_ALL]).clamp(0)  # (N, 8)
active_r   = (f_right > 0.10 N).sum(dim=1)   # number of pads with contact
no_contact = (active_r == 0)
palpating  = (active_r >= 1) & (active_r <= 2)
# force-closure phase: active_r >= 3 (implicit)
```

### Phase 1 — Search
```python
r_search = 0.05 * no_contact.float()
```
Tiny reward to keep exploration when not touching anything.

### Phase 2 — Palpation (1–2 pads active only)
```python
in_window   = (f_right - 0.10).clamp(0)
normed      = (in_window / 1.40).clamp(max=1.0)
best_gentle = normed.max(dim=1).values
r_palpation = 0.30 * best_gentle * palpating.float()
```
Gates OUT when ≥3 pads engage. Gradient always points toward more finger contact.

### Phase 3 — Force Closure (≥3 pads active)
```python
thumb_force    = f_right[:, 1:4].sum(dim=1)   # thumb0+1+2
opposing_force = f_right[:, 4:8].sum(dim=1)   # middle0+1 + index0+1
opposition     = torch.minimum(thumb_force, opposing_force)
r_closure      = 1.00 * torch.tanh(opposition / 3.00)
```
`min()` ensures BOTH sides must press simultaneously. `tanh` provides smooth saturation.

### Phase 4 — Lift (gated on has_grasp)
```python
# has_grasp: thumb_max > 0.5 N AND (index_max > 0.5 N OR middle_max > 0.5 N)
has_grasp = _check_force_closure(f_right)  # (N,) bool

lift_delta   = (block_z - 0.819 m).clamp(0)   # 0.819 = nominal block top
r_lift_prop  = 20.0 * lift_delta * has_grasp.float()
r_lift_bonus = 50.0 * (lift_delta > 0.03).float() * has_grasp.float()
```
**Sky-hook exploit prevention:** Without `has_grasp` gate, the policy could raise its wrist and drag the block upward by friction (palm/wrist contact). The gate requires a real pinch.

### Overforce penalty
```python
over   = (f_right - 5.0 N).clamp(0)
r_over = -0.05 * (over**2).sum(dim=1)
```
Old threshold was 1.5 N (physically impossible for real grasping). New 5 N allows actual pinch forces (Dex3 μ≈0.5, 50g block → need ~5 N/finger for tripod grasp).

### Smoothness
```python
accel  = (cur_hand_vel - prev_hand_vel).abs().sum(dim=1)
reward -= 0.002 * accel
```

### Diagnostic logging
Every 500 reward calls, prints to stdout:
```
[reward_diag] r_pal=... r_clo=... has_grasp=... active_pads=... thumb_f=... opp_f=...
[reward_diag] r_lift_prop=... r_lift_bonus=...
```

---

## 6. Policy Architecture — System0PPOWrapper (8-expert MoE)

**File:** `experiments/system0_rl/system0_moe.py`

### Input
```
obs_with_targets (92D) + intent (128D) → cat → 220D full_input
```
The intent is one-hot curriculum stage encoding: `intent[:, stage] = 1.0`, rest zero.

### Actor: `System0MoEActor`
```
input_encoder: Linear(220, 256) → LayerNorm(256) → SiLU
moe1:          LayerNorm(256) + MoEFFN(256, 8 experts, top-2, intent_dim=128)  [residual]
moe2:          LayerNorm(256) + MoEFFN(256, 8 experts, top-2, intent_dim=128)  [residual]
mean_head:     Linear(256, 7)
log_std:       nn.Parameter(shape=(7,), init=-1.0)  → std = exp(-1) ≈ 0.37, clamped to max 0.3
```

Each expert in `MoEFFN`: `Linear(256, 512) → SiLU → Linear(512, 256)`.

Router gate: `Linear(256+128, 8)` → softmax → top-2 → renormalize.

### Critic: `System0Critic`
```
Linear(220, 256) → LayerNorm → SiLU → Linear(256, 256) → SiLU → Linear(256, 1)
```

### Total parameters: ~2.0M

### std clamping (important)
`std = log_std.exp().clamp(min=1e-6, max=0.3)`. The std is learned but bounded to max 0.3 rad. This means the policy distribution over actions is fixed-width in the worst case, which may limit exploration range.

---

## 7. PPO Training Details

**Files:** `experiments/system0_rl/ppo.py`, `config.py`, `train.py`

### Hyperparameters
| Param | Value | Notes |
|---|---|---|
| `num_envs`       | 2048 (remote) / 64 (desktop vis) | |
| `rollout_steps`  | 256 per env | 256×2048 = 524,288 steps/rollout |
| `ppo_epochs`     | 4 | |
| `minibatch_size` | 512 | |
| `lr`             | 3e-4 | Adam |
| `gamma`          | 0.99 | |
| `gae_lambda`     | 0.95 | |
| `clip_eps`       | 0.2 | |
| `value_coeff`    | 0.5 | |
| `entropy_coeff`  | 0.001 | Reduced from 0.01 after entropy trap |
| `max_grad_norm`  | 0.5 | |
| `delta_max`      | 0.5 rad | action bound |
| `ou_theta`       | 0.15 | OU noise mean-reversion |
| `ou_sigma`       | 0.10 | OU noise scale |
| `total_timesteps` | 60M (Run 11) | |

### GAE Implementation (ppo.py)
Standard GAE with a **potential bug**: `compute_advantages` uses `last_gae` across episode boundaries. When `done[t+1]=1.0`, it does `(1-next_done)=0`, correctly zeroing the bootstrap. However, the buffer stores ALL envs interleaved (one `add()` call per env per step). The done signal for `env[i]` at step `t` is used at `t+1`, but `self.dones[t+1]` contains the done for a **different env** in the interleaved buffer. This is a known issue in naive vectorized PPO implementations.

### Rollout collection flow
```
for step in range(256):
    obs = build_obs_batch(env, sim_hand, device)          # (N, 85)
    obs_with_targets = cat([obs, coarse_targets], dim=1)  # (N, 92)
    raw_delta, log_probs, values = policy.act(obs_w_t, intent)
    action = tanh(raw_delta) * 0.5 + ou_noise.sample()
    action = clamp(action, -0.5, 0.5)
    env.step(action)  # applies to ALL 29 joints, non-hand joints via stiff PD
    rewards = compute_reward_blind(env, prev_vel, cur_vel, device)
    buffer.add(obs_w_t[i], intent[i], action[i], log_prob[i], reward[i], done[i], value[i])
    # NOTE: actions go to full 29-DOF action manager but policy only outputs 7D
    # The mapping from 7D to 29D is handled by the env's JointPositionActionCfg
```

**IMPORTANT: Action dimension mismatch concern.** `env.action_manager.total_action_dim` is the full robot DOF (29+). The policy outputs 7D (right hand). The training loop applies `action` (7D) directly via `env.step(action)`. This likely works because `JointPositionActionCfg` with `joint_names=[".*"]` maps the first 7 values to the first 7 joints alphabetically, OR the env only exposes the 7 right-hand joints through its action interface. **This should be verified — if the 7D action is being applied to the wrong joints, the policy is controlling joints it cannot observe.**

---

## 8. Curriculum

| Stage | Extra XY range | DR | Advance when |
|---|---|---|---|
| 0 | ±0 cm (only env's ±5 cm) | No | lift_rate ≥ 75% over last 1000 eps |
| 1 | ±2 cm | No | same |
| 2 | ±5 cm | No | same |
| 3 | ±5 cm | mass/friction | — |

We have never advanced beyond Stage 0 in any run.

---

## 9. Training History

| Run | WandB | Steps | Peak Lift | Peak R | Notes |
|---|---|---|---|---|---|
| 1 | lh8a264n | ~3M | 0% | — | No headless flag → EGL stall |
| 2 | yrugn52y | ~5M | 0% | -14K | OVERFORCE_COEFF=0.2 → collapse |
| 3 | qg9b59c4 | ~10M | 0% | +72 | entropy_coeff=0.01 → entropy trap |
| 4 | 8jtj2m0s | ~10M | 0% | — | All-negative rewards trap |
| 5 | ugfk3rr0 | ~10M | 0.5% | +60 | First lift events |
| 6 | v0sfb5q3 | ~10M | **1.0%** | +96 | Best ever, gentle_contact reward |
| 7 | lfkzq61k | ~10M | 0% | +93 | std_max=0.1, lift regressed |
| 8 | rmx1a3i3 | ~10M | 0.5% | +98 | std_max=0.1 restored to 0.3 |
| 9 | tvyr7hld | ~10M | 0.5% | **+103.7** | New reward record, std_max=0.3 |
| 10 | jtt5jmbv | ~10M | 0.5% | +89 | Gentle contact plateau confirmed |
| 11 | TBD | running | TBD | TBD | **Phase-aware reward v2** |

All runs 5-10 loaded from same checkpoint chain (`final.pt` → next run).

---

## 10. Known Bugs Fixed

1. **Wrong block scene key** (`"red_block"` → `"block"`) — caused silent zero lift reward for 3 runs
2. **No contact baseline** — idle arm-table contact forces ~75 N caused -1066/step overforce penalty; fixed by taring sensor after 5 warmup steps
3. **OVERFORCE_COEFF=0.2** → -29/step from dynamic finger contacts; reduced to 0.001, now 0.05
4. **ContactSensorCfg filter semantics** — `filter_prim_paths_expr` has exclusion semantics; was suppressing block contacts; removed
5. **entropy_coeff=0.01** → entropy explosion to 14.6 at 1.3M steps; reduced to 0.001
6. **std_max** — run 7 used std_max=0.1 (too tight), lift dropped to 0%; reverted to 0.3
7. **EGL headless** — first run without `--headless` caused 20-min stall on server

---

## 11. Suspected Issues / Open Questions for Expert

### 11.1 The Fundamental Local Optimum
The policy consistently reaches R≈+80–100/ep by learning to hover and tap fingers lightly against the block (gentle contact reward). The lift bonus (+5–50) is sparse and requires a real pinch. The policy never forms the transition from "touching" to "grasping".

**Why this is hard:** From the policy's perspective, it sees:
- Light touch → reward comes NOW (palpation signal)
- Grasping → requires coordination of 3+ pads simultaneously with force ~5N — a hard exploration problem
- Value function may assign low value to "many pads engaged" states if it has never seen them lead to lift

### 11.2 GAE interleaving bug (ppo.py:compute_advantages)
See Section 7. The buffer is a flat list of (N×rollout_steps) transitions where env 0 step 0 is followed by env 1 step 0, etc. The `compute_advantages` loop treats this as a single trajectory, so done signals from env `i` affect the bootstrap for env `j` at the next index. This means advantage estimates are corrupted at episode boundaries across ALL envs.

**Potential fix:** Store rollout as `(rollout_steps, N)` shape and compute GAE per-env, then flatten.

### 11.3 Action dimension mismatch
In `train.py`, `env.step(action)` is called with 7D action. But `ActionsCfg.joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*"])` suggests the action covers ALL robot joints. Need to verify the env only applies the 7D to the right hand joints, or whether there's padding happening automatically.

### 11.4 Contact baseline is per-env but captured once
The baseline is captured right after env reset at startup. If the arm's idle contact forces vary between envs (e.g., block slightly under palm in some envs), the baseline will be wrong for those envs. Currently the baseline is a (N,16) tensor per-env, so this is partially handled — but it's never refreshed after the first episode.

### 11.5 Palpation reward does NOT gate on phase 3
Looking at `rewards.py`:
```python
r_palpation = PAL_COEFF * best_gentle * palpating.float()  # palpating = (1 ≤ active ≤ 2)
r_closure   = CLOSURE_COEFF * tanh(opposition / CLOSURE_SAT)  # always active
```
`r_closure` is computed for ALL envs, including those with 0 or 1 active pads. If only the palm is engaged (not thumb or fingers), both `thumb_force` and `opposing_force` will be near zero → `r_closure ≈ 0`. But if the palm presses the block against the table and there's incidental thumb contact, `r_closure` could fire unexpectedly.

### 11.6 Lift success detection vs. block initial height
```python
BLOCK_INIT_Z = 0.819  # m — nominal block top
lift_delta = (block_z - 0.819).clamp(0)
```
This is the nominal resting height. But after environment reset, the block spawns at various XY positions. Its Z may vary slightly depending on table surface variation. If `block_z` at rest is actually 0.817 in some envs, the policy needs to lift 2 cm more before earning any lift reward.

### 11.7 OU noise on top of tanh-bounded action
```python
action = tanh(raw_delta) * 0.5 + ou_noise.sample()
action = clamp(action, -0.5, 0.5)
```
OU noise is added after `tanh`, so the policy can't "compensate" for exploration noise. More importantly, the noise may cancel finger-closing actions: if policy outputs max close action (+0.5), OU noise (±0.1) sometimes pushes it back. Since std=0.3 in the policy itself, and OU adds ±0.1 more, total exploration STD ≈ √(0.3²+0.1²) ≈ 0.31. This seems fine.

### 11.8 No value function curriculum: sparse lift reward
The value function must learn to assign high value to "near-grasp" states it has almost never seen. With lift_rate ≤ 1%, fewer than 1 in 100 episodes have any lift signal. The TD bootstrap chain from lift reward back to "many pads engaged" states may be too long to credit assignment reliably.

### 11.9 std is fixed per-dimension (not state-dependent)
`log_std` is a global parameter, not a function of the observation. All finger joints get the same exploration noise. Thumb joints may need different exploration scale than index joints for effective tripod formation.

---

## 12. File Map

```
experiments/system0_rl/
  train.py          — Main training loop (PPO, curriculum, checkpointing)
  rewards.py        — Phase-aware reward v2 (phase 1-4, has_grasp gate)
  system0_moe.py    — Policy: System0PPOWrapper (8-expert MoE) + RLSystem0Policy (4-expert)
  ppo.py            — RolloutBuffer + GAE + ppo_update
  config.py         — TrainConfig hyperparameters
  env_cfg.py        — Alternative env config (NOT used by train.py currently)
  checkpoints/      — final.pt (Run 9 = best checkpoint)

experiments/system0_skills/
  block_stack_env.py   — BlockStackEnvCfg (USED by train.py)
  block_stack_config.py — BlockStackConfig (robot pos, joint indices)

tasks/common_observations/
  tactile_state.py  — get_tactile_obs() / get_tactile_obs_extended() / DEX3_PAD_LINKS
```

---

## 13. Hardware

- **Training:** Remote PC `konstantinsmirnov@10.127.102.40`, NVIDIA RTX 6000 Ada (48GB VRAM), 2048 envs, ~40K FPS
- **Visualization:** Desktop `cosmos@192.168.1.201`, RTX 5080, 64 envs, ~2K FPS (with viewport)
- **Conda env:** `unitree_sim_env` on both machines
- **IsaacLab version:** 5.1 (IsaacSim 5.1 backend)

---

## 14. Current Run (Run 11)

- PID: 42891 (remote), bash wrapper 42865
- Started: ~13:25 GMT+4, 2026-04-26
- total_timesteps: 60M
- Checkpoint: loaded from Run 9 `final.pt` (step ~40M, R=103.7)
- Reward: Phase-aware v2 (first run with new reward)
- Watch signals: `r_palpation` dropping + `r_closure` rising by 1-2M steps
- If `r_closure` stays 0 at 3M steps → CLOSURE_COEFF too low, increase to 2.0

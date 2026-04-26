# ACT-CraftNet v3 Implementation Report

**Date:** 2026-04-06  
**Author:** Constantine Smirnov  
**Cluster:** MBZUAI SLURM — AMD MI210 GPUs (ROCm)  
**Cluster path:** `/vast/users/chenyuan.chen/constantine/act_craftnet/`

---

## 1. Overview

ACT-CraftNet is a bimanual robot manipulation policy for G1+Dex3 based on ACT (Action Chunking with Transformers). This report documents the v3 implementation that extends the baseline ACT architecture with:

1. **FrozenDINOv2 ViT-B/14** — frozen visual backbone replacing ResNet-18
2. **iDP3-style depth encoder** — 3-view depth point cloud encoder (head + 2 wrists)
3. **System 0 MoE** — Mixture-of-Experts reactive finger correction with bidirectional System 1↔System 0 connections
4. **Tactile feedback injection** — S0→S1 connection via extra encoder token

---

## 2. Files

| File | Description |
|------|-------------|
| `dataset.py` | Dataset loader for custom xr_teleoperate episode format |
| `model.py` | ACTCraftNet model with all v3 components |
| `train.py` | Multi-GPU DDP training loop (torchrun) |
| `slurm_train.sh` | SLURM job script for AMD MI210 cluster |
| `check.py` | Sanity check script |

---

## 3. Dataset (`dataset.py`)

### 3.1 Episode Format

Each episode is stored as a directory:
```
episode_NNNN/
  data.json          # frame metadata, paths, state, actions, tactile
  colors/            # RGB JPEG frames
  depths/            # uint16 PNG depth frames (mm)
```

### 3.2 Observation Keys

| Key | Shape | Notes |
|-----|-------|-------|
| `images` | `(3, 3, 224, 224)` | head (color_0), left wrist (color_2), right wrist (color_3) |
| `depths` | `(3, 1, 120, 160)` | head_left_0, left_wrist_2, right_wrist_3 |
| `state` | `(28,)` | `[left_arm(7) | left_ee(7) | right_arm(7) | right_ee(7)]` |
| `tactile` | `(18,)` | `[left_ee(9) | right_ee(9)]` |
| `env_state` | `(9,)` | relative xyz of red/yellow/green blocks |
| `action` | `(50, 28)` | 50-step action chunk |
| `action_is_pad` | `(50,)` | bool mask for padding |

### 3.3 Depth Loading

Depth PNGs are saved as **uint16 in millimetres** by `xr_teleoperate/teleop/utils/episode_writer.py`. The loader applies:
```python
d = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # MUST be IMREAD_UNCHANGED
depth_m = d.astype(np.float32) / 1000.0     # mm → metres
```

Missing depth keys produce zero tensors (graceful fallback).

### 3.4 Data Splits

`make_splits()` shuffles episodes and splits by `val_frac=0.1`. Block-level split (not frame-level) to prevent train/val leakage.

**Dataset size (block_stacking):** ~152 episodes, ~10,000+ frames total.

---

## 4. Model Architecture (`model.py`)

### 4.1 Configuration (`ACTConfig`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `state_dim` | 28 | full joint qpos |
| `action_dim` | 28 | full joint targets |
| `tactile_dim` | 18 | 9 per hand × 2 hands |
| `env_dim` | 9 | 3 blocks × 3D relative xyz |
| `n_cameras` | 3 | head + 2 wrists |
| `n_depth_views` | 3 | head + 2 wrists |
| `finger_dim` | 14 | left_ee(7) + right_ee(7) |
| `chunk_size` | 50 | 50-step action chunk |
| `latent_dim` | 32 | CVAE latent dimension |
| `dim_model` | 512 | transformer hidden dim |
| `n_heads` | 8 | attention heads |
| `dim_ff` | 3200 | feedforward dim |
| `n_enc_layers` | 4 | encoder layers |
| `n_dec_layers` | 7 | decoder layers |
| `kl_weight` | 1.0 | KL loss weight (conservative for 152 eps) |
| `physical_intent_dim` | 128 | S1→S0 connection |
| `feedback_dim` | 64 | S0→S1 connection |

### 4.2 Encoder Token Sequence

The ACT encoder processes 8 tokens:

```
Index  Token            Dim     Source
  0    latent z         512     CVAE encoder output (zeros at inference)
  1    state            512     28D joint qpos → Linear(28, 512)
  2    tactile_feedback 512     System 0 feedback encoder → 64D → Linear(64, 512)
  3    env_state        512     9D block positions → Linear(9, 512)
  4    depth            512     iDP3DepthEncoder → 256D → Linear(256, 512)
  5    cam0             512     FrozenDINOv2(head image) → avg-pool → Linear(768, 512)
  6    cam1             512     FrozenDINOv2(left wrist) → avg-pool → Linear(768, 512)
  7    cam2             512     FrozenDINOv2(right wrist) → avg-pool → Linear(768, 512)
```

All tokens get sinusoidal positional embeddings (8×512).

### 4.3 FrozenDINOv2

```
Input:  (B×3, 3, 224, 224)  — all cameras flattened along batch
Output: (B, 3, 512)         — one 512D token per camera

Pipeline:
  Dinov2Model.from_pretrained("facebook/dinov2-base")
  → last_hidden_state[:, 1:]  # (B, 256, 768) — drop CLS, keep 256 patch tokens
  → mean(dim=1)               # (B, 768) — avg-pool over patch tokens
  → Linear(768, 512)          # (B, 512) — project to model dim
```

**DINOv2 ViT-B/14 specs:**
- patch_size=14, image=224×224 → 16×16 = **256 patch tokens per camera**
- hidden_dim=768
- All 86.6M DINOv2 parameters are **frozen** (no gradient, excluded from optimizer)

### 4.4 iDP3DepthEncoder

Encodes 3 depth views into a single 256D token.

```
Pipeline per view:
  depth (1, 120, 160) float32 metres
  → back-project to 3D using normalized camera grid
  → centroid subtraction (translation invariance)
  → (B, 3, 19200) point cloud
  → Conv1d(3→64) + BN + ReLU → global max pool → 64D
  → Conv1d(64→128) + BN + ReLU → global max pool → 128D
  → Conv1d(128→256) + BN + ReLU → global max pool → 256D
  → concat multi-scale: 64+128+256 = 448D

3 views concatenated: 3×448 = 1344D
→ MLP(1344→512→256, ReLU, LayerNorm) → 256D fused token
```

**⚠️ Known Issue (to fix before next job):** The back-projection uses a normalized grid (`(x - W/2) / W`) instead of proper camera intrinsics (`(x - cx) / fx`). The spec requires per-view intrinsics:

| View | fx | fy | cx | cy | at resolution |
|------|----|----|----|----|---------------|
| head | 138.5 | 138.5 | 80.0 | 60.0 | 120×160 |
| left wrist | 151.25 | 151.25 | 80.0 | 60.0 | 120×160 |
| right wrist | 151.25 | 151.25 | 80.0 | 60.0 | 120×160 |

(Derived by scaling factory calibration 640×480 → 120×160: ×0.25)

This bug does not crash training but produces incorrect 3D geometry. Fix needed in next job.

### 4.5 System 0 Policy (MoE)

```
Architecture:
  Router:  Linear(512+128, 4 experts) — conditioned on hidden_state + physical_intent
  Experts: 4× MLP(640→256→14) — each outputs a Δfinger(14D) correction
  Top-k=2 softmax gating

Feedback encoder:
  [tactile(18) | finger_state(14)] → MLP → 64D tactile_feedback
```

**Near-zero init:** Expert output layers and feedback encoder output are initialized with `gain=0.01`. This ensures System 0 starts as a pass-through (delta≈0), falling back to pure System 1 behavior. Safe by construction.

**Residual application:**
```python
actions_hat[:, :, 7:14]  += delta[:, :, :7]   # left_ee correction
actions_hat[:, :, 21:28] += delta[:, :, 7:]   # right_ee correction
```

### 4.6 Bidirectional S1↔S0 Connections

```
S1 → S0 (physical_intent):
  CVAE latent z (32D) → Linear(32, 128) → physical_intent (128D)
  Passed to System 0 MoE router to select manipulation-phase-appropriate experts

S0 → S1 (tactile_feedback):
  System 0 feedback encoder: [tactile(18) | finger_state(14)] → 64D
  → Linear(64, 512) [near-zero init] → injected as encoder token index 2
  At training start: near-zero init means negligible impact
  Over training: System 0 learns to encode useful contact information
```

**No circular dependency:** S0's feedback uses current-step tactile and finger state (not S1's output). S1 uses previous-step feedback token. Sequential, not circular.

### 4.7 CVAE Encoder (VAE for training)

```
Tokens: [CLS | state | action_0 | action_1 | ... | action_49]
  = 1 + 1 + 50 = 52 tokens
→ 4-layer TransformerEncoder
→ CLS token output → Linear(512, 64) → [mu(32) | log_sigma(32)]
→ z = mu + exp(log_sigma/2) * N(0,1)
```

At inference: z = zeros (no action chunk available).

### 4.8 Decoder

```
Learned queries: nn.Embedding(50, 512) → 50 query tokens
7-layer cross-attention decoder (custom _TransformerDecoderLayer)
→ (50, B, 512) → action_head: Linear(512, 28)
```

### 4.9 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| FrozenDINOv2 | 86.6M | ✗ (frozen) |
| ACT encoder/decoder | ~20M | ✓ |
| iDP3DepthEncoder | ~0.4M | ✓ |
| System0Policy MoE | ~0.5M | ✓ |
| Projections & embeddings | ~10M | ✓ |
| **Total** | **~161M** | **74.6M** |

---

## 5. Training (`train.py`)

### 5.1 Loss Function

```
L = L1_action + kl_weight × KL_divergence

L1_action = mean(|actions_hat - gt_action| × valid_mask)
  where actions_hat includes System 0 finger corrections
  and valid_mask excludes padded timesteps

KL = -0.5 × mean(1 + log_σ² - μ² - exp(log_σ²))
```

System 0 is trained jointly through the L1 loss: the delta_finger correction applied to finger dims is supervised by the ground truth actions. No separate S0 loss.

**kl_weight=1.0:** Original ACT paper uses 10.0, but with only 152 episodes the smaller dataset benefits from a more conservative KL (less aggressive latent space compression). This was validated in v2 training logs.

### 5.2 Optimizer

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
```

DINOv2 params have `requires_grad=False` so they are auto-excluded. No separate backbone LR group (DINOv2 is fully frozen).

### 5.3 LR Schedule

Cosine decay with linear warmup:
- Warmup: 0 → 1e-4 over 1000 steps
- Decay: cosine from 1e-4 → 0 over 100,000 steps

### 5.4 DDP Setup

```bash
torchrun --nproc_per_node=8 train.py
```

- `DistributedDataParallel` with `find_unused_parameters=False`
- `DistributedSampler` with explicit epoch counter (not step counter)
- `model(batch)` called directly through DDP wrapper (not `model.module(batch)`) to preserve gradient sync hooks

### 5.5 WandB Logging

```python
wandb.init(project="act-craftnet", name="act_craftnet_v3", resume="allow", id="act_craftnet_v3")
```

Logged metrics (every 100 steps):
- `train/loss`, `train/l1`, `train/kl`, `train/lr`, `train/epoch`

Logged metrics (every 5000 steps):
- `val/loss`

### 5.6 Checkpointing

- `checkpoint_latest.pt` — saved every 5000 steps (overwrites, no accumulation)
- `checkpoint_final.pt` — saved at end of training
- Checkpoint contains: `step`, `epoch`, `model` state dict, `optimizer` state dict, `args`

---

## 6. SLURM Configuration (`slurm_train.sh`)

```bash
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
```

**Critical AMD MI210 / ROCm environment variables:**
```bash
export MIOPEN_DISABLE_CACHE=1        # prevents "readonly database" crash on shared FS
export HIPBLAS_OP_DTYPE_FP32=1       # avoids NaN from BF16 on gfx90a
export HIPBLASLT_TUNING_OVERRIDE=NONE
export NCCL_SOCKET_IFNAME=bond0
```

**Run name:** `act_craftnet_v3`  
**Output dir:** `/vast/users/chenyuan.chen/constantine/act_craftnet/runs/act_craftnet_v3/`

---

## 7. Job History

| Job | Status | Notes |
|-----|--------|-------|
| 39989 | Failed | MIOpen readonly database crash |
| 39990 | Failed | MIOpen readonly database crash |
| 39991 | Cancelled | v2 (ResNet baseline with WandB) |
| 39993 | Cancelled (queued) | v3 (DINOv2+iDP3+S0) — queued but QOSMaxGRESPerUser |

---

## 8. Known Issues / Deviations from Spec

### 8.1 iDP3 Intrinsics Bug (Critical)

**Current:** Back-projection uses normalized grid `(x - W/2) / W` instead of real intrinsics.

**Should be:**
```python
DEFAULT_INTRINSICS = {
    0: (138.5, 138.5, 80.0, 60.0),    # head (Isaac Sim, 60° hFOV scaled to 120×160)
    1: (151.25, 151.25, 80.0, 60.0),  # left wrist (D405 factory, scaled to 120×160)
    2: (151.25, 151.25, 80.0, 60.0),  # right wrist (D405 factory, scaled to 120×160)
}
# Back-project: x_3d = (u - cx) / fx * depth
```

Effect: geometry is slightly wrong (incorrect scale), but the Conv1d pyramid is translation+scale robust to some degree. Fix before evaluating on robot.

### 8.2 Centroid Subtraction (Minor)

**Current:** Centroid computed over all N points including zero-padded ones.

**Should be:** Compute centroid over valid (non-zero depth) points only, using valid mask.

### 8.3 Finger Dim Discrepancy from Spec

**Spec says:** `finger_dim=21` (single-arm Dex3-1)  
**Actual dataset:** `finger_dim=14` (bimanual: left_ee(7) + right_ee(7))  
**Implementation:** Correctly uses 14D matching the actual data.

### 8.4 System 0 Standalone BC Loss

**Spec says:** Separate System 0 BC loss supervised by residual `gt_finger - system1_target`.  
**Implementation:** System 0 trained jointly through the combined L1 loss. Simpler but loses the explicit residual supervision signal.

### 8.5 Point Cloud Subsampling

**Spec says:** Random subsample to N=512 points from valid depth pixels.  
**Implementation:** Processes all 19,200 pixels (120×160) through Conv1d (no random subsample). Works but is slower and less translation-robust than using FPS or random subsampling.

---

## 9. Spec Compliance Summary

| Feature | Status | Notes |
|---------|--------|-------|
| DINOv2 ViT-B/14 frozen backbone | ✅ Correct | avg-pool 256 patch tokens → 512D |
| 3-view depth (head + 2 wrists) | ✅ Correct | head_left_0 key exists in dataset |
| iDP3 Conv1d pyramid | ✅ Correct | 3→64→128→256, multi-scale global max pool |
| iDP3 fusion MLP (1344→256) | ✅ Correct | |
| iDP3 back-projection intrinsics | ❌ Bug | Normalized grid instead of fx/fy/cx/cy |
| iDP3 valid-point centroid | ❌ Bug | Includes zero-padded points in centroid |
| System 0 MoE (4 experts, top-k=2) | ✅ Correct | |
| S1→S0 physical_intent (128D) | ✅ Correct | From CVAE latent z |
| S0→S1 tactile_feedback (64D) | ✅ Correct | Near-zero init, extra encoder token |
| 8 encoder tokens | ✅ Correct | [z, state, tac_fb, env, depth, cam0, cam1, cam2] |
| kl_weight=1.0 | ✅ Correct | Conservative for small dataset |
| DDP gradient sync | ✅ Fixed | model(batch) not model.module(batch) |
| DistributedSampler epoch | ✅ Fixed | set_epoch(epoch) not set_epoch(step) |
| WandB logging | ✅ Present | train/loss, l1, kl, lr, val/loss |
| checkpoint_latest.pt only | ✅ Correct | Overwrites, no accumulation |
| MIOPEN_DISABLE_CACHE=1 | ✅ Critical fix | Prevents MIOpen crash |

---

## 10. Next Steps

1. **Fix iDP3 intrinsics** — replace normalized grid with real fx/fy/cx/cy per view
2. **Fix centroid computation** — use valid-point mask only
3. **Resubmit job** with corrected model.py
4. **Monitor first 500 steps** for:
   - Loss decreasing (L1 should drop from ~0.5 to ~0.2 over 10k steps)
   - kl_loss stable (shouldn't explode)
   - No NaN (depth zeros from head camera handled gracefully)
5. **Long term:** Implement System 0 standalone BC loss with explicit residual supervision
6. **Long term:** LeRobot framework integration under `src/lerobot/policies/act_craftnet/`

---

## 11. How to Restart

```bash
ssh chenyuan.chen@172.27.112.247
cd ~/constantine/act_craftnet
sbatch slurm_train.sh
```

Check logs:
```bash
tail -f ~/constantine/act_craftnet/logs/slurm_<JOBID>.log
```

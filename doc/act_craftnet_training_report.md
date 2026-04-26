# ACT-CraftNet Training Report
**Date:** 2026-04-05  
**Author:** Constantine Smirnov  
**Cluster:** MBZUAI HPC — `chenyuan.chen@172.27.112.247`  
**Job:** SLURM 39986 (running, 8× AMD MI210 GPUs)  

---

## 1. Problem Statement

Train a robot manipulation policy for **bimanual block stacking** using the G1+Dex3 robot.
The policy takes observations from the robot (joint states, tactile sensors, object positions, 3 cameras,
2 depth cameras) and outputs an **action chunk** — the next 50 joint position targets simultaneously.

The dataset was collected via VR teleoperation (`xr_teleoperate`) and stored in a custom episode format
on the MBZUAI cluster: 152 episodes (~42 GB total).

---

## 2. Dataset

### 2.1 Storage Format

Each episode is a directory:
```
block_stacking/
  episode_0001/
    data.json        ← frame list with paths and metadata
    colors/
      000001_color_0.jpg   ← head camera (640×480)
      000001_color_2.jpg   ← left wrist camera
      000001_color_3.jpg   ← right wrist camera
    depths/
      000001_left_wrist_2.png    ← uint16 PNG, millimeters
      000001_right_wrist_3.png
```

`data.json` structure:
```json
{
  "info": {"image": {"fps": 30, "width": 640, "height": 480}, ...},
  "data": [
    {
      "idx": 0,
      "colors":  {"color_0": "colors/000001_color_0.jpg", ...},
      "depths":  {"left_wrist_2": "depths/000001_left_wrist_2.png", ...},
      "states":  {"left_arm": {"qpos": [...]}, "left_ee": {"qpos": [...]}, ...},
      "actions": {"left_arm": {"qpos": [...]}, ...},
      "tactiles":{"left_ee": [...9 values...], "right_ee": [...9 values...]},
      "sim_state": {"object_positions": {"red_block": {"rel": [x,y,z]}, ...}}
    },
    ...
  ]
}
```

### 2.2 Sensor Modalities

| Modality | Key | Dimension | Notes |
|---|---|---|---|
| Joint state | left_arm + left_ee + right_arm + right_ee | **28D** | 7D each |
| Target action | same keys | **28D** | Joint position targets |
| Tactile | left_ee + right_ee | **18D** | 9D per hand — one value per pressure module; Dex3-1 has 33 sensing elements/hand but the topic publishes 9 aggregated module readings |
| Env state | red/yellow/green block relative xyz | **9D** | relative to robot base |
| Head camera | color_0 | 3×224×224 | ResNet-18, ImageNet-norm |
| Left wrist camera | color_2 | 3×224×224 | ResNet-18, ImageNet-norm |
| Right wrist camera | color_3 | 3×224×224 | ResNet-18, ImageNet-norm |
| Left wrist depth | left_wrist_2 | stored 480×640, loaded 1×120×160 | uint16 PNG → float32 metres; 4× downsampled at load time |
| Right wrist depth | right_wrist_3 | stored 480×640, loaded 1×120×160 | uint16 PNG → float32 metres; 4× downsampled at load time |

### 2.3 Depth PNG Format — Critical Detail

Depth images are saved by `episode_writer.py` as **uint16 PNG in millimetres**. The writer halves
the input resolution (`depth.shape // 2`), so the stored size depends on the capture resolution.
Verified on disk: all depth PNGs (head, wrist) are **480×640 (H×W)** — meaning the raw depth
stream was captured at 960×1280 before halving.

```python
depth_half = cv2.resize(depth, (depth.shape[1]//2, depth.shape[0]//2), interpolation=cv2.INTER_NEAREST)
depth_mm = (depth_half * 1000.0).clip(0, 65535).astype(np.uint16)
cv2.imwrite(path, depth_mm)
```

`dataset.py` further downsamples to 120×160 at load time (4× reduction from stored size):
```python
d = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # MUST use IMREAD_UNCHANGED — uint16
depth_m = d.astype(np.float32) / 1000.0     # mm → metres
d = cv2.resize(d, (160, 120))               # → (120, 160) in H×W tensor
```

Recovery at load time **must** use:
```python
d = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # MUST use IMREAD_UNCHANGED for uint16
depth_m = d.astype(np.float32) / 1000.0     # mm → metres
```

Using plain `cv2.imread()` silently reads only 8 bits, returning garbage.

### 2.4 Dataset Statistics

- **Total episodes:** 152 (after transfer from local PC to cluster)
- **Train split:** 137 episodes (~216,000 samples at chunk=50, skip=1)
- **Val split:** 15 episodes
- **Effective batch:** 8 per GPU × 8 GPUs = **64 samples/step**
- **Total steps:** 100,000 (estimated ~46 hours on 8× MI210)

---

## 3. Model Architecture: ACT-CraftNet

### 3.1 Overview

ACT-CraftNet is based on **ACT (Action Chunking with Transformers)** from the original paper by Zhao et al. (2023), extended to handle:
- Bimanual 28D action space (instead of single-arm)
- 3 cameras with a shared ResNet-18 backbone
- Additional modalities: tactile (18D), env state (9D), depth cameras (not yet used in the main transformer — loaded but not injected into tokens)
- CVAE latent conditioning

**Total parameters:** ~83.9M

### 3.2 Component Breakdown

#### Visual Encoder — CameraEncoder

```
Input:  (B, 3, 224, 224) — one camera image, ImageNet-normalised
Backbone: ResNet-18 (pretrained ImageNet) with final avgpool+fc stripped
          → output spatial feature map (B, 512, 7, 7)
AdaptiveAvgPool2d(1,1) → (B, 512, 1, 1)
Flatten → (B, 512)
```

**Key decision:** One `CameraEncoder` instance with **shared weights** across all 3 cameras. This was
chosen over 3 separate encoders because 152 episodes provides insufficient data to train 3×11M visual
parameters independently. The single backbone is called 3 times per forward pass.

A `cam_proj` Linear(512 → 512) maps each camera's pooled feature to `dim_model`.

#### CVAE Encoder (training only)

The CVAE encoder sees the **current state + future action chunk** to learn a latent distribution.
At inference, the latent `z` is set to zeros (mean of the prior).

```
Input tokens:
  [CLS embedding (1×D)]
  [state projection (1×D)]        ← state_dim=28 → D
  [action projections (chunk×D)]  ← 50 frames of 28D action → 50×D

Total sequence: (2 + chunk_size) = 52 tokens

Sinusoidal positional encoding added.

Transformer encoder: 4 layers (standard PyTorch TransformerEncoderLayer, pre-norm=False)
  d_model=512, heads=8, dim_ff=3200, dropout=0.1

CLS token output → Linear(512, 64) → split into:
  μ (32D), log_σ (32D)
  z = μ + exp(log_σ/2) * ε,  ε ~ N(0,1)
```

#### Main Transformer Encoder

Processes 7 tokens representing the full observation at time t:

```
Token 0: latent_proj(z)          ← 32D → 512D
Token 1: state_proj(state)       ← 28D → 512D
Token 2: tactile_proj(tactile)   ← 18D → 512D
Token 3: env_proj(env_state)     ← 9D  → 512D
Token 4: cam_proj(camera_0_feat) ← 512D → 512D (head camera)
Token 5: cam_proj(camera_2_feat) ← 512D → 512D (left wrist)
Token 6: cam_proj(camera_3_feat) ← 512D → 512D (right wrist)

Sinusoidal positional encoding added (shape: 7 × 512).

Transformer encoder: 4 layers
  d_model=512, heads=8, dim_ff=3200, dropout=0.1

Output: memory = (7, B, 512)
```

#### Main Transformer Decoder

Predicts the action chunk autoregressively in parallel (50 steps simultaneously):

```
50 learned query embeddings (nn.Embedding(50, 512))
  → (chunk_size, B, 512)

7 decoder layers (custom _TransformerDecoderLayer):
  Self-attention over queries
  Cross-attention over encoder memory (7 tokens)
  Feed-forward (512 → 3200 → 512)
  LayerNorm at each sub-layer

LayerNorm on output
Permute: (chunk, B, 512) → (B, chunk, 512)

Action head: Linear(512, 28) → (B, 50, 28)
```

### 3.3 Architecture Diagram

```
Observation at time t:
  state (28D) ──────────────────────────────────────────┐
  tactile (18D) ─────────────────────────────────────── │
  env_state (9D) ────────────────────────────────────── │
  image_head (3×224×224) → ResNet-18 → (512) ────────── │  7 tokens
  image_left_wrist (3×224×224) → ResNet-18 → (512) ──── │  ──────────
  image_right_wrist (3×224×224) → ResNet-18 → (512) ─── │  Encoder
  z (32D, from CVAE or zeros at inference) ──────────── │  (4 layers)
                                                         ▼
                                                   memory (7×512)
                                                         │
                                                         │ cross-attn
                                                         ▼
  50 learned queries (512D each) ──────→ Decoder (7 layers) ──→ Linear(512,28)
                                                         │
                                                         ▼
                                             action_chunk (50×28)
```

### 3.4 Loss Function

```
L = L1_action + kl_weight × KL

L1_action = mean(|action_hat - action_gt|) over valid (non-padded) frames
KL = -0.5 × sum(1 + log_σ - μ² - exp(log_σ))  (per sample, averaged over batch)
kl_weight = 1.0  (conservative; original ACT paper uses 10.0)
```

Padding mask: action chunks that extend beyond the episode end are marked as `action_is_pad=True`
and excluded from the L1 loss.

---

## 4. Training Infrastructure

### 4.1 Hardware & Software

| Item | Value |
|---|---|
| GPUs | 8× AMD Radeon Instinct MI210 (64GB HBM2e each) |
| Cluster | MBZUAI HPC, SLURM, faculty partition, gtqos QOS |
| Framework | PyTorch 2.5.1 + ROCm 6.2 |
| Conda env | `unifolm_vla` |
| DDP backend | NCCL |
| SLURM job | 39986, node auh7-1b-gpu-207 |

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Batch per GPU | 8 | 64 effective (8 GPUs) |
| Total steps | 100,000 | ~46h estimated |
| Learning rate | 1e-4 | Standard for ACT |
| Backbone LR | 1e-5 | 10× lower: fine-tunes ResNet gently |
| LR schedule | Linear warmup (1000 steps) + cosine decay | |
| Weight decay | 1e-4 | AdamW |
| Gradient clip | 1.0 | Prevents explosion during KL warmup |
| KL weight | 1.0 | Conservative — original ACT paper uses 10.0, but with 152 episodes the KL term can dominate before L1 stabilises |
| Chunk size | 50 | 1.67s at 30fps; covers grasp-to-place |
| Latent dim | 32 | Standard for 28D action space |
| Warmup steps | 1000 | ~15 epochs at 64 effective batch |
| Save every | 5000 steps | |
| Workers | 4 per GPU | |
| Val fraction | 0.1 | 15 / 152 episodes |

### 4.3 ROCm Environment Variables

Required for stable multi-GPU training on AMD MI210:

```bash
NCCL_SOCKET_IFNAME=bond0        # Use correct network interface for NCCL
NCCL_BLOCKING_WAIT=1            # Synchronous NCCL ops
NCCL_ASYNC_ERROR_HANDLING=1     # Surface errors immediately
NCCL_TIMEOUT=3600000            # 1h timeout for large syncs
HIPBLASLT_TUNING_OVERRIDE=NONE  # Disable hipBLASLt auto-tuning (can OOM)
HIP_FORCE_DEV_KERNELS=1         # Use device-side kernels
ROCBLAS_LAYER=0                 # Disable rocBLAS logging overhead
HIPBLAS_OP_DTYPE_FP32=1         # Force FP32 for hipBLAS ops (BF16 NaN bug on gfx90a)
```

### 4.4 SLURM Configuration

```bash
#SBATCH --partition=faculty     # correct partition (not "gpu" — doesn't exist)
#SBATCH --qos=gtqos             # Constantine's QOS: allows up to 12 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 task launches torchrun internally
#SBATCH --gres=gpu:8            # 8 GPUs on single node
#SBATCH --cpus-per-task=64      # 8 workers × 8 GPUs
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00       # 72 hours max
```

**torchrun** is used (not `mpirun`) — it manages LOCAL_RANK and is the standard PyTorch DDP launcher:
```bash
torchrun --nproc_per_node=8 --nnodes=1 train.py [args...]
```

### 4.5 Weights & Biases Logging

WandB (v0.25.1) is installed in `unifolm_vla` and integrated into `train.py`. Logging runs only
on rank 0 (`is_main()`). The running job 39986 does **not** have WandB (added after submission);
the next job submission will log automatically.

**Metrics logged every 100 steps:**
```
train/loss   — total loss (L1 + kl_weight × KL)
train/l1     — action L1 loss only
train/kl     — KL divergence term only
train/lr     — current learning rate (main params)
train/epoch  — epoch number
val/loss     — validation loss (logged every 5000 steps at checkpoint)
```

**Project:** `act-craftnet` on WandB  
**Run name:** `act_craftnet_v2`  
**Init:** `wandb.init(project="act-craftnet", name="act_craftnet_v2", resume="allow", id="act_craftnet_v2")`

`--wandb_off` flag disables WandB for offline/debug runs.

To watch training:
```bash
# Or just visit wandb.ai → act-craftnet → act_craftnet_v2
```

---

## 5. Bugs Found and Fixed

### Bug 1 — DDP Gradient Sync Bypass [HIGH — Silent, Training-Breaking]

**Location:** `train.py`, forward call in training loop  
**Version:** v1 (job 39985, cancelled)

**Broken code:**
```python
net = model.module if hasattr(model, "module") else model
actions_hat, mu, log_sigma = net(batch)   # ← bypasses DDP wrapper!
```

**Root cause:** PyTorch DDP wraps the model so that gradient synchronization (all-reduce) happens
automatically after `backward()`. This sync is triggered via hooks registered on the **DDP wrapper's
`forward()` method**. Calling `model.module.forward()` directly (i.e., the unwrapped module)
completely bypasses these hooks — so each GPU computes its own gradients and updates its own weights
independently. The 8 GPUs diverge immediately. No error is thrown.

**Fix:**
```python
actions_hat, mu, log_sigma = model(batch)   # ← always call DDP wrapper
```
The `unwrap()` helper is now only used for `state_dict()` access when saving checkpoints:
```python
def unwrap(model):
    return model.module if isinstance(model, DDP) else model
```

---

### Bug 2 — DistributedSampler Epoch Set to Step Count [HIGH — Data Distribution]

**Location:** `train.py`, training loop iterator reset  
**Version:** v1 (job 39985, cancelled)

**Broken code:**
```python
except StopIteration:
    if train_sampler is not None:
        train_sampler.set_epoch(step)   # ← step is global step, not epoch count
```

**Root cause:** `DistributedSampler.set_epoch(epoch)` uses the epoch number to seed the shuffle.
Each GPU must call it with the **same value** before creating a new iterator, so all GPUs see
the same permutation of the dataset. Using the step count is technically still a changing integer
(no deadlock), but it breaks reproducibility and can cause subtle data distribution issues.
The deeper bug was that `set_epoch` was only called on `StopIteration`, not systematically.
With 8 GPUs and ~216k samples, the loader exhausts at different times depending on timing.

**Fix:** Explicit epoch counter incremented on each `StopIteration`:
```python
epoch = 0
loader_iter = iter(train_loader)

for step in range(start_step, args.steps):
    try:
        batch = next(loader_iter)
    except StopIteration:
        epoch += 1                                   # explicit epoch counter
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)           # same value on all GPUs
        loader_iter = iter(train_loader)
        batch = next(loader_iter)
```

---

### Bug 3 — KL Weight Conservative Reduction [MEDIUM — Training Stability]

**Location:** `model.py` ACTConfig default, `compute_loss()`, `slurm_train.sh`  
**Version:** v1 default changed for v2

**Original code:**
```python
class ACTConfig:
    kl_weight: float = 10.0   # original ACT paper default (Zhao et al. 2023)
```

**Context:** `kl_weight=10.0` is the **original ACT paper default** (and the LeRobot implementation
default). It is not wrong in general. However, with only 152 episodes, the KL term can dominate
early optimisation before the L1 loss has had a chance to stabilise. The posterior
`q(z|state,action)` may be pushed toward the prior `N(0,I)` before the CVAE has learned to encode
useful action information, which can slow convergence on small datasets.

The mechanistic risk: at high `kl_weight`, `μ→0, σ→1` early in training, collapsing the latent
space to the prior. Whether this actually occurs is empirical and dataset-dependent. On large
datasets (1000+ episodes) `kl_weight=10.0` works correctly.

**Change:** Reduced to `kl_weight=1.0` as a conservative choice for the 152-episode setting.
This is not a "bug fix" in the classic sense — both values are defensible. If training shows
low KL throughout (kl < 0.1), the latent space is underused and `kl_weight` should be reduced
further or KL annealing applied.

```bash
torchrun ... train.py --kl_weight=1.0
```

---

### Bug 4 — Positional Encoding Buffers Re-transferred Every Forward Pass [MEDIUM — Performance]

**Location:** `model.py`, ACTCraftNet forward methods  
**Version:** v2 (job 39986, running) — **patched without restart**

**Broken code:**
```python
# In __init__:
self.register_buffer("vae_pos_enc",
    _sinusoidal_embed(n_vae_tokens, D, torch.device("cpu")).squeeze(0))

# In forward:
pos = self.vae_pos_enc.unsqueeze(0).to(device)   # ← creates new tensor on device every call
```

**Root cause:** PyTorch `register_buffer()` registers the tensor as a module buffer, which
automatically moves to the target device when `model.to(device)` is called. However, the buffer
was initialized explicitly on CPU (`torch.device("cpu")`). The `.to(device)` call in `forward()`
then creates a **new device tensor every forward pass** — once per training step, for both
`vae_pos_enc` and `enc_pos_enc`. This is a memory allocation and copy on every step.

**Fix (applied to model.py on cluster, takes effect on next job restart):**
```python
# In forward:
pos = self.vae_pos_enc.unsqueeze(0)   # buffer already on device via model.to(device)
pos = self.enc_pos_enc.unsqueeze(1)   # same
```
The `register_buffer` initialization with `torch.device("cpu")` is left as-is since `_sinusoidal_embed`
requires a device argument. The buffer moves to GPU automatically when `model.to(device)` is called.

---

### Bug 5 — SLURM Partition Invalid [DEPLOYMENT — Immediate]

**Symptom:** `sbatch: error: invalid partition specified`  
**Broken:** `#SBATCH --partition=gpu`  
**Fix:** `#SBATCH --partition=faculty` (discovered via `sinfo -s`)

---

### Bug 6 — SLURM QOS Invalid [DEPLOYMENT — Immediate]

**Symptom:** `sbatch: error: Invalid qos specification`  
**Broken:** No `--qos` specified (default QOS doesn't have GPU access for this user)  
**Fix:** `#SBATCH --qos=gtqos` (Constantine's QOS, allows up to 12 GPUs on faculty partition)

---

## 6. File Layout on Cluster

```
/vast/users/chenyuan.chen/constantine/act_craftnet/
  model.py           ← ACTCraftNet + ACTConfig + compute_loss
  dataset.py         ← BlockStackingDataset + EpisodeIndex + make_splits
  train.py           ← DDP training loop (v2, all bugs fixed)
  check.py           ← CPU sanity check (dataset load + forward pass)
  slurm_train.sh     ← SLURM submission script
  logs/
    slurm_39986.log  ← active training log
  runs/
    act_craftnet_v2/
      checkpoint_005000.pt  ← (first checkpoint, at step 5000)
      ...
      checkpoint_final.pt   ← (at step 100000)

/vast/users/chenyuan.chen/constantine/block_stacking/
  episode_0001/ ... episode_0152/
```

---

## 7. Training Progress

| Job | Status | Notes |
|---|---|---|
| 39985 | Cancelled | v1 code — DDP bypass + epoch bug + kl_weight=10.0 |
| 39986 | **Running** | v2 code — all bugs fixed |

**Step 500 metrics (job 39986):**
```
loss=7.61   l1=6.55   kl=1.06
```
The KL loss of ~1.06 at step 500 is healthy — it is increasing gradually as the CVAE learns
to use the latent space (not collapsing to zero, which would indicate posterior collapse).

**Monitor command:**
```bash
ssh chenyuan.chen@172.27.112.247 "tail -f ~/constantine/act_craftnet/logs/slurm_39986.log"
```

**Expected checkpoint schedule:**
- Step 5,000 → first checkpoint (loss should be ~3-5)
- Step 20,000 → L1 should be below 1.0 if learning is stable
- Step 100,000 → final checkpoint

---

## 8. Evaluation Plan (Post-Training)

Once training completes, load a checkpoint and run the policy in the IsaacLab simulation:

```python
from model import ACTCraftNet, ACTConfig
import torch

cfg = ACTConfig(kl_weight=1.0)
model = ACTCraftNet(cfg)
ckpt = torch.load("runs/act_craftnet_v2/checkpoint_100000.pt", weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

# Inference
with torch.no_grad():
    action_chunk = model.predict(batch)  # (1, 50, 28)
    # Execute first action or use temporal ensemble
```

**Temporal ensembling** (recommended): run the policy at every timestep and average overlapping
action predictions with exponential weights — reduces jitter from the chunk boundary.

---

## 9. Known Limitations

1. **ResNet-18 instead of DINOv2 ViT-B/14.** The original spec called for a frozen DINOv2 ViT-B/14
   backbone producing 256 patch tokens per camera (784 tokens total for 3 cameras), with an iDP3
   depth encoder. The running implementation uses a shared ResNet-18 with global average pooling
   producing 1 token per camera (3 tokens total). This is a deliberate baseline simplification —
   DINOv2 ViT-B/14 gives 256 patch tokens per camera (224÷14=16, 16²=256) → 768 tokens total
   across 3 cameras, which is expensive to train on 152 episodes. The key tradeoff: ResNet-18 +
   global pooling discards all spatial information, which may hurt precise block position estimation
   at varying poses. If the policy fails to reliably grasp blocks at varying positions, upgrading to
   patch-level features is the primary architectural lever.

2. **Depth not injected into the transformer.** The depth images are loaded by `dataset.py` and
   returned in each batch (`depths: (2, 1, 120, 160)`), but `model.py` does not include depth tokens
   in the encoder. Adding depth: load with a small CNN (e.g. 3-layer conv → flatten → Linear → D),
   add as token 7 and 8, increase `n_enc_tokens` to 9.

3. **`cam_proj` Linear(512→512) is redundant as shared.** All 3 cameras share the same `cam_proj`
   weights. A linear map with no nonlinearity applied identically to all cameras adds no
   camera-specific adaptation. Options: (a) make it 3 separate projections so each camera learns
   its own linear adapter to the token space, or (b) remove it and use the backbone's 512D output
   directly since `dim_model=512`. Current behaviour is functionally harmless but wastes parameters.

4. **No camera-ID positional embedding.** The encoder distinguishes cameras only via sinusoidal
   position (slots 4, 5, 6). Explicit camera-ID embeddings (one per camera, added before the
   backbone projection) would give the model a stronger prior about which view each token represents.

5. **Tactile: 18D (9 per hand) not 33D.** The Dex3-1 hardware has 33 sensing elements per hand.
   The teleoperation stack publishes 9 aggregated module readings (one per pressure pad module).
   The model trains on the 18D aggregated signal. This is consistent with the published topic —
   just ensure the thesis description distinguishes "33 sensing elements" (hardware) from "9 module
   readings" (software interface used in training).

6. **No data augmentation currently active.** `augment=True` is passed to the dataset but no
   augmentation transforms are implemented (random crop, colour jitter, etc.). Important for
   generalisation with only 152 episodes.

7. **No System 0 MoE integration.** The spec included System 0 specialists (grasp/release MoE)
   and System 1↔S0 bidirectional connections. The current implementation is a standalone ACT
   baseline with no hierarchical structure — a deliberate first step.

# VLA Policy Evaluation in IsaacLab Simulation: Best Practices

## 1. Existing Eval Infrastructure (Do Not Reinvent)

### Eval Client: `eval_g1_sim.py`
- **Location**: `~/unitree_IL_lerobot/unitree_lerobot/eval_robot/eval_g1_sim.py`
- **Purpose**: Universal eval client for both Dex1 and Dex3 sim evaluation. Handles observation collection (images + joint states), policy inference (local or remote), and action execution via DDS.
- **Supports**: Local policy (`--policy.path=...`), remote single-process policy (`--remote_policy_host`), and split System1/System2 architecture.

### Policy Servers
| File | Role | Runs Where | Hz |
|------|------|------------|-----|
| `policy_server.py` | Single-process policy server (both VLM + DiT) | Any GPU machine | ~5-10 Hz |
| `system1_server.py` | DiT action head only (lightweight) | Desktop (same as sim) | ~30 Hz |
| `system2_server.py` | Qwen3-VL backbone only (heavy) | Remote PC (RTX 6000 Ada) | ~1 Hz |

### System 0 Eval (RL Specialists)
- **Location**: `~/unitree_sim_isaaclab/experiments/system0_skills/eval_stacking.py`
- **Purpose**: Evaluates RL-trained specialist policies (grasp + release) directly in IsaacLab without DDS. Uses `ParameterizedArmTrajectory` for arm control, specialist networks for finger control.
- **Not** for VLA eval -- this is for the low-level System 0 finger specialists.

### Isaac-GR00T Eval (NVIDIA Reference)
- `~/Isaac-GR00T/gr00t/eval/open_loop_eval.py` -- Open-loop action prediction vs. ground truth dataset replay. Useful for measuring action prediction MSE but does NOT run closed-loop sim.
- `~/Isaac-GR00T/gr00t/eval/sim/` -- Wrappers for SimplerEnv, LIBERO, RoboCasa, GR00T-WholeBodyControl. These are NOT for the Unitree G1 Dex3 task.

### Launch Scripts (Ready to Use)
| Script | Purpose |
|--------|---------|
| `~/run_sim_dex3.sh` | Terminal 1: IsaacLab sim for Dex3 block stacking |
| `~/run_compositor.sh` | Terminal 2: Image compositor (camera streams -> eval client) |
| `~/run_eval_dex3_remote.sh` | Terminal 3: Eval client connecting to remote policy server |
| `~/run_sim_dex1.sh` | Terminal 1: IsaacLab sim for Dex1 (different task) |
| `~/run_eval_dex1sim.sh` | Terminal 3: Eval client with local Dex1 policy |

---

## 2. How to Set Up Dex3 Sim Eval

### 2.1. Three-Terminal Architecture (Single Policy Server)

```
Terminal 1: IsaacLab Simulation      (unitree_sim_env conda env)
Terminal 2: Image Compositor          (unitree_sim_env conda env)
Terminal 3: Eval Client               (unitree_lerobot conda env)
+ Policy Server (local or remote)     (unitree_lerobot conda env)
```

### 2.2. Four-Terminal Split Architecture (CraftNet)

```
Remote PC:   System 2 Server (Qwen3-VL ~1Hz)    unitree_lerobot env
Desktop T1:  IsaacLab Simulation                  unitree_sim_env env
Desktop T2:  Image Compositor                     unitree_sim_env env
Desktop T3:  System 1 Server (DiT ~30Hz)          unitree_lerobot env
Desktop T4:  Eval Client                          unitree_lerobot env
```

### 2.3. Required Environment Variables (All Terminals)

```bash
export UNITREE_DDS_IFACE=lo
export UNITREE_DDS_DOMAIN_ID=1
export CYCLONEDDS_URI=file:///home/cosmos/cyclonedds_local.xml
```

### 2.4. Sim Launch Command (Dex3)

```bash
cd ~/unitree_sim_isaaclab
conda activate unitree_sim_env
python sim_main.py \
  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint \
  --enable_dex3_dds \
  --robot_type g129 \
  --enable_cameras \
  --camera_include "front_camera_left,front_camera_right,left_wrist_camera,right_wrist_camera" \
  --device cpu
```

### 2.5. Eval Client Command (Remote Policy)

```bash
cd ~/unitree_IL_lerobot
conda activate unitree_lerobot
echo "s" | python unitree_lerobot/eval_robot/eval_g1_sim.py \
  --repo_id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
  --ee dex3 \
  --sim true \
  --remote_policy_host <HOST> --remote_policy_port 5556 \
  --task_override "'stack three block'"
```

---

## 3. Camera Configuration Checklist

### Sim-Side Cameras (in env cfg)

The Dex3 environment (`stack_rgyblock_g1_29dof_dex3_joint_env_cfg.py`) defines:
- `front_camera_left` -- binocular left eye (D435, +25mm Y offset, 50mm baseline)
- `front_camera_right` -- binocular right eye (-25mm Y offset)
- `left_wrist_camera` -- left Dex3 wrist cam (specific mount offset)
- `right_wrist_camera` -- right Dex3 wrist cam

All cameras: 480x640, focal_length=12.0 (wrist) / 7.6 (head), update_period=0.02s (50Hz).

### Camera Config Source: `~/unitree_sim_isaaclab/tasks/common_config/camera_configs.py`

Key camera presets:
- **Dex3 wrist cams** use `left_dex3_wrist_camera()` / `right_dex3_wrist_camera()` -- different mount point and offsets from Dex1 gripper wrist cams.
- **Head cameras** are binocular (left+right) for Dex3, monocular for some other tasks.

### Common Pitfall: Dex3 vs. Dex1 Wrist Camera Prim Paths

| Hand | Camera Prim Path |
|------|-----------------|
| Dex3 left | `/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera` |
| Dex3 right | `/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera` |
| Dex1 left | `/World/envs/env_.*/Robot/left_hand_base_link/left_wrist_camera` |
| Dex1 right | `/World/envs/env_.*/Robot/right_hand_base_link/right_wrist_camera` |

Using the wrong prim path produces blank or misplaced images.

### Image Pipeline

1. IsaacLab renders camera frames at `update_period` rate
2. `camera_state.py` publishes frames via ZMQ (ports 55555/55556/55557)
3. `image_compositor.py` composites left+right head cams into stereo and forwards to port 5555
4. Eval client's `ImageClient` receives from port 5555

**Depth**: Add `--enable_depth` to sim_main.py. Camera data_types get patched to include `distance_to_camera`. Depth must be grayscale-replicated for policy input (NOT turbo colormap).

---

## 4. State/Action Mapping Checklist

### Observation State Dimensions

The policy `observation.state` is constructed as:
```
[arm_joints (14)] + [left_ee_state (ee_dof)] + [right_ee_state (ee_dof)]
```

| End-effector | ee_dof | Total state dim |
|-------------|--------|----------------|
| Dex3        | 7      | 14 + 7 + 7 = 28 |
| Dex1        | 1      | 14 + 1 + 1 = 16 |

### Dex3 Joint Ordering Issue (CRITICAL)

**Legacy sim order** (what IsaacLab publishes via DDS):
```
[thumb0, thumb1, thumb2, middle0, middle1, index0, index1]  (per hand)
```

**Policy/dataset order** (what the model expects):
```
[thumb0, thumb1, thumb2, index0, index1, middle0, middle1]  (per hand)
```

The permutation `[0, 1, 2, 5, 6, 3, 4]` maps between them. It is self-inverse.

**Where this is handled**:
- `eval_g1_sim.py` checks `cfg.dex3_right_order_legacy_sim` (default: `True`)
- `reorder_dex3_right_legacy_sim()` in `utils/dex3_order.py` applies the permutation
- Applied to BOTH observation (right_ee_state input) AND action (right_ee_action output)
- Only applies to the RIGHT hand -- left hand order matches between sim and policy

### Dex3 Sim Joint Indices (in `dex3_state.py`)

The sim reports 14 hand joints at indices:
```python
gripper_joint_indices = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
```
First 7 = left hand, last 7 = right hand (in sim order).

### Dex1 Stroke Conversion

Dex1 sim reports joint angles [0.03 (open) to -0.02 (closed)], but training data uses stroke [0 (open) to 5.4 (closed)]. The conversion is handled in `eval_g1_sim.py` when `cfg.ee == "dex1" and cfg.sim`.

Do NOT apply this conversion for Dex3 -- Dex3 joints are used directly.

### Action Dimensions

Actions follow the same layout as observations:
```
action = [arm_joints (14)] + [left_ee_action (ee_dof)] + [right_ee_action (ee_dof)]
```

### Arm Joint Ordering (14-dim)

Both observation and action arm joints follow the policy convention:
```
[left_arm (7): shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw]
[right_arm (7): same order]
```

---

## 5. IK Prior Configuration for Inference

### What It Is

The IK prior (`~/unitree_IL_lerobot/unitree_lerobot/eval_robot/ik_prior.py`) computes a straight-line joint-space trajectory from current arm pose to a target 3D position. This trajectory is injected as SDEdit conditioning for the DiT action head.

### Training Configuration

The DiT was trained with `ik_prior_prob=0.4`, meaning 40% of training batches used an IK trajectory as the denoising start point instead of pure noise.

### Inference Usage

At inference, providing the IK trajectory improves reaching behavior. The trajectory is passed via `action_input["ik_trajectory"]` to the DiT.

### Coordinate Frame

- **IK target must be in robot base frame** (not camera frame, not world frame)
- The `compute_ik_trajectory_from_bbox()` helper converts: bbox + depth -> camera 3D -> robot 3D via `cam_extrinsics @ p_cam`
- The `cam_extrinsics` is a (4,4) camera-to-robot transform

### URDF and Joint Mapping

- URDF: `eval_robot/assets/g1/g1_body29_hand14.urdf` (29 body + 14 hand DOF)
- Policy arm indices 0-6 map to URDF joints 15-21 (left arm)
- Policy arm indices 7-13 map to URDF joints 29-35 (right arm)
- The IKPriorComputer handles this mapping internally

### Hand Selection

- `IKPriorComputer.select_hand()` picks whichever hand's end-effector is closer to the target
- Can be overridden with explicit `hand="left"` or `hand="right"`

### Output Format

- Returns `(n_steps, 28)` trajectory: 14 arm dims interpolated, 14 finger dims = 0
- `n_steps` must match `action_horizon` (default 16)

### Recommended Inference Settings

- For block stacking evaluation: use `ik_prior_prob=1.0` (always use IK prior) if the task involves reaching a known target position
- For general-purpose evaluation: keep `ik_prior_prob=0.4` to match training distribution
- The IK prior is SDEdit conditioning (a better starting point for denoising), NOT direct IK control -- the DiT still generates the final action

---

## 6. Common Pitfalls

### Using Dex1 Environment Instead of Dex3

- **Task name matters**: `Isaac-Stack-RgyBlock-G129-Dex3-Joint` vs. `Isaac-Stack-RgyBlock-G129-Dex1-Joint`
- **DDS flag matters**: `--enable_dex3_dds` vs. `--enable_dex1_dds`
- **Eval client flag matters**: `--ee dex3` vs. `--ee dex1`
- **Mismatching any of these causes silent failures**: wrong joint counts, wrong camera positions, actions applied to wrong joints

### Dataset Mismatch

- Dex3 dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (28-dim state/action)
- Dex1 dataset: `unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim` (16-dim state/action)
- The dataset provides normalization statistics. Using the wrong dataset means incorrect unnormalization of actions.

### Normalization Stats from Stale Cache

- The `meta/` directory at the dataset cache root can contain stale stats from a different dataset version
- Symptoms: actions clamped to wrong range, robot barely moves or moves erratically
- Defense: `reinject_dataset_stats()` in `train_utils.py` re-applies correct stats; at eval time, the dataset stats are passed as explicit overrides to the preprocessor/postprocessor

### Image Key Filtering

`eval_g1_sim.py` (line 265-268) drops image keys that the model was not trained on:
```python
trained_img_keys = {k for k in dataset.meta.features if k.startswith("observation.images.")}
for k in list(observation):
    if k.startswith("observation.images.") and k not in trained_img_keys:
        observation.pop(k)
```
If the sim produces camera keys that don't match the training dataset feature names, images are silently dropped. Verify camera names match between sim and dataset.

### Network Latency for System 1

System 1 (DiT) must run locally (same machine as sim). At 30Hz, each step has a 33ms budget. Network round-trip latency of 10-50ms consumes most of this budget. System 2 (VLM at ~1Hz) can run remotely without penalty.

### DDS Configuration

All processes must share the same DDS configuration:
- `UNITREE_DDS_IFACE=lo` (loopback for local sim)
- `UNITREE_DDS_DOMAIN_ID=1` (must match across all processes)
- `CYCLONEDDS_URI` pointing to the same XML config

### Working Directory for Eval Client

`eval_g1_sim.py` must be run from `~/unitree_IL_lerobot` because `robot_arm_ik.py` uses relative URDF paths from that directory.

### Conda Environment Separation

- `unitree_sim_env`: IsaacLab simulation + image compositor
- `unitree_lerobot`: Eval client + policy servers (requires transformers, lerobot, etc.)
- Never mix these -- IsaacLab has strict dependency requirements that conflict with transformers/lerobot.

---

## 7. Evaluation Metrics

### What to Measure

For block stacking, the metrics hierarchy is:
1. **Tower rate** (primary): All 3 blocks successfully stacked
2. **Per-block place rate**: Each block placed within tolerance
3. **Per-block lift rate**: Each block lifted above threshold

### Automated Success Detection

The sim environment computes rewards internally (`compute_reward` in mdp). The eval client can subscribe to reward signals via `sim_reward_subscriber` (DDS) to detect episode completion and success.

### Manual/Visual Inspection

For initial debugging, use `--visualization true` in the eval client to enable Rerun logging, which shows observation images, states, and actions in real-time.

# VR Teleoperation Dataset Collection Pipeline

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Camera Setup](#3-camera-setup)
4. [Depth Format](#4-depth-format)
5. [Data Format](#5-data-format)
6. [Launch Instructions](#6-launch-instructions)
7. [Keyboard Controls](#7-keyboard-controls)
8. [Modified Files](#8-modified-files)

---

## 1. Overview

Data collection for the block stacking task uses VR teleoperation (Apple Vision Pro or Meta Quest via `xr_teleoperate`). The operator wears the headset, sees a live stereo view from the robot's head cameras, and performs the stacking task. Data is recorded in real time during teleoperation — no replay step is needed.

**Task**: `Isaac-Stack-RgyBlock-G129-Dex3-Joint`
**Robot**: G1 29DOF + Dex3 hands
**Sim**: IsaacLab, DDS channel 1 (loopback)
**Operator interface**: Vuer web UI at `https://vuer.ai?ws=wss://192.168.1.201:8012&grid=False`

Each episode captures:
- RGB images from 4 camera views (2 head binocular eyes, 2 wrists)
- Float32 depth maps from 4 views (head left/right split, 2 wrists)
- Arm joint positions and torques (7 per arm)
- Hand joint positions (7 per hand)
- IK-solved target actions
- 6-fingertip tactile contact forces (18 floats total)
- Object positions in world and robot-base frames

---

## 2. Architecture

Two processes run on the same machine (the desktop at `192.168.1.201`):

```
Terminal 1: IsaacLab Simulation        (unitree_sim_env conda env)
Terminal 2: xr_teleoperate             (tv conda env)
VR Headset: Vuer browser interface     (connects via WebSocket on port 8012)
```

Communication paths:

```
IsaacLab sim
  ├── DDS rt/lowstate      →  xr_teleoperate (arm/hand joint states + tau_est)
  ├── DDS rt/sim_state     →  xr_teleoperate (object positions, tactile forces)
  ├── ZMQ ports 55555-55561 →  xr_teleoperate image client (RGB frames)
  └── Named SHM (float32)  →  xr_teleoperate depth reader (bypasses ZMQ)

xr_teleoperate
  ├── DDS rt/arm_sdk        →  IsaacLab sim (IK-solved arm targets)
  ├── DDS rt/dex3_cmd       →  IsaacLab sim (hand finger targets)
  └── WebSocket port 8012   →  VR headset (stereo video + hand tracking)
```

### Depth bypass via shared memory

Float32 depth data is too large for low-latency ZMQ transmission without lossy compression. Instead, the sim writes raw float32 depth arrays directly to named POSIX shared memory segments, and `xr_teleoperate` reads from those same segments. The ZMQ depth ports (55559-55561) carry grayscale uint8 PNG previews for monitoring only — the actual recorded `.npy` files come from SHM.

---

## 3. Camera Setup

Config file: `unitree_sim_isaaclab/teleimager/cam_config_server.yaml`

Config server port: `60000`

| Camera | ZMQ port | Resolution | Type | Notes |
|--------|----------|-----------|------|-------|
| `head_camera` | 55555 | 480×1280 | RGB JPEG | Binocular left+right combined side-by-side |
| `head_right_camera` | 55558 | 480×640 | RGB JPEG | Right eye only; merged into head ZMQ stream |
| `left_wrist_camera` | 55556 | 480×640 | RGB JPEG | Left Dex3 wrist |
| `right_wrist_camera` | 55557 | 480×640 | RGB JPEG | Right Dex3 wrist |
| `head_depth_camera` | 55559 | 480×640 | Depth (grayscale uint8 PNG, lossless) | Preview only; recording uses SHM |
| `left_wrist_depth_camera` | 55560 | 480×640 | Depth (grayscale uint8 PNG, lossless) | Preview only; recording uses SHM |
| `right_wrist_depth_camera` | 55561 | 480×640 | Depth (grayscale uint8 PNG, lossless) | Preview only; recording uses SHM |

Config notes:
- `binocular: true` — head camera concatenates left and right eyes horizontally into a single 480×1280 frame before ZMQ send
- `webrtc: false` — WebRTC is disabled; all streams go via ZMQ only
- Depth max ranges: head 3.0 m, left wrist 2.0 m, right wrist 2.0 m

### Shared memory segment names

| SHM name | Content |
|----------|---------|
| `isaac_head_depth_f32_shm` | Head camera depth (480×640 float32 meters) |
| `isaac_left_depth_f32_shm` | Left wrist depth (480×640 float32 meters) |
| `isaac_right_depth_f32_shm` | Right wrist depth (480×640 float32 meters) |

---

## 4. Depth Format

The sim uses **linear depth** (distance from camera plane in meters) stored as raw `float32`. This is the format required by the iDP3 `PointCloudDepthEncoder` inside CraftNet.

**Do not apply** colormap normalization (turbo, grayscale, etc.) to depth before saving. Colormap or uint8 conversion destroys the metric precision needed for 3D point cloud reconstruction.

Saved format: `.npy` files containing a `(480, 640)` array of `dtype=float32`. Values are in meters. No normalization is applied. `nan` or `inf` values indicate out-of-range pixels.

The head depth is split into left and right halves at read time:
- `head_left`: columns `[:, :640]` of the 480×1280 SHM array (written as two stacked 480×640 halves)
- `head_right`: columns `[:, 640:]`

---

## 5. Data Format

### Episode directory structure

```
episode_XXXX/
├── colors/
│   ├── 000000_color_0.jpg    # Head left eye (480×640)
│   ├── 000000_color_1.jpg    # Head right eye (480×640)
│   ├── 000000_color_2.jpg    # Left wrist (480×640)
│   └── 000000_color_3.jpg    # Right wrist (480×640)
├── depths/
│   ├── 000000_head_left_0.npy    # Float32 meters, 480×640
│   ├── 000000_head_right_1.npy   # Float32 meters, 480×640
│   ├── 000000_left_wrist_2.npy   # Float32 meters, 480×640
│   └── 000000_right_wrist_3.npy  # Float32 meters, 480×640
└── data.json
```

File naming: `XXXXXX` is the zero-padded step index within the episode.

### data.json — per step

Each entry in `data.json` contains one control step. Fields:

**States**

| Field | Shape | Description |
|-------|-------|-------------|
| `states.left_arm.qpos` | `[7]` | Left arm joint positions (rad) |
| `states.left_arm.torque` | `[7]` | Left arm estimated joint torques (N·m) |
| `states.right_arm.qpos` | `[7]` | Right arm joint positions (rad) |
| `states.right_arm.torque` | `[7]` | Right arm estimated joint torques (N·m) |
| `states.left_ee.qpos` | `[7]` | Left Dex3 hand joint positions |
| `states.right_ee.qpos` | `[7]` | Right Dex3 hand joint positions |

**Actions** (IK-solved targets sent to sim)

| Field | Shape | Description |
|-------|-------|-------------|
| `actions.left_arm.qpos` | `[7]` | Left arm target joint positions (rad) |
| `actions.right_arm.qpos` | `[7]` | Right arm target joint positions (rad) |

**Tactile**

| Field | Shape | Description |
|-------|-------|-------------|
| `tactiles.left_ee` | `[9]` | Left hand contact forces: thumb xyz, middle xyz, index xyz |
| `tactiles.right_ee` | `[9]` | Right hand contact forces: thumb xyz, middle xyz, index xyz |

Tactile values come from `sim_state["tactile"]` via `rt/sim_state`. Each fingertip reports x/y/z forces in Newtons.

**Sim state**

| Field | Description |
|-------|-------------|
| `sim_state.object_positions.red_block.world` | Red block position [x, y, z] in world frame |
| `sim_state.object_positions.red_block.rel` | Red block position [x, y, z] relative to robot base |
| `sim_state.object_positions.yellow_block.world` | Yellow block world position |
| `sim_state.object_positions.yellow_block.rel` | Yellow block robot-base-relative position |
| `sim_state.object_positions.green_block.world` | Green block world position |
| `sim_state.object_positions.green_block.rel` | Green block robot-base-relative position |
| `sim_state.object_positions.robot_pos` | Robot base position [x, y, z] in world frame |
| `sim_state.object_positions.robot_quat_wxyz` | Robot base orientation as quaternion [w, x, y, z] |
| `sim_state.init_state` | Full IsaacLab scene state snapshot (used for environment reset) |

Object positions are published every step via the `rt/sim_state` DDS topic.

---

## 6. Launch Instructions

Open two terminals. Set DDS environment variables in each:

```bash
export UNITREE_DDS_IFACE=lo
export UNITREE_DDS_DOMAIN_ID=1
export CYCLONEDDS_URI=file:///home/cosmos/cyclonedds_local.xml
```

### Terminal 1 — IsaacLab simulation

```bash
cd /home/cosmos/robotics/unitree_sim_isaaclab
conda run -n unitree_sim_env --no-capture-output python sim_main.py \
    --task Isaac-Stack-RgyBlock-G129-Dex3-Joint \
    --enable_dex3_dds \
    --robot_type g129 \
    --enable_cameras \
    --enable_depth \
    --device cpu
```

Wait for the sim window to appear and the robot to settle before proceeding.

### Terminal 2 — xr_teleoperate

```bash
cd /home/cosmos/xr_teleoperate/teleop
conda run -n tv --no-capture-output python teleop_hand_and_arm.py \
    --arm G1_29 \
    --ee dex3 \
    --sim \
    --img-server-ip 127.0.0.1 \
    --network-interface lo \
    --record \
    --task-dir ./utils/data/ \
    --task-name "block_stacking"
```

### VR headset connection

Put on the headset and navigate to:

```
https://vuer.ai?ws=wss://192.168.1.201:8012&grid=False
```

The Vuer interface streams stereo RGB from the head cameras. Hand tracking data is sent back to `xr_teleoperate` over the same WebSocket.

### Collection workflow

1. Start Terminal 1 (sim), wait for startup.
2. Start Terminal 2 (xr_teleoperate).
3. Connect VR headset via Vuer URL.
4. Press `r` to sync the robot with your hand movements.
5. Press `s` to begin recording.
6. Perform the block stacking task.
7. Press `s` again to stop and save the episode.
8. Press `e` to reset the environment (repositions blocks to a new random configuration).
9. Repeat from step 4 for the next episode.
10. Press `q` to exit when done.

Episodes are saved incrementally to `--task-dir` as `episode_0000/`, `episode_0001/`, etc.

---

## 7. Keyboard Controls

All keys are active in the terminal running `xr_teleoperate`.

| Key | Action |
|-----|--------|
| `r` | Start robot sync — robot begins mirroring VR hand movements |
| `s` | Toggle recording — first press starts, second press saves the episode |
| `e` | Reset environment — repositions blocks to a random initial configuration |
| `q` | Stop teleoperation and exit |

---

## 8. Modified Files

These files were changed from their upstream versions to support this pipeline:

| File | Change |
|------|--------|
| `unitree_sim_isaaclab/teleimager/cam_config_server.yaml` | Binocular head camera config; depth ports 55559-55561; WebRTC disabled |
| `unitree_sim_isaaclab/teleimager/src/teleimager/image_server.py` | Binocular SHM fix (combines `head_camera` + `head_right_camera`); lossless PNG encoding for depth ZMQ streams |
| `unitree_sim_isaaclab/tasks/common_observations/camera_state.py` | Float32 SHM depth writer; linear (camera-plane) depth instead of ray-cast depth |
| `unitree_sim_isaaclab/sim_main.py` | Publishes object positions and tactile forces in `rt/sim_state` every step |
| `xr_teleoperate/teleop/teleop_hand_and_arm.py` | Float32 depth recording from SHM; tactile extraction from `sim_state`; torque fields in `states`; environment reset on `e` key |
| `xr_teleoperate/teleop/robot_control/robot_arm.py` | Reads `tau_est` from `rt/lowstate` motor state; exposes `get_current_dual_arm_tau()` |
| `xr_teleoperate/teleop/teleimager/src/teleimager/image_client.py` | Float32 SHM depth reader methods for head, left wrist, and right wrist |
| `xr_teleoperate/teleop/utils/episode_writer.py` | Saves depth frames as `.npy` (float32) instead of JPEG or uint8 PNG |

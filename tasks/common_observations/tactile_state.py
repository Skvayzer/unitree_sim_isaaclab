# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Tactile observation for Dex3 — scalar contact force magnitude per pad link.

Output: (num_envs, 16) — 8 links per hand × 2 hands, in fixed order.
Each value = magnitude of net contact force on that pad link (Newtons).

Real robot mapping (SDK press_sensor_state[0..8] per hand):
  Sim uses 8 URDF links per hand; real Dex3 SDK exposes 9 modules per hand.
  The missing 9th (likely inner palm zone) has no separate URDF link.
  Calibrate sim→real with affine fit or binary threshold before deployment.

Link order (same for left_ and right_ prefix):
  [0] palm_link
  [1] thumb_0_link  (proximal)
  [2] thumb_1_link  (mid)
  [3] thumb_2_link  (tip)
  [4] middle_0_link (proximal)
  [5] middle_1_link (tip)
  [6] index_0_link  (proximal)
  [7] index_1_link  (tip)
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# All 16 pad links in fixed order: left hand first, then right hand.
# This order is the contract between sim training and real-robot deployment.
DEX3_PAD_LINKS = [
    "left_hand_palm_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
]

_pad_indices: list[int] | None = None


def get_tactile_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return scalar contact force magnitude per pad link.

    Shape: (num_envs, 16)
    Order: DEX3_PAD_LINKS (left palm, left thumb 0/1/2, left middle 0/1,
           left index 0/1, then same for right hand).
    """
    global _pad_indices

    sensor = env.scene["fingertip_contacts"]
    forces = sensor.data.net_forces_w  # (num_envs, n_bodies, 3)

    if _pad_indices is None:
        body_names = list(sensor.body_names)
        print(f"[tactile] Sensor covers {len(body_names)} bodies: {body_names}")
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        _pad_indices = []
        for link in DEX3_PAD_LINKS:
            idx = name_to_idx.get(link)
            if idx is None:
                raise RuntimeError(
                    f"[tactile] Link '{link}' not found in contact sensor bodies.\n"
                    f"Available: {body_names}\n"
                    f"Check ContactSensorCfg prim_path covers all hand links."
                )
            _pad_indices.append(idx)
        print(f"[tactile] Pad index mapping built: {dict(zip(DEX3_PAD_LINKS, _pad_indices))}")

    selected = forces[:, _pad_indices, :]          # (num_envs, 16, 3)
    magnitudes = selected.norm(dim=-1)             # (num_envs, 16)
    return magnitudes


# ---------------------------------------------------------------------------
# Extended 64-D tactile observation (4 channels × 16 pads)
# ---------------------------------------------------------------------------

CONTACT_THRESHOLD = 0.05   # N — minimum force to count as contact
PRESSURE_CLIP     = 10.0   # N — log1p normalization reference (not a hard clip)
_LOG1P_SCALE      = 2.3979  # = math.log1p(10.0) — precomputed to avoid import
MAX_DURATION_STEPS = 50    # steps before duration saturates at 1.0

# Per-env stateful tensors (reset on episode boundary).
_prev_binary: "torch.Tensor | None" = None
_contact_duration: "torch.Tensor | None" = None


def reset_tactile_state(env_ids: "torch.Tensor | None" = None, N: int = 0, device=None) -> None:
    """Clear per-step tactile state for specific envs (or all if env_ids is None).

    Call after each env reset so contact-duration and delta channels don't
    carry stale state from the previous episode.
    """
    global _prev_binary, _contact_duration
    if env_ids is None or _prev_binary is None:
        _prev_binary = None
        _contact_duration = None
    else:
        _prev_binary[env_ids] = 0.0
        _contact_duration[env_ids] = 0.0


def get_tactile_obs_extended(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return 64-D extended tactile observation.

    Channels (16 pads each, concat in this order):
      pressure  [0:16]  — normalized force magnitude [0, 1]
      binary    [16:32] — 1 if force > CONTACT_THRESHOLD else 0
      delta     [32:48] — +1 new contact, −1 lost contact, 0 stable
      duration  [48:64] — steps continuously in contact, normalized [0, 1]

    Shape: (num_envs, 64)
    Deployment note: pressure/binary map directly to Dex3 press_sensor_state;
    delta/duration are computed from the SDK's streaming data the same way.
    """
    global _prev_binary, _contact_duration

    pressure_raw = get_tactile_obs(env)            # (N, 16)
    N = pressure_raw.shape[0]
    device = pressure_raw.device

    pressure = (torch.log1p(pressure_raw) / _LOG1P_SCALE).clamp(0.0, 1.0)
    binary   = (pressure_raw > CONTACT_THRESHOLD).float()

    if _prev_binary is None or _prev_binary.shape[0] != N or _prev_binary.device != device:
        _prev_binary      = binary.clone()
        _contact_duration = torch.zeros(N, 16, device=device)

    delta = binary - _prev_binary                              # {-1, 0, +1}
    # duration increments while in contact, resets to 0 on release
    _contact_duration = (_contact_duration + binary) * binary
    duration = (_contact_duration / MAX_DURATION_STEPS).clamp(0.0, 1.0)

    _prev_binary = binary.clone()

    return torch.cat([pressure, binary, delta, duration], dim=1)  # (N, 64)

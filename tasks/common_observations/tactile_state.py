# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Tactile observation for Dex3 — scalar contact force magnitude per pad module.

Output: (num_envs, 18) — 9 modules per hand × 2 hands, in fixed order.
Each value = magnitude of net contact force on that pad module (Newtons).

Real robot mapping (SDK press_sensor_state[0..8] per hand):
  9 modules per hand, matching the 9 entries below.
  Palm is spatially subdivided into 3 equal zones (middle-side/centre/index-side).
  No thumb_2 module on real hardware — sim's thumb_2_link is excluded here.
  Idle noise on real hardware: ~20-30 ADC counts; clear touch >= 100 ADC.

Module order (same for left_ and right_ prefix):
  [0] palm_0_zone  (middle side) — equal-split of palm_link force
  [1] palm_1_zone  (centre)      — equal-split of palm_link force
  [2] palm_2_zone  (index side)  — equal-split of palm_link force
  [3] thumb_0_link (proximal)
  [4] thumb_1_link (tip)
  [5] middle_0_link (proximal)
  [6] middle_1_link (tip)
  [7] index_0_link  (proximal)
  [8] index_1_link  (tip)

Sim->real alignment notes:
  - Palm zones share one URDF body link; each zone receives palm_force / 3.
    The real SDK provides per-zone contact via spatial position of press contacts;
    equal-split is the valid sim approximation until a spatial contact API is exposed.
  - Calibrate sim->real with affine fit or binary threshold before deployment.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# All 18 pad modules in fixed order: left hand first, then right hand.
# Palm zones share the same URDF body link; equal-split is applied in get_tactile_obs.
# This order is the contract between sim training and real-robot deployment.
DEX3_PAD_LINKS = [
    # Left hand — 9 modules
    "left_hand_palm_link",      # L-m0 palm_0 (middle side)  } equal-split
    "left_hand_palm_link",      # L-m1 palm_1 (centre)       } of palm_link
    "left_hand_palm_link",      # L-m2 palm_2 (index side)   } force / 3
    "left_hand_thumb_0_link",   # L-m3 thumb_0 (proximal)
    "left_hand_thumb_1_link",   # L-m4 thumb_1 (tip)
    "left_hand_middle_0_link",  # L-m5 middle_0 (proximal)
    "left_hand_middle_1_link",  # L-m6 middle_1 (tip)
    "left_hand_index_0_link",   # L-m7 index_0 (proximal)
    "left_hand_index_1_link",   # L-m8 index_1 (tip)
    # Right hand — 9 modules
    "right_hand_palm_link",     # R-m0 palm_0 (middle side)  } equal-split
    "right_hand_palm_link",     # R-m1 palm_1 (centre)       } of palm_link
    "right_hand_palm_link",     # R-m2 palm_2 (index side)   } force / 3
    "right_hand_thumb_0_link",  # R-m3 thumb_0 (proximal)
    "right_hand_thumb_1_link",  # R-m4 thumb_1 (tip)
    "right_hand_middle_0_link", # R-m5 middle_0 (proximal)
    "right_hand_middle_1_link", # R-m6 middle_1 (tip)
    "right_hand_index_0_link",  # R-m7 index_0 (proximal)
    "right_hand_index_1_link",  # R-m8 index_1 (tip)
]

# Output indices that are palm equal-split zones; each receives palm_force / 3.
_PALM_ZONE_INDICES = [0, 1, 2, 9, 10, 11]

_pad_indices: list[int] | None = None


def get_tactile_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return scalar contact force magnitude per pad module.

    Shape: (num_envs, 18)
    Order: DEX3_PAD_LINKS (left palm_0/1/2, left thumb_0/1, left middle_0/1,
           left index_0/1, then same 9 for right hand).

    Palm zones: each palm slot = palm_link_force / 3 (equal-split approximation).
    """
    global _pad_indices

    sensor = env.scene["fingertip_contacts"]
    forces = sensor.data.net_forces_w  # (num_envs, n_bodies, 3)

    if _pad_indices is None:
        body_names = list(sensor.body_names)
        print(f"[tactile] Sensor covers {len(body_names)} bodies: {body_names}")
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        _pad_indices = []
        # DEX3_PAD_LINKS contains duplicate palm entries — intentional for equal-split.
        for link in DEX3_PAD_LINKS:
            idx = name_to_idx.get(link)
            if idx is None:
                raise RuntimeError(
                    f"[tactile] Link '{link}' not found in contact sensor bodies.\n"
                    f"Available: {body_names}\n"
                    f"Check ContactSensorCfg prim_path covers all hand links."
                )
            _pad_indices.append(idx)
        print(f"[tactile] Pad index mapping built: {len(_pad_indices)} entries")

    selected = forces[:, _pad_indices, :]          # (num_envs, 18, 3)
    magnitudes = selected.norm(dim=-1)             # (num_envs, 18)
    # Equal-split palm: 3 zones share one body link, each receives 1/3 of the force.
    magnitudes = magnitudes.clone()
    magnitudes[:, _PALM_ZONE_INDICES] = magnitudes[:, _PALM_ZONE_INDICES] / 3.0
    return magnitudes


# ---------------------------------------------------------------------------
# Extended 72-D tactile observation (4 channels × 18 pads)
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
    """Return 72-D extended tactile observation.

    Channels (18 pads each, concat in this order):
      pressure  [0:18]  — normalized force magnitude [0, 1]
      binary    [18:36] — 1 if force > CONTACT_THRESHOLD else 0
      delta     [36:54] — +1 new contact, -1 lost contact, 0 stable
      duration  [54:72] — steps continuously in contact, normalized [0, 1]

    Shape: (num_envs, 72)
    Deployment note: pressure/binary map directly to Dex3 press_sensor_state;
    delta/duration are computed from the SDK's streaming data the same way.
    """
    global _prev_binary, _contact_duration

    pressure_raw = get_tactile_obs(env)            # (N, 18)
    N = pressure_raw.shape[0]
    device = pressure_raw.device

    pressure = (torch.log1p(pressure_raw) / _LOG1P_SCALE).clamp(0.0, 1.0)
    binary   = (pressure_raw > CONTACT_THRESHOLD).float()

    if _prev_binary is None or _prev_binary.shape[0] != N or _prev_binary.device != device:
        _prev_binary      = binary.clone()
        _contact_duration = torch.zeros(N, 18, device=device)

    delta = binary - _prev_binary                              # {-1, 0, +1}
    # duration increments while in contact, resets to 0 on release
    _contact_duration = (_contact_duration + binary) * binary
    duration = (_contact_duration / MAX_DURATION_STEPS).clamp(0.0, 1.0)

    _prev_binary = binary.clone()

    return torch.cat([pressure, binary, delta, duration], dim=1)  # (N, 72)

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

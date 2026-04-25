"""Dense per-step reward for System 0 RL training.

All signals are proprioceptive (EE position via FK, contact forces, torques)
so the same reward can be computed on the real robot without any object tracking.
"""

import torch
from .scripted_controller import Phase

TABLE_Z = 0.822


def compute_reward(
    env, phase, hand, block_name, current_block_idx,
    coarse_targets, refined_targets, config, device="cpu", env_idx=0,
):
    reward = 0.0

    try:
        ee_pos = _get_ee_pos(env, hand, env_idx)
        contact = _get_contact_force(env, hand, env_idx)
    except (KeyError, AttributeError, IndexError):
        return 0.0

    stack_target = torch.tensor([-4.19, -3.95, TABLE_Z], device=device)
    place_pos = stack_target.clone()
    place_pos[2] += current_block_idx * config.block_height + config.block_height
    dist_to_place = (ee_pos[:2] - place_pos[:2]).norm().item()

    if phase == Phase.APPROACH:
        pass  # no reward — policy must find the block via exploration and tactile feedback

    elif phase == Phase.PRE_GRASP:
        reward += 0.2

    elif phase == Phase.GRASP:
        reward += 2.0 * (contact > config.grasp_force_threshold)
        reward += -0.1 * max(0, contact - 5.0)

    elif phase == Phase.LIFT:
        ee_height = ee_pos[2].item() - TABLE_Z
        reward += 3.0 * max(0, ee_height)
        reward += 1.0 * (ee_height > config.lift_height * 0.8)

    elif phase == Phase.TRANSPORT:
        reward += -1.5 * dist_to_place
        reward += 0.5 * (dist_to_place < config.approach_dist * 2)

    elif phase == Phase.DESCEND:
        height_error = abs(ee_pos[2].item() - place_pos[2].item())
        reward += -2.0 * height_error
        reward += -0.3 * max(0, contact - 3.0)

    elif phase == Phase.RELEASE:
        # Contact drop near place target = proxy for successful block placement
        reward += 5.0 * (contact < config.grasp_force_threshold)

    elif phase == Phase.RETREAT:
        reward += 0.1

    try:
        torques = env.scene["robot"].data.applied_torque[env_idx, -28:]
        reward += -0.005 * (torques ** 2).sum().item()
    except (KeyError, AttributeError, IndexError):
        pass

    delta_q = refined_targets - coarse_targets
    reward += -1.0 * (delta_q ** 2).sum().item()

    return reward


def _get_ee_pos(env, hand, env_idx=0):
    body_names = env.scene["robot"].data.body_names
    wrist_name = f"{hand}_wrist_yaw"
    for i, name in enumerate(body_names):
        if wrist_name in name:
            return env.scene["robot"].data.body_pos_w[env_idx, i]
    return torch.zeros(3)


def _get_contact_force(env, hand, env_idx=0):
    try:
        forces = env.scene["fingertip_contacts"].data.net_forces_w[env_idx]
        if hand == "right":
            return forces[3:6].norm(dim=-1).sum().item()
        else:
            return forces[0:3].norm(dim=-1).sum().item()
    except (KeyError, AttributeError):
        return 0.0

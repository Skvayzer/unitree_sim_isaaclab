"""
Reward functions for multi-block tower stacking.

Extends single-block rewards with:
- Per-block stack height targets
- Tower completion bonus
- Current block tracking
"""

import torch
from experiments.system0_skills.arm_trajectory import Phase


def compute_multi_block_reward(
    phase_ids: torch.Tensor,       # [num_envs] int (0-7 within cycle)
    block_idx: torch.Tensor,       # [num_envs] int (0, 1, or 2)
    current_block_z: torch.Tensor, # [num_envs] z of the block being manipulated
    current_block_xy: torch.Tensor,# [num_envs, 2] xy of current block
    target_xy: torch.Tensor,       # [num_envs, 2] target placement xy
    contact_forces: torch.Tensor,  # [num_envs, 3] force magnitudes
    action: torch.Tensor,          # [num_envs, 7]
    block_initial_z: float,
    stack_heights: list,           # [3] target z for each block
    cfg,
    block_was_lifted: torch.Tensor = None,  # [num_envs] bool — gates placement rewards
) -> tuple:
    """Compute reward for multi-block stacking.

    Returns:
        reward: [num_envs] total reward
        info: dict of component rewards for logging
    """
    device = phase_ids.device
    num_envs = phase_ids.shape[0]
    reward = torch.zeros(num_envs, device=device)

    # Contact binary
    contacts_binary = (contact_forces > 0.1).float()
    n_contacts = contacts_binary.sum(dim=-1)

    # Lifted gate: prevents reward hacking when block starts near target
    lifted_gate = block_was_lifted.float() if block_was_lifted is not None else torch.ones(num_envs, device=device)

    # XY error to target
    if target_xy.dim() == 1:
        target_xy = target_xy.unsqueeze(0).expand(num_envs, 2)
    xy_error = (current_block_xy - target_xy).norm(dim=-1)

    # Stack height for current block
    stack_heights_t = torch.tensor(stack_heights, device=device, dtype=torch.float32)
    current_stack_height = stack_heights_t[block_idx]  # [num_envs]

    # === PRIMARY: Block lifted off table ===
    lift_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT) |
        (phase_ids == Phase.DESCEND_TO_PLACE) |
        (phase_ids == Phase.RELEASE_HOLD)
    )
    lift = (current_block_z - block_initial_z).clamp(min=0.0)
    lift_reward = cfg.reward_block_lifted * (lift / 0.05).clamp(max=1.0)
    reward += lift_phases.float() * lift_reward

    # === PRIMARY: Block placed at correct stack height ===
    retreat_phase = (phase_ids == Phase.RETREAT)
    z_error = (current_block_z - current_stack_height).abs()
    placed_well = (xy_error < 0.03) & (z_error < 0.04)   # tight for 4cm blocks — prevents toppling
    placed_partial = (xy_error < 0.06) & ~placed_well
    # Scale placement reward by block index (higher blocks = harder = more reward)
    place_multiplier = 1.0 + block_idx.float() * 0.5  # 1.0, 1.5, 2.0
    place_reward = (placed_well.float() * cfg.reward_block_placed + placed_partial.float() * 10.0) * place_multiplier
    reward += retreat_phase.float() * place_reward * lifted_gate

    # === DENSE: Approach-to-target during placement ===
    place_approach_phases = (
        (phase_ids == Phase.DESCEND_TO_PLACE) |
        (phase_ids == Phase.RELEASE_HOLD)
    )
    approach_reward = cfg.reward_approach_target * (1.0 - (xy_error / 0.20).clamp(max=1.0))
    reward += place_approach_phases.float() * approach_reward * lifted_gate

    # === DENSE: Finger opening during release ===
    release_phase = (phase_ids == Phase.RELEASE_HOLD)
    near_target = (xy_error < 0.10).float()
    release_reward = cfg.reward_release * (1.0 - n_contacts / 3.0) * near_target
    reward += release_phase.float() * release_reward * lifted_gate

    # === SECONDARY: Contact during grasp ===
    grasp_phases = (
        (phase_ids == Phase.DESCEND_TO_GRASP) |
        (phase_ids == Phase.GRASP_HOLD)
    )
    contact_reward = cfg.reward_contact_during_grasp * n_contacts
    reward += grasp_phases.float() * contact_reward

    # === BONUS: Hold during transport ===
    transport_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT) |
        (phase_ids == Phase.DESCEND_TO_PLACE)
    )
    hold_reward = (current_block_z > block_initial_z).float() * cfg.reward_hold_during_transport
    reward += transport_phases.float() * hold_reward * lifted_gate

    # === PENALTY: Block dropped during lift/transport ===
    drop_penalty_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT)
    )
    dropped = (current_block_z < block_initial_z - 0.02) & drop_penalty_phases
    reward += dropped.float() * (-cfg.penalty_block_dropped)

    # === PENALTY: Block knocked during approach ===
    approach_phases = (
        (phase_ids == Phase.HOVER_ABOVE) |
        (phase_ids == Phase.DESCEND_TO_GRASP)
    )
    knocked = (current_block_z < block_initial_z - 0.05) & approach_phases
    reward += knocked.float() * (-cfg.penalty_block_knocked)

    # === Action smoothness ===
    action_penalty = -cfg.penalty_action_magnitude * (action ** 2).sum(dim=-1)
    reward += action_penalty

    info = {
        "lift_reward": lift_reward.mean().item(),
        "contact_reward": (grasp_phases.float() * contact_reward).mean().item(),
        "place_reward": (retreat_phase.float() * place_reward).mean().item(),
        "approach_reward": (place_approach_phases.float() * approach_reward).mean().item(),
        "release_reward": (release_phase.float() * release_reward).mean().item(),
        "drop_penalty": dropped.float().mean().item(),
        "action_penalty": action_penalty.mean().item(),
        "n_contacts_during_grasp": (n_contacts * grasp_phases.float()).sum().item() / max(grasp_phases.sum().item(), 1),
        "blocks_lifted": (lift > 0.02).float().mean().item(),
        "blocks_placed": placed_well.float().mean().item(),
        "xy_error_at_release": (release_phase.float() * xy_error).sum().item() / max(release_phase.sum().item(), 1),
        "z_error_at_retreat": (retreat_phase.float() * z_error).sum().item() / max(retreat_phase.sum().item(), 1),
    }

    return reward, info


def compute_tower_bonus(
    block_positions: list,  # list of [num_envs, 3] tensors for each block
    target_xy: torch.Tensor,  # [num_envs, 2]
    stack_heights: list,  # [3] target z heights
    xy_tolerance: float = 0.05,
    z_tolerance: float = 0.05,
) -> tuple:
    """Check if tower is correctly stacked at episode end.

    Returns:
        tower_complete: [num_envs] bool
        blocks_correct: [num_envs] int (0-3)
        info: dict with per-block status
    """
    device = block_positions[0].device
    num_envs = block_positions[0].shape[0]

    if target_xy.dim() == 1:
        target_xy = target_xy.unsqueeze(0).expand(num_envs, 2)

    blocks_correct = torch.zeros(num_envs, device=device)
    per_block_correct = []

    for i, (block_pos, target_z) in enumerate(zip(block_positions, stack_heights)):
        xy_err = (block_pos[:, :2] - target_xy).norm(dim=-1)
        z_err = (block_pos[:, 2] - target_z).abs()
        correct = (xy_err < xy_tolerance) & (z_err < z_tolerance)
        blocks_correct += correct.float()
        per_block_correct.append(correct)

    tower_complete = blocks_correct >= 3.0

    info = {
        "tower_complete_frac": tower_complete.float().mean().item(),
        "blocks_correct_mean": blocks_correct.mean().item(),
    }
    for i, correct in enumerate(per_block_correct):
        info[f"block_{i}_correct_frac"] = correct.float().mean().item()

    return tower_complete, blocks_correct, info

"""
Reward functions for block stacking.

Outcome-based rewards: lift height, placement accuracy, contact during grasp.
Dense approach-to-target reward during placement phases.
Finger opening reward during release phase.
No force regulation (it dominated signal in previous failures).
"""

import torch
from experiments.system0_skills.arm_trajectory import Phase


def compute_block_stack_reward(
    phase_ids: torch.Tensor,     # [num_envs] int
    block_z: torch.Tensor,       # [num_envs]
    block_xy: torch.Tensor,      # [num_envs, 2]
    target_xy: torch.Tensor,     # [num_envs, 2] or [2]
    contact_forces: torch.Tensor,  # [num_envs, 3] force magnitudes per fingertip
    action: torch.Tensor,        # [num_envs, 7]
    block_initial_z: float,
    stack_height: float,
    cfg,
    block_was_lifted: torch.Tensor = None,  # [num_envs] bool — gates placement rewards
) -> tuple:
    """Compute reward for each environment.

    Returns:
        reward: [num_envs] total reward
        info: dict of component rewards for logging
    """
    device = phase_ids.device
    num_envs = phase_ids.shape[0]
    reward = torch.zeros(num_envs, device=device)

    # Contact binary: > 0.1N
    contacts_binary = (contact_forces > 0.1).float()  # [num_envs, 3]
    n_contacts = contacts_binary.sum(dim=-1)  # [num_envs]

    # Lifted gate: prevents reward hacking when block starts near target
    lifted_gate = block_was_lifted.float() if block_was_lifted is not None else torch.ones(num_envs, device=device)

    # Expand target_xy if needed
    if target_xy.dim() == 1:
        target_xy = target_xy.unsqueeze(0).expand(num_envs, 2)
    xy_error = (block_xy - target_xy).norm(dim=-1)

    # === PRIMARY: Block lifted off table ===
    # Active during LIFT, TRANSPORT, DESCEND_TO_PLACE, RELEASE_HOLD
    lift_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT) |
        (phase_ids == Phase.DESCEND_TO_PLACE) |
        (phase_ids == Phase.RELEASE_HOLD)
    )
    lift = (block_z - block_initial_z).clamp(min=0.0)
    lift_reward = cfg.reward_block_lifted * (lift / 0.05).clamp(max=1.0)
    reward += lift_phases.float() * lift_reward

    # === PRIMARY: Block placed at target ===
    retreat_phase = (phase_ids == Phase.RETREAT)
    z_error = (block_z - stack_height).abs()
    placed_well = (xy_error < 0.05) & (z_error < 0.05)
    placed_partial = (xy_error < 0.10) & ~placed_well
    place_reward = placed_well.float() * cfg.reward_block_placed + placed_partial.float() * 10.0
    reward += retreat_phase.float() * place_reward * lifted_gate

    # === DENSE: Approach-to-target reward during placement phases ===
    # Gives gradient toward target position during DESCEND_TO_PLACE and RELEASE_HOLD
    place_approach_phases = (
        (phase_ids == Phase.DESCEND_TO_PLACE) |
        (phase_ids == Phase.RELEASE_HOLD)
    )
    # Reward proportional to closeness: max 5.0 when xy_error=0, 0 when xy_error>=0.15
    approach_reward = cfg.reward_approach_target * (1.0 - (xy_error / 0.20).clamp(max=1.0))
    reward += place_approach_phases.float() * approach_reward * lifted_gate

    # === DENSE: Finger opening reward during RELEASE_HOLD ===
    # Encourage fingers to release the block once near target
    release_phase = (phase_ids == Phase.RELEASE_HOLD)
    # Low contact force = good during release (inverse of grasp reward)
    # Reward: +reward_release when no contacts, 0 when all 3 contacts
    near_target = (xy_error < 0.10).float()
    release_reward = cfg.reward_release * (1.0 - n_contacts / 3.0) * near_target
    reward += release_phase.float() * release_reward * lifted_gate

    # === SECONDARY: Contact during grasp phase ===
    grasp_phases = (
        (phase_ids == Phase.DESCEND_TO_GRASP) |
        (phase_ids == Phase.GRASP_HOLD)
    )
    contact_reward = cfg.reward_contact_during_grasp * n_contacts
    reward += grasp_phases.float() * contact_reward

    # === BONUS: Hold reward during transport ===
    transport_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT) |
        (phase_ids == Phase.DESCEND_TO_PLACE)
    )
    hold_reward = (block_z > block_initial_z).float() * cfg.reward_hold_during_transport
    reward += transport_phases.float() * hold_reward * lifted_gate

    # === PENALTY: Block dropped during lift/transport (NOT release — dropping is expected there) ===
    drop_penalty_phases = (
        (phase_ids == Phase.LIFT) |
        (phase_ids == Phase.TRANSPORT)
    )
    dropped = (block_z < block_initial_z - 0.02) & drop_penalty_phases
    reward += dropped.float() * (-cfg.penalty_block_dropped)

    # === PENALTY: Block knocked off during approach ===
    approach_phases = (
        (phase_ids == Phase.HOVER_ABOVE) |
        (phase_ids == Phase.DESCEND_TO_GRASP)
    )
    knocked = (block_z < block_initial_z - 0.05) & approach_phases
    reward += knocked.float() * (-cfg.penalty_block_knocked)

    # === SMALL: Action smoothness ===
    action_penalty = -cfg.penalty_action_magnitude * (action ** 2).sum(dim=-1)
    reward += action_penalty

    # Info dict for logging
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
    }

    return reward, info

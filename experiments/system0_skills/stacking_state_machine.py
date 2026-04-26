"""
Block Stacking State Machine — chains grasp + release specialists.

NO RL TRAINING. This is pure evaluation: scripted arm movement + learned
grasp/release specialists. Uses ParameterizedArmTrajectory (same as training)
to ensure arm motion matches what the specialist was trained with.

For each block: run one full pick-place cycle using ParameterizedArmTrajectory,
with grasp specialist during contact phases and release specialist during release.

3 blocks × ~310 steps = ~930 total steps.
"""

import torch
from experiments.system0_skills.block_stack_config import BlockStackConfig
from experiments.system0_skills.parameterized_trajectory import ParameterizedArmTrajectory
from experiments.system0_skills.arm_trajectory import Phase
from experiments.system0_skills.policy import System0Actor

CFG = BlockStackConfig()


class BlockStackingStateMachine:
    """Chains grasp + release specialists with ParameterizedArmTrajectory.

    Each block cycle uses the EXACT same arm trajectory as training.
    The grasp specialist runs during grasp/lift/transport/descend_to_place.
    The release specialist runs during release_hold.
    Fingers are open during hover/descend/retreat.
    """

    def __init__(
        self,
        grasp_actor: System0Actor,
        release_actor: System0Actor,
        device: str,
        num_envs: int,
        num_blocks: int = 3,
        action_ema: float = 0.7,
        lift_sp_boost: float = -0.2,
    ):
        self.grasp_actor = grasp_actor
        self.release_actor = release_actor
        self.device = device
        self.num_envs = num_envs
        self.num_blocks = num_blocks
        self.action_ema = action_ema
        self.lift_sp_boost = lift_sp_boost  # Extra SP offset for LIFT/TRANSPORT to clear adjacent blocks

        # Create trajectory object (will be re-configured per block)
        self.traj = ParameterizedArmTrajectory(CFG, device, num_envs)
        self.steps_per_cycle = self.traj.total_steps  # ~310 steps

        # Per-env state
        self.current_block = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cycle_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_action = torch.zeros(num_envs, 7, device=device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Tracking
        self.blocks_lifted = torch.zeros(num_envs, num_blocks, dtype=torch.bool, device=device)
        self.blocks_placed = torch.zeros(num_envs, num_blocks, dtype=torch.bool, device=device)
        self.total_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Block positions [num_blocks] y-coordinates (set by configure_blocks)
        self.block_y_positions = None
        # Stack height SP offsets per block (more negative SP = higher placement)
        self.sp_place_offsets = [0.0, -0.08, -0.16]  # empirical: ~4cm block height

    def configure_blocks(self, block_y_positions: list):
        """Set block y-positions for the current evaluation.

        Args:
            block_y_positions: list of y-coordinates for each block.
        """
        self.block_y_positions = block_y_positions
        # Configure trajectory for block 0
        self._configure_traj_for_block(0)

    def _configure_traj_for_block(self, block_idx: int):
        """Configure ParameterizedArmTrajectory for the given block.

        Sets the pick position (block's y) and adjusts place height
        for stacking on top of previously placed blocks.
        """
        if self.block_y_positions is None:
            return

        block_y = self.block_y_positions[block_idx]
        block_y_t = torch.full((self.num_envs,), block_y, device=self.device)
        self.traj.set_block_positions(block_y_t)

        # Boost LIFT and TRANSPORT height to clear adjacent blocks on the table.
        # The arm sweeps laterally during transport (from pick SR to place SR).
        # Without extra height, the carried block collides with adjacent blocks,
        # dropping success from ~80% to ~8%.
        # SP is more negative = higher. Base lift/transport SP=-0.3 raises ~5cm.
        # With boost=-0.2, total SP=-0.5 raises ~8-9cm, clearing 4cm blocks.
        if self.lift_sp_boost != 0.0:
            # Phase 3: LIFT end, Phase 4: TRANSPORT start/end
            self.traj.end_joints[:, 3, 0] += self.lift_sp_boost   # LIFT end
            self.traj.start_joints[:, 4, 0] += self.lift_sp_boost  # TRANSPORT start
            self.traj.end_joints[:, 4, 0] += self.lift_sp_boost    # TRANSPORT end
            self.traj.start_joints[:, 5, 0] += self.lift_sp_boost  # DESCEND_TO_PLACE start

        # Adjust place position SP for stack height
        sp_offset = self.sp_place_offsets[min(block_idx, len(self.sp_place_offsets) - 1)]
        if sp_offset != 0.0:
            # Modify the place and release joints to account for stack height
            # base_place has SP=0 for table-level placement
            # For higher blocks, we need less pitch (more negative SP raises hand)
            for phase_idx in [5, 6]:  # DESCEND_TO_PLACE, RELEASE_HOLD
                self.traj.end_joints[:, phase_idx, 0] += sp_offset
                self.traj.start_joints[:, phase_idx, 0] += sp_offset
            # Also adjust retreat start (it starts from place position)
            self.traj.start_joints[:, 7, 0] += sp_offset

    def build_obs(self, env) -> torch.Tensor:
        """Build 28D observation matching specialist training format."""
        robot = env.scene["robot"]
        hand_idx = torch.tensor(CFG.right_hand_indices, device=self.device)

        finger_pos = robot.data.joint_pos[:, hand_idx]
        finger_vel = robot.data.joint_vel[:, hand_idx]
        forces = env.scene["fingertip_contacts"].data.net_forces_w
        right_forces = forces[:, CFG.right_fingertip_contact_indices, :]
        force_mags = right_forces.norm(dim=-1).clamp(0, 10.0)
        contact_binary = (force_mags > 0.1).float()

        # Phase encoding from trajectory (matches training exactly)
        phase_onehot = self.traj.get_phase_onehot(self.cycle_step)

        obs = torch.cat([finger_pos, finger_vel, force_mags, contact_binary,
                         phase_onehot], dim=-1)
        return obs.nan_to_num(0.0).clamp(-10.0, 10.0)

    def step(self, env) -> torch.Tensor:
        """One step of the full stacking pipeline.

        Sets arm targets and returns finger actions [num_envs, 7].
        """
        robot = env.scene["robot"]
        arm_idx = torch.tensor(CFG.right_arm_indices, device=self.device)

        # Get arm targets from ParameterizedArmTrajectory (SAME as training)
        arm_targets = self.traj.get_arm_targets(self.cycle_step)
        robot.data.joint_pos_target[:, arm_idx] = arm_targets

        # Get phase for each env
        phase_ids = self.traj.get_phase_ids(self.cycle_step)

        # Build observation
        obs = self.build_obs(env)

        # Determine which specialist to use
        is_release_phase = (phase_ids == Phase.RELEASE_HOLD)

        # Batched inference for all envs
        with torch.no_grad():
            grasp_mean, _ = self.grasp_actor(obs)
            release_mean, _ = self.release_actor(obs)

        # Use grasp policy output for ALL phases except release.
        # During HOVER/DESCEND/RETREAT, the policy's output provides
        # beneficial finger pre-positioning that improves grasp success
        # (forcing -1.0 here drops success from ~80% to ~33%).
        action = grasp_mean.clamp(-1.0, 1.0)
        # Release specialist for release phase
        if is_release_phase.any():
            action[is_release_phase] = release_mean[is_release_phase].clamp(-1.0, 1.0)

        # Done envs stay idle
        action[self.done] = 0.0

        # EMA smoothing
        smoothed = self.action_ema * action + (1 - self.action_ema) * self.prev_action
        self.prev_action = smoothed.clone()

        # Advance step counters
        self.cycle_step += 1
        self.total_steps += 1

        # Check for cycle completion (per env)
        cycle_done = self.cycle_step >= self.steps_per_cycle
        if cycle_done.any():
            done_mask = cycle_done & ~self.done
            if done_mask.any():
                # Advance to next block
                next_block = self.current_block[done_mask] + 1
                finished = next_block >= self.num_blocks

                # Mark fully done envs
                finished_envs = done_mask.clone()
                finished_envs[done_mask] = finished
                self.done[finished_envs] = True

                # Reset cycle for envs with more blocks
                continuing = done_mask & ~finished_envs
                if continuing.any():
                    self.current_block[continuing] = next_block[~finished]
                    self.cycle_step[continuing] = 0
                    self.prev_action[continuing] = 0.0

                    # Reconfigure trajectory for next block
                    # Since all envs process the same block sequence,
                    # configure for the new block index
                    new_block_idx = next_block[~finished][0].item()
                    self._configure_traj_for_block(new_block_idx)

        return smoothed

    def reset_all(self):
        """Reset state machine for all environments."""
        self.current_block.zero_()
        self.cycle_step.zero_()
        self.prev_action.zero_()
        self.done.zero_()
        self.blocks_lifted.zero_()
        self.blocks_placed.zero_()
        self.total_steps.zero_()
        if self.block_y_positions is not None:
            self._configure_traj_for_block(0)

    def is_all_done(self) -> bool:
        """True if all envs have completed all block cycles."""
        return self.done.all().item()

    def get_status(self, env_idx: int = 0) -> str:
        """Get human-readable status for one env."""
        if self.done[env_idx]:
            return "DONE"
        block = self.current_block[env_idx].item()
        step = self.cycle_step[env_idx].item()
        phase = self.traj.get_phase_ids(self.cycle_step[env_idx:env_idx+1])[0].item()
        phase_name = Phase(phase).name
        return f"block={block} step={step} phase={phase_name}"

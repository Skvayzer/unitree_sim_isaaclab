"""
Multi-block arm trajectory for 3-block tower stacking.

Builds 3 concatenated pick-place cycles, each targeting a different block
position for pickup and a different stack height for placement.

Total: 3 * 310 = 930 steps per episode.

Each cycle uses the same 8 phases as ArmTrajectory, but with block-specific
grasp position (shoulder_roll) and place height (shoulder_pitch).

Usage:
    traj = MultiBlockArmTrajectory(config, device)
    phase_ids = traj.get_phase_ids(ep_step)       # [num_envs] (0-7, within current cycle)
    block_idx = traj.get_block_idx(ep_step)        # [num_envs] (0, 1, or 2)
    arm_targets = traj.get_arm_targets(ep_step)    # [num_envs, 7]
    phase_onehot = traj.get_phase_onehot(ep_step)  # [num_envs, 8]
    block_onehot = traj.get_block_onehot(ep_step)  # [num_envs, 3]
"""

import torch
from experiments.system0_skills.arm_trajectory import Phase, ARM_JOINT_NAMES


class MultiBlockArmTrajectory:
    """Vectorized multi-block arm trajectory: 3 pick-place cycles."""

    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.num_blocks = config.num_blocks  # 3

        c = config
        phase_durations = [
            c.steps_hover, c.steps_descend, c.steps_grasp_hold,
            c.steps_lift, c.steps_transport, c.steps_descend_place,
            c.steps_release_hold, c.steps_retreat,
        ]
        self.num_phases = 8
        self.steps_per_cycle = sum(phase_durations)  # 310
        self.total_steps = self.steps_per_cycle * self.num_blocks  # 930

        def dict_to_array(d):
            return torch.tensor([d[name] for name in ARM_JOINT_NAMES],
                                dtype=torch.float32, device=device)

        # Build per-block trajectories
        # For each block, we have 8 phases with specific start/end joint configs
        # Shape: [num_blocks, 8, 7] for start/end joints
        # Shape: [num_blocks, 8] for phase starts/ends/durations

        all_phase_starts = []   # [num_blocks, 8]
        all_phase_ends = []     # [num_blocks, 8]
        all_phase_durations = []  # [num_blocks, 8]
        all_start_joints = []   # [num_blocks, 8, 7]
        all_end_joints = []     # [num_blocks, 8, 7]

        for block_idx in range(self.num_blocks):
            # Block-specific arm configs
            hover_joints = c.arm_hover_joints_per_block[block_idx]
            grasp_joints = c.arm_grasp_joints_per_block[block_idx]

            # Lift joints: raise from grasp position
            lift_joints = dict(grasp_joints)
            lift_joints["right_shoulder_pitch_joint"] = -0.3

            # Transport joints: move laterally to place position
            transport_joints = {
                "right_shoulder_pitch_joint": -0.3,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_yaw_joint": 0.1,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            }

            # Place joints: descend to target with block-specific height
            place_sp = c.arm_place_sp_per_block[block_idx]
            place_joints = {
                "right_shoulder_pitch_joint": place_sp,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_yaw_joint": 0.1,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            }

            phase_joint_pairs = [
                (hover_joints, hover_joints),          # HOVER: hold
                (hover_joints, grasp_joints),           # DESCEND: hover -> grasp
                (grasp_joints, grasp_joints),           # GRASP_HOLD: hold
                (grasp_joints, lift_joints),            # LIFT: grasp -> lift
                (lift_joints, transport_joints),        # TRANSPORT: lift -> transport
                (transport_joints, place_joints),       # DESCEND_TO_PLACE: transport -> place
                (place_joints, place_joints),           # RELEASE_HOLD: hold
                (place_joints, lift_joints),            # RETREAT: place -> lift (return up)
            ]

            cycle_offset = block_idx * self.steps_per_cycle
            phase_starts = []
            phase_ends = []
            start_joints = []
            end_joints = []
            step = cycle_offset
            for dur, (s_dict, e_dict) in zip(phase_durations, phase_joint_pairs):
                phase_starts.append(step)
                phase_ends.append(step + dur)
                start_joints.append(dict_to_array(s_dict))
                end_joints.append(dict_to_array(e_dict))
                step += dur

            all_phase_starts.append(torch.tensor(phase_starts, dtype=torch.long, device=device))
            all_phase_ends.append(torch.tensor(phase_ends, dtype=torch.long, device=device))
            all_phase_durations.append(torch.tensor(phase_durations, dtype=torch.float32, device=device))
            all_start_joints.append(torch.stack(start_joints))
            all_end_joints.append(torch.stack(end_joints))

        # Stack: [num_blocks, 8]
        self.all_phase_starts = torch.stack(all_phase_starts)     # [3, 8]
        self.all_phase_ends = torch.stack(all_phase_ends)         # [3, 8]
        self.all_phase_durations = torch.stack(all_phase_durations)  # [3, 8]
        self.all_start_joints = torch.stack(all_start_joints)     # [3, 8, 7]
        self.all_end_joints = torch.stack(all_end_joints)         # [3, 8, 7]

    def get_block_idx(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return which block is being manipulated. [num_envs] -> [num_envs]."""
        return (ep_step.clamp(0, self.total_steps - 1) // self.steps_per_cycle).clamp(0, self.num_blocks - 1)

    def get_block_onehot(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return [num_envs, num_blocks] one-hot block encoding."""
        block_idx = self.get_block_idx(ep_step)
        onehot = torch.zeros(ep_step.shape[0], self.num_blocks, device=self.device)
        onehot.scatter_(1, block_idx.unsqueeze(1), 1.0)
        return onehot

    def get_phase_ids(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return phase ID (0-7) within current cycle. [num_envs] -> [num_envs]."""
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1)
        block_idx = self.get_block_idx(ep_step)  # [num_envs]
        cycle_step = ep_step_clamped - block_idx * self.steps_per_cycle  # [num_envs]

        # For each env, get phase starts/ends for its block
        # all_phase_starts: [3, 8], block_idx: [num_envs]
        # We need starts/ends relative to cycle start (subtract cycle offset)
        # Phase starts within cycle: [0, 30, 60, 110, 150, 200, 240, 280]
        # Use first block's starts as reference (they're all the same durations)
        starts = self.all_phase_starts[0]  # [8] — same pattern for all blocks
        ends = self.all_phase_ends[0] - 0  # [8]

        # But these include the cycle offset for block 0 (which is 0)
        # For generality, compute within-cycle boundaries
        phase_durations = self.all_phase_durations[0]  # [8]
        cum_durations = torch.zeros(9, device=self.device, dtype=torch.long)
        cum_durations[1:] = phase_durations.long().cumsum(0)

        # cycle_step in [0, steps_per_cycle)
        expanded = cycle_step.unsqueeze(1)  # [num_envs, 1]
        starts_within = cum_durations[:-1].unsqueeze(0)  # [1, 8]
        ends_within = cum_durations[1:].unsqueeze(0)     # [1, 8]

        in_phase = (expanded >= starts_within) & (expanded < ends_within)  # [num_envs, 8]
        phase_ids = in_phase.long().argmax(dim=1)  # [num_envs]

        # Handle past end
        past_end = ep_step >= self.total_steps
        phase_ids[past_end] = 7
        return phase_ids

    def get_arm_targets(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return interpolated arm joint targets. [num_envs] -> [num_envs, 7]."""
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1)
        block_idx = self.get_block_idx(ep_step)  # [num_envs]
        phase_ids = self.get_phase_ids(ep_step)  # [num_envs]

        # Gather start/end joints for each env's block and phase
        # all_start_joints: [3, 8, 7]
        start_j = self.all_start_joints[block_idx, phase_ids]  # [num_envs, 7]
        end_j = self.all_end_joints[block_idx, phase_ids]      # [num_envs, 7]

        # Compute interpolation within phase
        cycle_step = ep_step_clamped - block_idx * self.steps_per_cycle
        phase_durations = self.all_phase_durations[0]
        cum_durations = torch.zeros(9, device=self.device, dtype=torch.float32)
        cum_durations[1:] = phase_durations.cumsum(0)

        p_start = cum_durations[phase_ids]          # [num_envs]
        p_dur = phase_durations[phase_ids]           # [num_envs]

        t = ((cycle_step.float() - p_start) / p_dur.clamp(min=1.0)).clamp(0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smooth-step

        targets = start_j + t.unsqueeze(1) * (end_j - start_j)
        return targets

    def get_phase_onehot(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return [num_envs, 8] one-hot phase encoding."""
        phase_ids = self.get_phase_ids(ep_step)
        onehot = torch.zeros(ep_step.shape[0], 8, device=self.device)
        onehot.scatter_(1, phase_ids.unsqueeze(1), 1.0)
        return onehot

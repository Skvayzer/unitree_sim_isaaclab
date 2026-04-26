"""
Scripted arm trajectory for block stacking.

The arm follows a fixed sequence of joint configurations, interpolated
smoothly using cubic smooth-step. The RL policy only controls fingers.

Fully vectorized — no Python loops over environments.

Usage:
    traj = ArmTrajectory(config, device)
    phase_ids = traj.get_phase_ids(ep_step)  # [num_envs]
    arm_targets = traj.get_arm_targets(ep_step)  # [num_envs, 7]
    phase_onehot = traj.get_phase_onehot(ep_step)  # [num_envs, 8]
"""

import torch
from enum import IntEnum


class Phase(IntEnum):
    HOVER_ABOVE = 0
    DESCEND_TO_GRASP = 1
    GRASP_HOLD = 2
    LIFT = 3
    TRANSPORT = 4
    DESCEND_TO_PLACE = 5
    RELEASE_HOLD = 6
    RETREAT = 7


# Ordered list of arm joint names (matches right_arm_indices order)
ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class ArmTrajectory:
    """Vectorized scripted arm trajectory controller."""

    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        c = config

        # Build phase boundaries: list of (start_step, end_step)
        # and corresponding start/end joint arrays
        phase_durations = [
            c.steps_hover, c.steps_descend, c.steps_grasp_hold,
            c.steps_lift, c.steps_transport, c.steps_descend_place,
            c.steps_release_hold, c.steps_retreat,
        ]

        # Joint configs for each phase transition (start -> end)
        phase_joint_pairs = [
            (c.arm_hover_joints, c.arm_hover_joints),       # HOVER: hold position
            (c.arm_hover_joints, c.arm_grasp_joints),        # DESCEND: hover -> grasp
            (c.arm_grasp_joints, c.arm_grasp_joints),        # GRASP_HOLD: hold
            (c.arm_grasp_joints, c.arm_lift_joints),         # LIFT: grasp -> lift
            (c.arm_lift_joints, c.arm_transport_joints),     # TRANSPORT: lift -> transport
            (c.arm_transport_joints, c.arm_place_joints),    # DESCEND_TO_PLACE: transport -> place
            (c.arm_place_joints, c.arm_place_joints),        # RELEASE_HOLD: hold
            (c.arm_place_joints, c.arm_lift_joints),         # RETREAT: place -> lift
        ]

        # Convert joint dicts to ordered [7] arrays
        def dict_to_array(d):
            return torch.tensor([d[name] for name in ARM_JOINT_NAMES],
                                dtype=torch.float32, device=device)

        # Build lookup tables
        self.num_phases = 8
        self.phase_starts = []
        self.phase_ends = []
        self.phase_durations = []
        self.start_joints = []  # [8, 7]
        self.end_joints = []    # [8, 7]

        step = 0
        for i, (duration, (start_dict, end_dict)) in enumerate(zip(phase_durations, phase_joint_pairs)):
            self.phase_starts.append(step)
            self.phase_ends.append(step + duration)
            self.phase_durations.append(duration)
            self.start_joints.append(dict_to_array(start_dict))
            self.end_joints.append(dict_to_array(end_dict))
            step += duration

        self.total_steps = step

        # Stack into tensors for vectorized lookup
        self.phase_starts_t = torch.tensor(self.phase_starts, dtype=torch.long, device=device)
        self.phase_ends_t = torch.tensor(self.phase_ends, dtype=torch.long, device=device)
        self.phase_durations_t = torch.tensor(self.phase_durations, dtype=torch.float32, device=device)
        self.start_joints_t = torch.stack(self.start_joints)  # [8, 7]
        self.end_joints_t = torch.stack(self.end_joints)      # [8, 7]

    def get_phase_ids(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return phase ID for each environment. [num_envs] -> [num_envs].

        ep_step: [num_envs] integer tensor of current episode step.
        """
        # Clamp to valid range
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1)

        # For each env, find which phase it's in
        # ep_step >= phase_start AND ep_step < phase_end
        # Shape: [num_envs, 8]
        expanded_step = ep_step_clamped.unsqueeze(1)  # [num_envs, 1]
        starts = self.phase_starts_t.unsqueeze(0)      # [1, 8]
        ends = self.phase_ends_t.unsqueeze(0)           # [1, 8]

        in_phase = (expanded_step >= starts) & (expanded_step < ends)  # [num_envs, 8]

        # If step is past all phases, default to last phase
        phase_ids = in_phase.long().argmax(dim=1)  # [num_envs]

        # Handle case where no phase matches (past end)
        past_end = ep_step >= self.total_steps
        phase_ids[past_end] = 7  # RETREAT

        return phase_ids

    def get_arm_targets(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return interpolated arm joint targets. [num_envs] -> [num_envs, 7]."""
        phase_ids = self.get_phase_ids(ep_step)
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1).float()

        # Gather start/end joints and timing for each env's phase
        start_j = self.start_joints_t[phase_ids]    # [num_envs, 7]
        end_j = self.end_joints_t[phase_ids]          # [num_envs, 7]
        p_start = self.phase_starts_t[phase_ids].float()  # [num_envs]
        p_dur = self.phase_durations_t[phase_ids]          # [num_envs]

        # Compute interpolation factor with smooth-step
        t = ((ep_step_clamped - p_start) / p_dur.clamp(min=1.0)).clamp(0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smooth-step

        # Interpolate
        targets = start_j + t.unsqueeze(1) * (end_j - start_j)
        return targets

    def get_phase_onehot(self, ep_step: torch.Tensor) -> torch.Tensor:
        """Return [num_envs, 8] one-hot phase encoding."""
        phase_ids = self.get_phase_ids(ep_step)
        onehot = torch.zeros(ep_step.shape[0], 8, device=self.device)
        onehot.scatter_(1, phase_ids.unsqueeze(1), 1.0)
        return onehot

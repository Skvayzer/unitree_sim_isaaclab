"""
Parameterized arm trajectory for position-invariant grasping.

Unlike ArmTrajectory which uses fixed joint configs, this class
computes per-environment arm joint targets based on each env's
block position. The shoulder_roll varies with block_y to position
the hand over the block.

The finger policy (System 0) sees identical contact patterns
regardless of block position, because the arm always positions
the hand in the same relative configuration to the block.

Usage:
    traj = ParameterizedArmTrajectory(config, device, num_envs)
    traj.set_block_positions(block_xy)  # called at reset
    arm_targets = traj.get_arm_targets(ep_step)
"""

import torch
from experiments.system0_skills.arm_trajectory import Phase, ARM_JOINT_NAMES


# SR diagnostic data (2026-03-24): fingertip center_y at each SR value
# Measured with diagnostic_sr_sweep.py using full robot reset between configs
# ALL SR values -0.70 to -0.20 are reachable. Center y=-0.180 (SR≈-0.40).
# Valid range: y=-0.232 to y=-0.135 (~10cm).
# Format: (center_y, SR) sorted by center_y descending (most negative first)
_SR_DIAG_CENTER_Y = torch.tensor([
    -0.232, -0.213, -0.194, -0.179, -0.170, -0.159, -0.146, -0.135
])
_SR_DIAG_SR = torch.tensor([
    -0.70, -0.60, -0.50, -0.40, -0.35, -0.30, -0.25, -0.20
])

# Offset: at reference (block_y=-0.180, sr=-0.50), center_y=-0.194
# Fingers are ~0.014 farther from body than the block center
_FINGERTIP_OFFSET = 0.014


class ParameterizedArmTrajectory:
    """Vectorized arm trajectory with per-env block position adaptation."""

    def __init__(self, config, device, num_envs):
        self.config = config
        self.device = device
        self.num_envs = num_envs
        c = config

        self.num_phases = 8
        phase_durations = [
            c.steps_hover, c.steps_descend, c.steps_grasp_hold,
            c.steps_lift, c.steps_transport, c.steps_descend_place,
            c.steps_release_hold, c.steps_retreat,
        ]

        self.phase_starts = []
        self.phase_ends = []
        step = 0
        for d in phase_durations:
            self.phase_starts.append(step)
            self.phase_ends.append(step + d)
            step += d
        self.total_steps = step

        self.phase_starts_t = torch.tensor(self.phase_starts, dtype=torch.long, device=device)
        self.phase_ends_t = torch.tensor(self.phase_ends, dtype=torch.long, device=device)
        self.phase_durations_t = torch.tensor(phase_durations, dtype=torch.float32, device=device)

        # Diagnostic lookup tables on device
        self._diag_cy = _SR_DIAG_CENTER_Y.to(device)
        self._diag_sr = _SR_DIAG_SR.to(device)

        # Base joint configs (from single-block config, sr=-0.5 reference)
        def dict_to_array(d):
            return torch.tensor([d[name] for name in ARM_JOINT_NAMES],
                                dtype=torch.float32, device=device)

        self.base_hover = dict_to_array(c.arm_hover_joints)      # sr=-0.5
        self.base_grasp = dict_to_array(c.arm_grasp_joints)      # sr=-0.5
        self.base_lift = dict_to_array(c.arm_lift_joints)        # sr=-0.5
        self.base_transport = dict_to_array(c.arm_transport_joints)  # sr=-0.2
        self.base_place = dict_to_array(c.arm_place_joints)      # sr=-0.2

        # Target position sr
        self.target_y = c.target_pos[1]
        self.target_sr = self._y_to_sr_scalar(self.target_y)

        # Per-env joint tables: [num_envs, 8, 7]
        self.start_joints = torch.zeros(num_envs, 8, 7, device=device)
        self.end_joints = torch.zeros(num_envs, 8, 7, device=device)

        # Initialize with default positions
        default_y = torch.full((num_envs,), -0.180, device=device)
        self.set_block_positions(default_y)

    def _y_to_sr_scalar(self, y):
        """Piecewise-linear mapping from block_y to SR (scalar version).

        Uses diagnostic data to find the SR that positions fingertips
        at the correct center_y for a given block position.

        Diagnostic validated: full SR range [-0.70, -0.20] works.
        """
        target_center = y - _FINGERTIP_OFFSET
        # Linear interpolation in diagnostic table
        cy = _SR_DIAG_CENTER_Y
        sr = _SR_DIAG_SR
        if target_center <= cy[0]:
            return sr[0].item()  # clamp to most negative
        if target_center >= cy[-1]:
            return sr[-1].item()  # clamp to least negative
        for i in range(len(cy) - 1):
            if cy[i] <= target_center <= cy[i + 1]:
                t = (target_center - cy[i]) / (cy[i + 1] - cy[i])
                return (sr[i] + t * (sr[i + 1] - sr[i])).item()
        return sr[-1].item()

    def _y_to_sr(self, block_y):
        """Piecewise-linear mapping from block_y to SR (vectorized).

        Uses diagnostic data from diagnostic_sr_sweep.py to accurately
        map block positions to shoulder_roll values.

        Previous linear mapping (sr_slope=5/3=1.667) was wrong:
        actual slope varies from ~5.5 at sr=-0.7 to ~4.5 at sr=-0.2.
        This caused 4.6cm positioning error and 0% lift at far positions.
        """
        target_center = block_y - _FINGERTIP_OFFSET
        cy = self._diag_cy  # sorted ascending (most negative first)
        sr = self._diag_sr

        # Vectorized piecewise-linear interpolation
        # Clamp to table range
        clamped = target_center.clamp(cy[0], cy[-1])

        # Find segment index for each value
        # expanded: [N, 1] vs [1, K-1]
        expanded = clamped.unsqueeze(-1)  # [N, 1]
        boundaries = cy[:-1].unsqueeze(0)  # [1, K-1]
        # Each value falls in the rightmost segment where cy[i] <= value
        seg_idx = (expanded >= boundaries).long().sum(dim=-1) - 1  # [N]
        seg_idx = seg_idx.clamp(0, len(cy) - 2)

        cy_lo = cy[seg_idx]
        cy_hi = cy[seg_idx + 1]
        sr_lo = sr[seg_idx]
        sr_hi = sr[seg_idx + 1]

        t = ((clamped - cy_lo) / (cy_hi - cy_lo + 1e-8)).clamp(0, 1)
        return sr_lo + t * (sr_hi - sr_lo)

    def _make_joints(self, base_joints, per_env_sr):
        """Create [num_envs, 7] joint config with per-env shoulder_roll."""
        joints = base_joints.unsqueeze(0).expand(self.num_envs, -1).clone()
        joints[:, 1] = per_env_sr  # index 1 = shoulder_roll
        return joints

    def set_block_positions(self, block_y, env_mask=None):
        """Update arm configs for given block y-positions.

        Args:
            block_y: [num_envs] or [K] y-positions of blocks
            env_mask: optional [num_envs] bool mask for partial updates
        """
        if env_mask is not None:
            pick_sr = self._y_to_sr(block_y)
            target_sr_t = torch.full_like(pick_sr, self.target_sr)

            hover = self._make_joints_partial(self.base_hover, pick_sr, env_mask)
            grasp = self._make_joints_partial(self.base_grasp, pick_sr, env_mask)
            lift_j = self._make_joints_partial(self.base_lift, pick_sr, env_mask)
            transport = self._make_joints_partial(self.base_transport, target_sr_t, env_mask)
            place = self._make_joints_partial(self.base_place, target_sr_t, env_mask)

            # HOVER: hold above block
            self.start_joints[env_mask, 0] = hover
            self.end_joints[env_mask, 0] = hover
            # DESCEND: hover -> grasp
            self.start_joints[env_mask, 1] = hover
            self.end_joints[env_mask, 1] = grasp
            # GRASP_HOLD: hold
            self.start_joints[env_mask, 2] = grasp
            self.end_joints[env_mask, 2] = grasp
            # LIFT: grasp -> lift
            self.start_joints[env_mask, 3] = grasp
            self.end_joints[env_mask, 3] = lift_j
            # TRANSPORT: lift -> transport
            self.start_joints[env_mask, 4] = lift_j
            self.end_joints[env_mask, 4] = transport
            # DESCEND_TO_PLACE: transport -> place
            self.start_joints[env_mask, 5] = transport
            self.end_joints[env_mask, 5] = place
            # RELEASE_HOLD: hold
            self.start_joints[env_mask, 6] = place
            self.end_joints[env_mask, 6] = place
            # RETREAT: place -> lift
            self.start_joints[env_mask, 7] = place
            self.end_joints[env_mask, 7] = lift_j
        else:
            pick_sr = self._y_to_sr(block_y)
            target_sr_t = torch.full_like(pick_sr, self.target_sr)

            hover = self._make_joints(self.base_hover, pick_sr)
            grasp = self._make_joints(self.base_grasp, pick_sr)
            lift_j = self._make_joints(self.base_lift, pick_sr)
            transport = self._make_joints(self.base_transport, target_sr_t)
            place = self._make_joints(self.base_place, target_sr_t)

            self.start_joints[:, 0] = hover
            self.end_joints[:, 0] = hover
            self.start_joints[:, 1] = hover
            self.end_joints[:, 1] = grasp
            self.start_joints[:, 2] = grasp
            self.end_joints[:, 2] = grasp
            self.start_joints[:, 3] = grasp
            self.end_joints[:, 3] = lift_j
            self.start_joints[:, 4] = lift_j
            self.end_joints[:, 4] = transport
            self.start_joints[:, 5] = transport
            self.end_joints[:, 5] = place
            self.start_joints[:, 6] = place
            self.end_joints[:, 6] = place
            self.start_joints[:, 7] = place
            self.end_joints[:, 7] = lift_j

    def _make_joints_partial(self, base_joints, per_env_sr, env_mask):
        """Create joint configs for masked envs only."""
        n = env_mask.sum().item()
        joints = base_joints.unsqueeze(0).expand(n, -1).clone()
        joints[:, 1] = per_env_sr
        return joints

    def get_phase_ids(self, ep_step):
        """Return phase ID for each env. [num_envs] -> [num_envs]."""
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1)
        expanded_step = ep_step_clamped.unsqueeze(1)
        starts = self.phase_starts_t.unsqueeze(0)
        ends = self.phase_ends_t.unsqueeze(0)
        in_phase = (expanded_step >= starts) & (expanded_step < ends)
        phase_ids = in_phase.long().argmax(dim=1)
        past_end = ep_step >= self.total_steps
        phase_ids[past_end] = 7
        return phase_ids

    def get_arm_targets(self, ep_step):
        """Return per-env interpolated arm joint targets. [num_envs] -> [num_envs, 7]."""
        phase_ids = self.get_phase_ids(ep_step)
        ep_step_clamped = ep_step.clamp(0, self.total_steps - 1).float()

        # Gather per-env start/end joints
        env_idx = torch.arange(self.num_envs, device=self.device)
        start_j = self.start_joints[env_idx, phase_ids]  # [num_envs, 7]
        end_j = self.end_joints[env_idx, phase_ids]       # [num_envs, 7]
        p_start = self.phase_starts_t[phase_ids].float()
        p_dur = self.phase_durations_t[phase_ids]

        t = ((ep_step_clamped - p_start) / p_dur.clamp(min=1.0)).clamp(0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smooth-step

        return start_j + t.unsqueeze(1) * (end_j - start_j)

    def get_phase_onehot(self, ep_step):
        """Return [num_envs, 8] one-hot phase encoding."""
        phase_ids = self.get_phase_ids(ep_step)
        onehot = torch.zeros(ep_step.shape[0], 8, device=self.device)
        onehot.scatter_(1, phase_ids.unsqueeze(1), 1.0)
        return onehot

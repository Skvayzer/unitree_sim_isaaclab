"""
Multi-Block Tower Stacking Configuration.

Extends BlockStackConfig for 3-block tower stacking.
Block 0 (red) -> table, Block 1 (yellow) -> on block 0, Block 2 (green) -> on block 1.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from experiments.system0_skills.block_stack_config import BlockStackConfig


@dataclass
class MultiBlockConfig(BlockStackConfig):
    # === 3 block positions (spaced 10cm along y-axis) ===
    # 6cm spacing FAILED: arm/forearm collides with adjacent blocks during trajectory,
    # dropping block 0 grasp from ~80% to ~8%. 10cm spacing verified to recover ~73%.
    # The arm's SR reachable range is only ~10cm (y=-0.217 to y=-0.120), so blocks
    # 1,2 are at the edge/beyond the trained curriculum (±3.5cm). They'll need
    # expanded curriculum or position-invariant checkpoints to grasp reliably.
    num_blocks: int = 3
    block_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.321, -0.180, 0.819),  # Block 0 (red) — center position
        (0.321, -0.280, 0.819),  # Block 1 (yellow) — 10cm in -y (SR clamped to -0.70)
        (0.321, -0.080, 0.819),  # Block 2 (green) — 10cm in +y (SR clamped to -0.20)
    ])
    block_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.0, 0.0),  # red
        (1.0, 0.9, 0.0),  # yellow
        (0.0, 0.8, 0.0),  # green
    ])

    # === Arm grasp configs per block (different shoulder_roll to reach different y) ===
    # NOTE: These are reference configs. The actual SR is computed by
    # ParameterizedArmTrajectory.set_block_positions() using piecewise-linear
    # mapping from block y-position. These are NOT used by the state machine.
    # Block 0 at y=-0.180: SR≈-0.50 (reference)
    # Block 1 at y=-0.280: SR=-0.70 (clamped, beyond diagnostic range)
    # Block 2 at y=-0.080: SR=-0.20 (clamped, beyond diagnostic range)
    arm_grasp_joints_per_block: List[Dict[str, float]] = field(default_factory=lambda: [
        {  # Block 0: y=-0.180
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.5,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        {  # Block 1: y=-0.280 (SR clamped to -0.70)
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.70,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        {  # Block 2: y=-0.080 (SR clamped to -0.20)
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ])

    # Hover configs per block (same sr as grasp to approach correctly)
    # NOTE: NOT used by the state machine / ParameterizedArmTrajectory.
    # Kept for reference only. Hover SP should match single-block (0.0).
    arm_hover_joints_per_block: List[Dict[str, float]] = field(default_factory=lambda: [
        {  # Block 0: y=-0.180
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.5,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        {  # Block 1: y=-0.280 (SR clamped to -0.70)
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.70,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        {  # Block 2: y=-0.080 (SR clamped to -0.20)
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ])

    # === Placement shoulder_pitch per block (controls height) ===
    # Block 0: place at table level, sp=0.0
    # Block 1: place at table + 4cm, sp=-0.05 (slightly higher)
    # Block 2: place at table + 8cm, sp=-0.10 (even higher)
    arm_place_sp_per_block: List[float] = field(default_factory=lambda: [
        0.0,    # Block 0: table level
        -0.05,  # Block 1: ~4cm higher
        -0.10,  # Block 2: ~8cm higher
    ])

    # === Stack heights for reward computation ===
    # block_z for correct placement = table_height + block_size/2 + block_idx * block_size
    # table_height=0.797, block_size=0.04
    # Block 0: 0.797 + 0.02 = 0.817  (≈ block_initial_z)
    # Block 1: 0.817 + 0.04 = 0.857
    # Block 2: 0.857 + 0.04 = 0.897
    stack_heights: List[float] = field(default_factory=lambda: [
        0.819,  # Block 0: table + half block (same as initial z)
        0.859,  # Block 1: on top of block 0
        0.899,  # Block 2: on top of block 1
    ])

    # === Observation ===
    # finger_pos(7) + finger_vel(7) + press_modules(9*12=108) + phase_onehot(8) + block_idx_onehot(3) = 133
    obs_dim: int = 133

    # === Sim: longer episodes for 3 blocks ===
    # 3 cycles * 310 steps = 930 steps
    # 930 * 0.005 * 2 = 9.3s, round to 15.0s for margin
    episode_length_s: float = 15.0

    # Total steps per cycle (inherited: 310)
    steps_per_cycle: int = 310  # same as total_phase_steps

    # === Training ===
    max_iterations: int = 3000
    steps_per_rollout: int = 128  # longer for better coverage of 3-block episodes
    entropy_coeff: float = 0.03  # slightly higher for more exploration in harder task

    # === Reward weights (multi-block) ===
    reward_tower_bonus: float = 100.0  # bonus for all 3 blocks stacked
    tower_xy_tolerance: float = 0.025  # tight XY for 4cm blocks (5cm would cause toppling)
    tower_z_tolerance: float = 0.04    # Z tolerance for tower check

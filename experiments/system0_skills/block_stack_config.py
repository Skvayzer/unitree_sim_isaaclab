"""
System 0 Block Stacking Configuration.

All constants derived from diagnostic_output.txt (2026-03-21).
Separate from config.py which is used by existing grasp/hold skills.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class BlockStackConfig:
    # === Joint indices (from diagnostic) ===
    right_arm_indices: List[int] = field(default_factory=lambda: [12, 16, 20, 22, 24, 26, 28])
    right_hand_indices: List[int] = field(default_factory=lambda: [32, 33, 34, 38, 39, 40, 42])

    # Contact sensor body indices for right hand fingertips
    # Sensor matches .*_hand_.*_link -> 18 bodies total
    # Order: L_cam(0), L_palm(1), R_cam(2), R_palm(3),
    #   L_idx0(4), L_mid0(5), L_thm0(6), R_idx0(7), R_mid0(8), R_thm0(9),
    #   L_idx1(10), L_mid1(11), L_thm1(12), R_idx1(13), R_mid1(14), R_thm1(15),
    #   L_thm2(16), R_thm2(17)
    right_fingertip_contact_indices: List[int] = field(default_factory=lambda: [13, 14, 17])
    # 13 = right_hand_index_1_link, 14 = right_hand_middle_1_link, 17 = right_hand_thumb_2_link

    # All 9 right-hand pressure sensor module bodies (matches hardware PressSensorState_ sequence).
    # Module index → contact sensor body index → link name:
    #   0: R_cam   (2)  — wrist/camera mount area
    #   1: R_palm  (3)  — palm
    #   2: R_idx0  (7)  — index proximal phalanx
    #   3: R_mid0  (8)  — middle proximal phalanx
    #   4: R_thm0  (9)  — thumb proximal phalanx
    #   5: R_idx1  (13) — index distal phalanx
    #   6: R_mid1  (14) — middle distal phalanx
    #   7: R_thm1  (15) — thumb middle phalanx
    #   8: R_thm2  (17) — thumb distal phalanx (tip)
    right_hand_module_contact_indices: List[int] = field(
        default_factory=lambda: [2, 3, 7, 8, 9, 13, 14, 15, 17]
    )
    num_press_modules: int = 9   # 9 modules per hand (matches hardware)
    num_press_readings: int = 12  # 12 pressure cells per module (matches PressSensorState_.pressure)
    # Force scale for normalising net contact force to [0, 1] pressure readings.
    # 10 N covers typical manipulation contact forces comfortably.
    press_force_scale: float = 10.0

    # === Scene geometry (from diagnostic) ===
    robot_pos: Tuple[float, float, float] = (0.0, 0.0, 0.76)
    robot_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Table: kinematic cuboid under the finger workspace
    # Using arm config #0 (el=0): finger tips at z=0.79-0.85 (sweep is mostly horizontal)
    # 50% closure tips: thumb(0.267,-0.165,0.814), index(0.350,-0.174,0.844), middle(0.346,-0.201,0.794)
    # Block center z = avg(0.814,0.844,0.794) = 0.817
    # Table top at z = block_center - half_block = 0.817 - 0.02 = 0.797
    table_height: float = 0.797
    table_size: Tuple[float, float, float] = (0.5, 0.5, 0.797)
    table_pos: Tuple[float, float, float] = (0.32, -0.18, 0.3985)  # center of table

    # Block: 4cm cube on table
    block_size: float = 0.04
    block_mass: float = 0.05  # light enough to grasp
    block_friction: float = 2.0
    # Block at center of 50% closure triangle
    # x_center = (0.267 + 0.350 + 0.346)/3 = 0.321
    # y_center = (-0.165 + -0.174 + -0.201)/3 = -0.180
    block_initial_pos: Tuple[float, float, float] = (0.321, -0.180, 0.819)  # table_height + half_block + 0.002
    block_initial_z: float = 0.819

    # Target placement position (6cm lateral from pick position, arm-verified)
    # At sr=-0.2, sy=0.1, sp=0: tips at (0.321, -0.118, 0.825)
    target_pos: Tuple[float, float, float] = (0.295, -0.152, 0.819)

    # === Arm waypoints (7 joints each: sp, sr, sy, el, wr, wp, wy) ===
    # Use base config #0: sp=0, sr=-0.5, sy=0.1, el=0 (best reach, fingers horizontal)
    # HOVER/GRASP: same config — arm stays at this position while fingers act
    # The block is placed within the 50% closure sweep envelope at this config
    arm_hover_joints: Dict[str, float] = field(default_factory=lambda: {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.5,
        "right_shoulder_yaw_joint": 0.1,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    })

    # ARM_GRASP: same as hover (arm stays still, fingers do the work)
    arm_grasp_joints: Dict[str, float] = field(default_factory=lambda: {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.5,
        "right_shoulder_yaw_joint": 0.1,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    })

    # ARM_LIFT: reduce shoulder_pitch to raise hand (tested: sp=-0.3 raises ~5cm)
    arm_lift_joints: Dict[str, float] = field(default_factory=lambda: {
        "right_shoulder_pitch_joint": -0.3,
        "right_shoulder_roll_joint": -0.5,
        "right_shoulder_yaw_joint": 0.1,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    })

    # ARM_TRANSPORT: lift + shift laterally (sr=-0.2 verified: tips y=-0.090 at transport height)
    arm_transport_joints: Dict[str, float] = field(default_factory=lambda: {
        "right_shoulder_pitch_joint": -0.3,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.1,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    })

    # ARM_PLACE: lower to target position (sr=-0.2 verified: tips at (0.321, -0.118, 0.825))
    arm_place_joints: Dict[str, float] = field(default_factory=lambda: {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.1,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    })

    # === Phase timing (sim steps at control frequency) ===
    steps_hover: int = 30
    steps_descend: int = 30
    steps_grasp_hold: int = 50
    steps_lift: int = 40
    steps_transport: int = 50
    steps_descend_place: int = 40
    steps_release_hold: int = 40
    steps_retreat: int = 30

    @property
    def total_phase_steps(self) -> int:
        return (self.steps_hover + self.steps_descend + self.steps_grasp_hold +
                self.steps_lift + self.steps_transport + self.steps_descend_place +
                self.steps_release_hold + self.steps_retreat)

    # === Training ===
    num_envs: int = 512
    env_spacing: float = 5.0
    max_iterations: int = 5000
    steps_per_rollout: int = 64
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.03  # higher for exploration
    value_coeff: float = 1.0
    ppo_epochs: int = 5
    mini_batches: int = 4
    max_grad_norm: float = 1.0

    # === Action ===
    # Scale must be large enough to cover full finger range
    # Index: [0.079, 1.492] range=1.41, from default 0.1 need delta=1.39 for full close
    # With scale=1.5, action=+1 gives target = default + 1.5 = 1.6 (clamped to 1.492)
    action_scale: float = 1.5
    action_ema_alpha: float = 0.7
    action_dim: int = 7

    # === Observation ===
    # finger_pos(7) + finger_vel(7) + press_modules(9*12=108) + phase_onehot(8) = 130
    obs_dim: int = 130

    # === Reward weights ===
    # Scaled down to prevent value function explosion (returns were hitting 400+)
    reward_block_lifted: float = 3.0
    reward_block_placed: float = 30.0
    reward_contact_during_grasp: float = 0.3
    reward_approach_target: float = 5.0   # dense approach reward during placement
    reward_release: float = 5.0           # finger opening reward during release
    penalty_block_dropped: float = 2.0
    penalty_block_knocked: float = 3.0
    penalty_action_magnitude: float = 0.002
    reward_hold_during_transport: float = 0.05

    # === Sim ===
    sim_dt: float = 0.005
    decimation: int = 2
    episode_length_s: float = 5.0  # ~310 steps for 1 block cycle

    # === Arm actuator settings ===
    # Very high to track scripted targets quickly (arm is heavy)
    arm_stiffness: float = 5000.0
    arm_damping: float = 500.0
    hand_stiffness: float = 100.0
    hand_damping: float = 10.0

    # === Policy network ===
    hidden_dim: int = 128

    # === WandB ===
    wandb_project: str = "System0_MoE"
    wandb_entity: str = "skvayzer"

"""
System 0 Grasp Skill Configuration.

All constants derived from diagnostic output of test_env.py.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class System0Config:
    # === Joint indices (from test_env.py diagnostic) ===
    right_hand_indices: List[int] = field(default_factory=lambda: [32, 33, 34, 38, 39, 40, 42])
    right_arm_indices: List[int] = field(default_factory=lambda: [12, 16, 20, 22, 24, 26, 28])

    # Contact sensor indices for right hand fingertips.
    # Verified against BlockStackConfig — matches contact_sensor body ordering.
    #   index 13 = right_hand_index_1_link (index tip)
    #   index 14 = right_hand_middle_1_link (middle tip)
    #   index 17 = right_hand_thumb_2_link (thumb tip)
    right_fingertip_contact_indices: List[int] = field(default_factory=lambda: [13, 14, 17])

    # === Scene positions ===
    robot_pos: Tuple[float, float, float] = (0.0, 0.0, 0.76)
    robot_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    block_pos: Tuple[float, float, float] = (0.25, -0.15, 0.85)  # closer to palm
    block_initial_z: float = 0.85

    # === Environment ===
    num_envs: int = 512
    env_spacing: float = 3.0
    episode_length_s: float = 10.0  # 200 steps at dt=0.005 * decimation=5 => 200*0.025=5s
    decimation: int = 2
    sim_dt: float = 0.005

    # === Action ===
    action_scale: float = 0.3  # max rad per step — large for visible finger motion
    action_dim: int = 7

    # === Observation ===
    obs_dim: int = 21  # 7 + 7 + 3 + 3 + 1

    # === Reward weights ===
    target_force: float = 1.0  # N, target contact force per finger
    grasp_reward_weight: float = 1.0
    force_penalty_weight: float = 0.05
    drop_penalty_weight: float = 5.0
    smooth_penalty_weight: float = 0.01

    # === PPO hyperparameters ===
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 1.0
    max_grad_norm: float = 1.0
    ppo_epochs: int = 5
    mini_batches: int = 4
    steps_per_rollout: int = 24
    max_iterations: int = 5000

    # === Policy network ===
    hidden_dim: int = 128

    # === WandB ===
    wandb_project: str = "System0_MoE"
    wandb_entity: str = "skvayzer"

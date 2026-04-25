"""Hyperparameters for System 0 MoE standalone RL training."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Environment
    task_id: str = "Isaac-Stack-RgyBlock-G129-Dex3-Joint"
    num_envs: int = 1
    headless: bool = True
    sim_dt: float = 0.005  # 200 Hz sim
    control_dt: float = 0.01  # 100 Hz control

    # System 0 MoE
    joint_dim: int = 28
    vel_dim: int = 28
    tactile_dim: int = 18  # 6 fingertips x 3 axes in sim
    torque_dim: int = 28
    target_dim: int = 28
    intent_dim: int = 128  # scripted controller phase encoding
    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2
    feedback_dim: int = 64


    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 2048  # steps per rollout before PPO update

    # Training
    total_timesteps: int = 2_000_000
    log_interval: int = 10  # PPO updates between logs
    save_interval: int = 50  # PPO updates between saves
    checkpoint_dir: str = "experiments/system0_rl/checkpoints"

    # Scripted controller
    approach_dist: float = 0.13  # m, close enough to start pre-grasp (includes height offset)
    grasp_force_threshold: float = 0.3  # N, contact force for successful grasp
    lift_height: float = 0.15  # m above table
    block_height: float = 0.05  # m
    stacking_order: list[str] = field(
        default_factory=lambda: ["red_block", "yellow_block", "green_block"]
    )

    # Gains
    kp_base_arm: float = 80.0
    kp_base_finger: float = 1.5
    kd_base_arm: float = 5.0
    kd_base_finger: float = 0.1

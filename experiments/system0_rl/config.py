"""Hyperparameters for System 0 blind-grasping RL training (BlockStackEnvCfg)."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ── Environment ────────────────────────────────────────────────────────
    num_envs: int = 512
    headless: bool = True
    sim_dt: float = 0.005   # 200 Hz sim
    control_dt: float = 0.01  # 100 Hz control

    # ── Observation dims ───────────────────────────────────────────────────
    # Arm: 5 controllable joints (shoulder_pitch/roll, elbow, wrist_roll/pitch).
    arm_dim: int = 5      # right arm pos (and separately vel)
    # Right hand: 7 finger joints.
    joint_dim: int = 7    # right finger qpos
    vel_dim: int = 7      # right finger qvel
    torque_dim: int = 7   # right finger applied torques
    # Extended tactile: 4 channels × 18 pads = 72.
    tactile_dim: int = 72
    # Coarse target fed to MoE as context — matches 12-DOF action space.
    target_dim: int = 12
    # Privileged dims for critic ONLY (never seen by actor):
    # block_xyz(3)+block_to_palm_vec(3)+block_to_thumb_vec(3)+block_vel(3)+block_quat(4)
    # +contact_bool(5)+friction(1)+stage_onehot(4) = 26
    priv_dim: int = 26

    # ── MoE architecture ──────────────────────────────────────────────────
    intent_dim: int = 128  # curriculum-stage encoding (one-hot @ [:4], rest zero)
    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2
    feedback_dim: int = 64

    # ── Action ────────────────────────────────────────────────────────────
    # Action = tanh(policy_raw) * delta_max  (offset from default hand pose).
    # BlockStackEnvCfg action_manager applies scale=1.5, use_default_offset=True,
    # so the effective joint displacement is delta_max * 1.5 radians max.
    delta_max: float = 0.5

    # ── Exploration: Ornstein-Uhlenbeck noise ─────────────────────────────
    ou_theta: float = 0.15   # mean-reversion rate
    ou_sigma: float = 0.10   # noise scale

    # ── Curriculum (block XY offset beyond env's built-in ±8 mm) ─────────
    # Stages: 0=fixed, 1=±2 cm, 2=±5 cm, 3=±5 cm + DR
    curriculum_xy_ranges: tuple = (0.00, 0.02, 0.05, 0.05)
    curriculum_success_threshold: float = 0.75  # lift rate to advance
    curriculum_min_episodes: int = 1000         # min episodes before advancing

    # ── PPO ───────────────────────────────────────────────────────────────
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.005  # raised 5× from 0.001; 0.001 cannot prevent/recover std collapse
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 512
    rollout_steps: int = 256  # steps per rollout per env before PPO update

    # ── Training schedule ─────────────────────────────────────────────────
    total_timesteps: int = 20_000_000  # was 10M; bumped to allow continuing from 10M checkpoint
    log_interval: int = 10   # PPO updates between logs
    save_interval: int = 5   # PPO updates between saves (~2.6M steps at 2048 envs)
    checkpoint_dir: str = "experiments/system0_rl/checkpoints"

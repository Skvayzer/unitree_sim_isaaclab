"""
System 0 MoE policy with PPO wrapper for standalone RL training.

STANDALONE: no lerobot/grootCoT imports. Self-contained for Isaac Lab.

Classes
-------
System0PPOWrapper  — 8-expert MoE (current training run, legacy)
RLSystem0Policy    — 4-expert flat MoE (Phase 3 spec, SYSTEM0_FACTS.md)

RLSystem0Policy observation layout (92D):
    tactile_ext(64) | right_torques(7) | right_qpos(7) | left_torques(7) | left_qpos(7)
Intent (128D): one-hot curriculum stage [:4], rest zero.
Gate input (220D): obs + intent concatenated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class System0Config:
    # Arm: 5 controllable joints (shoulder_pitch/roll, elbow, wrist_roll/pitch)
    arm_dim: int = 5
    # Fingers: 7 right-hand joints
    joint_dim: int = 7
    vel_dim: int = 7
    tactile_dim: int = 72
    torque_dim: int = 7
    # target_dim matches action space: 5 arm + 7 fingers = 12
    target_dim: int = 12
    intent_dim: int = 128
    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2
    # action_dim = 5 arm + 7 fingers = 12
    action_dim: int = 12
    feedback_dim: int = 64
    priv_dim: int = 26   # privileged dims for critic only (block state + contacts); 0 = symmetric

    @property
    def input_dim(self) -> int:
        # actor: arm_pos(5)+arm_vel(5)+finger_pos(7)+finger_vel(7)+tactile(72)+torques(7)+targets(12)+intent(128)
        return (self.arm_dim + self.arm_dim +        # arm pos + arm vel = 10
                self.joint_dim + self.vel_dim +       # finger pos + vel = 14
                self.tactile_dim +                    # 72
                self.torque_dim +                     # 7
                self.target_dim +                     # 12
                self.intent_dim)                      # 128  → total 243

    @property
    def critic_input_dim(self) -> int:
        # critic: actor_obs_with_targets(115) + priv(26) + intent(128) = 269
        return self.input_dim + self.priv_dim


class MoEFFN(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int, top_k: int, intent_dim: int):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(n_experts)
        ])
        router_dim = hidden_dim + intent_dim if intent_dim > 0 else hidden_dim
        self.router = nn.Linear(router_dim, n_experts)
        self.has_intent = intent_dim > 0

    def forward(self, x, intent=None):
        if self.has_intent and intent is not None:
            router_input = torch.cat([x, intent], dim=-1)
        else:
            router_input = x
        logits = self.router(router_input)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k)
        # Ensure 2D even for batch_size=1
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
            indices = indices.unsqueeze(0)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            output = output.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        for k in range(self.top_k):
            for e_idx in range(self.n_experts):
                mask = indices[:, k] == e_idx
                if mask.any():
                    output[mask] += weights[mask, k:k+1] * self.experts[e_idx](x[mask])

        if squeeze_back:
            output = output.squeeze(0)
        return output


class System0MoEActor(nn.Module):
    """Actor network: obs + intent → delta_q (deterministic mean)."""

    def __init__(self, cfg: System0Config):
        super().__init__()
        self.cfg = cfg
        self.input_encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.moe1 = MoEFFN(cfg.hidden_dim, cfg.n_experts, cfg.top_k, cfg.intent_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.moe2 = MoEFFN(cfg.hidden_dim, cfg.n_experts, cfg.top_k, cfg.intent_dim)
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim) - 1.0)

    def forward(self, obs, intent=None):
        # Privileged-leak guard: actor must never receive more dims than obs_with_targets.
        _expected = self.cfg.input_dim - self.cfg.intent_dim  # 115 = 103+12
        assert obs.shape[-1] == _expected, (
            f"Actor received {obs.shape[-1]}D, expected {_expected}D — privileged leak?")

        if intent is None and self.cfg.intent_dim > 0:
            intent = torch.zeros(obs.shape[0], self.cfg.intent_dim, device=obs.device)
        full_input = torch.cat([obs, intent], dim=-1)

        x = self.input_encoder(full_input)
        x = x + self.moe1(self.norm1(x), intent)
        x = x + self.moe2(self.norm2(x), intent)
        mean = self.mean_head(x)
        return mean

    def get_distribution(self, obs, intent=None):
        mean = self.forward(obs, intent)
        # Clamp to prevent NaN from propagating through the distribution
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        mean = mean.clamp(-5.0, 5.0)
        std = self.log_std.exp().clamp(min=1e-6, max=0.3).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class System0Critic(nn.Module):
    """Value function for PPO. Accepts optional privileged obs (critic-only at train time)."""

    def __init__(self, cfg: System0Config):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.critic_input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, obs, intent=None, priv_obs=None):
        N = obs.shape[0]
        if intent is None:
            intent = torch.zeros(N, self.cfg.intent_dim, device=obs.device)
        if priv_obs is None:
            priv_obs = torch.zeros(N, self.cfg.priv_dim, device=obs.device)
        # obs = actor_obs_with_targets (N, 115); layout: [obs | priv | intent] → (N, 269)
        full_input = torch.cat([obs, priv_obs, intent], dim=-1)
        return self.net(full_input).squeeze(-1)


class System0PPOWrapper(nn.Module):
    """Combined actor-critic for PPO training."""

    def __init__(self, cfg: System0Config):
        super().__init__()
        self.actor = System0MoEActor(cfg)
        self.critic = System0Critic(cfg)
        self.cfg = cfg

    def act(self, obs, intent=None, deterministic=False, priv_obs=None):
        """Sample action for environment interaction."""
        dist = self.actor.get_distribution(obs, intent)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs, intent, priv_obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, intent, actions, priv_obs=None):
        """Evaluate actions for PPO update."""
        dist = self.actor.get_distribution(obs, intent)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs, intent, priv_obs)
        return log_prob, entropy, value


# ── Phase 3: RLSystem0Policy ──────────────────────────────────────────────────
# Per SYSTEM0_FACTS.md scope decision:
#   4 experts, top-2, each expert Linear(220,256)→ReLU→Linear(256,action_dim).
#   Obs layout differs from System0PPOWrapper: includes left hand, drops qvel/targets.


class _ExpertMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLSystem0Actor(nn.Module):
    """
    Flat sparse-MoE actor.

    Input:   obs(obs_dim) + intent(intent_dim) → x(gate_dim)
    Gate:    Linear(gate_dim, n_experts) with soft top-k routing
    Experts: each _ExpertMLP(gate_dim, hidden_dim, action_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        intent_dim: int,
        hidden_dim: int,
        n_experts: int,
        top_k: int,
        action_dim: int,
    ):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        gate_dim = obs_dim + intent_dim
        self.gate = nn.Linear(gate_dim, n_experts)
        self.experts = nn.ModuleList([
            _ExpertMLP(gate_dim, hidden_dim, action_dim) for _ in range(n_experts)
        ])
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self._action_dim = action_dim

    def _moe_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_w = F.softmax(self.gate(x), dim=-1)
        top_w, top_idx = torch.topk(gate_w, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        out = torch.zeros(x.shape[0], self._action_dim, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    out[mask] += top_w[mask, k:k+1] * self.experts[e](x[mask])
        return out

    def forward(self, obs: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, intent], dim=-1)
        mean = self._moe_forward(x)
        return torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-5.0, 5.0)

    def get_distribution(
        self, obs: torch.Tensor, intent: torch.Tensor
    ) -> torch.distributions.Normal:
        mean = self.forward(obs, intent)
        std = self.log_std.exp().clamp(min=1e-6, max=0.3).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class RLSystem0Critic(nn.Module):
    """Value network: (obs + intent) → scalar."""

    def __init__(self, obs_dim: int, intent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + intent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, intent], dim=-1)).squeeze(-1)


class RLSystem0Policy(nn.Module):
    """
    PPO actor-critic for System 0 blind tactile grasping (Phase 3 spec).

    Observation layout (92D):
        tactile_ext(64) | right_torques(7) | right_qpos(7) | left_torques(7) | left_qpos(7)
    Intent (128D): one-hot curriculum stage ([:4]) + zeros.
    Gate input (220D): obs(92) + intent(128).

    Architecture (SYSTEM0_FACTS.md Phase 3):
        n_experts=4, top_k=2
        Each expert: Linear(220, 256) → ReLU → Linear(256, action_dim)
        action_dim=7 for standalone RL (right hand only).
        Set action_dim=14 for CraftNet integration (both hands).

    To wire into train.py, use build_rl_system0_obs() (see train.py TODO) instead
    of build_obs_batch() so the obs layout matches this policy's expectations.
    """

    OBS_DIM    = 92
    INTENT_DIM = 128
    HIDDEN_DIM = 256
    N_EXPERTS  = 4
    TOP_K      = 2
    ACTION_DIM = 7

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        intent_dim: int = INTENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_experts:  int = N_EXPERTS,
        top_k:      int = TOP_K,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.actor  = RLSystem0Actor(obs_dim, intent_dim, hidden_dim, n_experts, top_k, action_dim)
        self.critic = RLSystem0Critic(obs_dim, intent_dim, hidden_dim)

    def act(
        self,
        obs: torch.Tensor,
        intent: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple:
        """Sample action for rollout collection. Returns (action, log_prob, value)."""
        dist = self.actor.get_distribution(obs, intent)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs, intent)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        intent: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple:
        """Evaluate stored actions for PPO update. Returns (log_prob, entropy, value)."""
        dist = self.actor.get_distribution(obs, intent)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(obs, intent)
        return log_prob, entropy, value

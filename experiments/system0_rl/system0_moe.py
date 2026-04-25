"""
System 0 MoE policy with PPO wrapper for standalone RL training.

STANDALONE: no lerobot/grootCoT imports. Self-contained for Isaac Lab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class System0Config:
    joint_dim: int = 28
    vel_dim: int = 28
    tactile_dim: int = 18  # 6 fingertips x 3 axes in sim
    torque_dim: int = 28
    target_dim: int = 28
    intent_dim: int = 128
    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2
    action_dim: int = 28
    feedback_dim: int = 64

    @property
    def input_dim(self) -> int:
        return (self.joint_dim + self.vel_dim + self.tactile_dim +
                self.torque_dim + self.target_dim + self.intent_dim)


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
        if intent is None and self.cfg.intent_dim > 0:
            intent = torch.zeros(obs.shape[0], self.cfg.intent_dim, device=obs.device)
        if self.cfg.intent_dim > 0 and obs.shape[-1] < self.cfg.input_dim:
            full_input = torch.cat([obs, intent], dim=-1)
        else:
            full_input = obs

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
        std = self.log_std.exp().clamp(min=1e-6, max=2.0).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class System0Critic(nn.Module):
    """Value function for PPO."""

    def __init__(self, cfg: System0Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.cfg = cfg

    def forward(self, obs, intent=None):
        if intent is None and self.cfg.intent_dim > 0:
            intent = torch.zeros(obs.shape[0], self.cfg.intent_dim, device=obs.device)
        if self.cfg.intent_dim > 0 and obs.shape[-1] < self.cfg.input_dim:
            full_input = torch.cat([obs, intent], dim=-1)
        else:
            full_input = obs
        return self.net(full_input).squeeze(-1)


class System0PPOWrapper(nn.Module):
    """Combined actor-critic for PPO training."""

    def __init__(self, cfg: System0Config):
        super().__init__()
        self.actor = System0MoEActor(cfg)
        self.critic = System0Critic(cfg)
        self.cfg = cfg

    def act(self, obs, intent=None, deterministic=False):
        """Sample action for environment interaction."""
        dist = self.actor.get_distribution(obs, intent)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs, intent)
        return action, log_prob, value

    def evaluate_actions(self, obs, intent, actions):
        """Evaluate actions for PPO update."""
        dist = self.actor.get_distribution(obs, intent)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs, intent)
        return log_prob, entropy, value

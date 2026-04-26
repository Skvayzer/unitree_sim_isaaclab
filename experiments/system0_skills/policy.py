"""
Simple MLP actor-critic for System 0 skills.

Small (~50K parameters), fast, easy to debug.
Shared architecture across all 3 skills (grasp, hold, release).
"""

import torch
import torch.nn as nn


class System0Actor(nn.Module):
    """MLP Gaussian policy: obs -> (mean, std) for each action dimension."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, action_dim),
        )
        # Small initial weights for stable start (from DexPBT paper)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Learnable log standard deviation, initialized to exp(-1) ~ 0.37
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def forward(self, obs: torch.Tensor):
        mean = self.net(obs)
        std = self.log_std.clamp(-2.0, 0.5).exp().expand_as(mean)
        return mean, std


class System0Critic(nn.Module):
    """MLP value function: obs -> scalar value."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class System0AsymmetricCritic(nn.Module):
    """Privileged critic for asymmetric actor-critic.

    Sees actor obs (28D) + privileged info (18D) = 46D.
    Larger network (256x256) since it has more information to process.
    Discarded at deployment — only used during sim training.

    Privileged obs:
        block_pos_relative_to_palm(3)  — where is block relative to hand?
        block_quat(4)                   — block orientation
        block_linear_vel(3)             — is block slipping/moving?
        grip_force_per_finger(3)        — detailed force per fingertip
        distance_to_target(3)           — vector from block to target
        friction_coefficient(1)         — surface properties
        block_mass(1)                   — object mass
    """

    PRIVILEGED_DIM = 18

    def __init__(self, actor_obs_dim: int, hidden: int = 256):
        super().__init__()
        total_dim = actor_obs_dim + self.PRIVILEGED_DIM
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, actor_obs: torch.Tensor, privileged_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([actor_obs, privileged_obs], dim=-1)
        return self.net(combined).squeeze(-1)

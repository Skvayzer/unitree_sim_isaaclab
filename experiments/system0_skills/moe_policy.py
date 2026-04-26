"""
System 0 MoE Policy: Combines 3 trained skill experts with a trainable router.

Architecture:
    obs(28D) → expert_0(obs_padded) → action_0  (grasp: uses dims 0:21)
    obs(28D) → expert_1(obs)        → action_1  (hold:  uses all 28D)
    obs(28D) → expert_2(obs_padded) → action_2  (release: uses dims 0:22)
    obs(28D) → router              → [w_0, w_1, w_2]
    action = sum(w_i * action_i)

Each expert is a COMPLETE MLP (obs→hidden→hidden→action), loaded from
trained skill checkpoints. Experts are FROZEN — only the router is trained.
The router takes raw unified obs and learns to select experts based on
physical context (contact forces, target_force, arm velocity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullExpert(nn.Module):
    """Complete obs→action MLP (same arch as System0Actor)."""

    def __init__(self, obs_dim: int, action_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, action_dim),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, unified_dim=28). Expert uses only its own obs_dim."""
        return self.net(obs[:, :self.obs_dim])


class System0MoEActor(nn.Module):
    """MoE actor with 3 frozen full-MLP experts and a trainable router.

    Each expert takes the unified obs (28D) but only uses its relevant dims.
    The router takes all 28D and learns to select the right expert.
    """

    UNIFIED_OBS_DIM = 28
    ACTION_DIM = 7
    N_EXPERTS = 3

    # Skill obs dims
    SKILL_OBS_DIMS = [28, 28, 28]  # grasp, hold, release — all use BlockStackConfig obs_dim

    def __init__(self, hidden: int = 128, temperature: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature

        # 3 full experts with different input dims
        self.experts = nn.ModuleList([
            FullExpert(obs_dim=self.SKILL_OBS_DIMS[i], action_dim=self.ACTION_DIM, hidden=hidden)
            for i in range(self.N_EXPERTS)
        ])

        # Router: raw obs → expert weights
        self.router = nn.Sequential(
            nn.Linear(self.UNIFIED_OBS_DIM, 64),
            nn.ELU(),
            nn.Linear(64, self.N_EXPERTS),
        )

        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(self.ACTION_DIM) - 1.0)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, 28) unified observation

        Returns:
            mean: (B, 7) action mean
            std: (B, 7) action std
        """
        # Router weights from physical context (sharp routing via temperature)
        logits = self.router(obs)
        weights = F.softmax(logits / self.temperature, dim=-1)  # (B, 3)

        # Expert outputs
        expert_outputs = torch.stack(
            [expert(obs) for expert in self.experts], dim=1
        )  # (B, 3, 7)

        # Weighted combination
        mean = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, 7)
        std = self.log_std.exp().expand_as(mean)

        return mean, std

    def get_router_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Return router weights for logging/analysis."""
        with torch.no_grad():
            logits = self.router(obs)
            return F.softmax(logits / self.temperature, dim=-1)

    def load_skill_experts(self, grasp_ckpt: str, hold_ckpt: str, release_ckpt: str,
                           device: torch.device):
        """Load trained skill actor weights into the 3 experts.

        Each skill checkpoint has actor state_dict with keys:
        net.0.weight, net.0.bias, net.2.weight, net.2.bias, net.4.weight, net.4.bias, log_std
        """
        ckpt_paths = [grasp_ckpt, hold_ckpt, release_ckpt]
        skill_names = ["grasp", "hold", "release"]

        for i, (path, name) in enumerate(zip(ckpt_paths, skill_names)):
            ckpt = torch.load(path, map_location=device, weights_only=True)

            # Get actor state dict
            if "model_state_dict" in ckpt:
                # RSL-RL format
                sd = ckpt["model_state_dict"]
                actor_sd = {}
                for k, v in sd.items():
                    if k.startswith("actor."):
                        actor_sd[k.replace("actor.", "net.", 1)] = v
            elif "actor" in ckpt:
                actor_sd = ckpt["actor"]
            else:
                print(f"[WARN] Unknown format for {name}: {list(ckpt.keys())}")
                continue

            # Load into expert — direct mapping since architecture matches
            expert = self.experts[i]
            expert_sd = expert.state_dict()
            loaded = 0
            for k in expert_sd:
                if k in actor_sd:
                    src = actor_sd[k]
                    dst = expert_sd[k]
                    if src.shape == dst.shape:
                        expert_sd[k] = src
                        loaded += 1
                    else:
                        print(f"  [{name}] SKIP {k}: shape {src.shape} vs {dst.shape}")
            expert.load_state_dict(expert_sd)
            print(f"  [{name}] Loaded {loaded}/{len(expert_sd)} params into expert {i}")

    def freeze_experts(self):
        """Freeze all expert parameters. Only router + log_std remain trainable."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad_(False)
        n_frozen = sum(p.numel() for p in self.experts.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen {n_frozen} expert params, {n_trainable} trainable params remain")


class System0MoECritic(nn.Module):
    """Value function for MoE training. Fresh network."""

    def __init__(self, obs_dim: int = 28, hidden: int = 128):
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

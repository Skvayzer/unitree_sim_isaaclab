"""
ActionProvider that wraps trained System 0 MoE + scripted controller
for use with Isaac Lab's sim_main.py evaluation pipeline.

Usage:
    python sim_main.py --action_source custom \
        --custom_provider experiments.system0_rl.system0_action_provider.System0ActionProvider \
        --system0_checkpoint experiments/system0_rl/checkpoints/final.pt
"""

import torch
from typing import Optional

try:
    from action_provider.action_base import ActionProvider
except ImportError:
    class ActionProvider:
        def __init__(self, name): self.name = name
        def get_action(self, env): return None
        def start(self): pass
        def stop(self): pass
        def cleanup(self): pass

from .config import TrainConfig
from .system0_moe import System0Config, System0PPOWrapper
from .scripted_controller import ScriptedController
from .train import build_obs, map_to_sim


class System0ActionProvider(ActionProvider):
    """Deploy trained System 0 MoE in Isaac Lab."""

    def __init__(self, checkpoint_path: str, device: str = "cuda:0", name: str = "system0_moe"):
        super().__init__(name)
        self.device = torch.device(device)
        self.config = TrainConfig()

        # Load policy
        moe_cfg = System0Config(
            tactile_dim=self.config.tactile_dim,
            intent_dim=self.config.intent_dim,
        )
        self.policy = System0PPOWrapper(moe_cfg).to(self.device)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(ckpt["policy"])
            print(f"[System0] Loaded checkpoint: {checkpoint_path}")

        self.policy.eval()
        self.controller = ScriptedController(self.config, device=self.device)

    def get_action(self, env) -> Optional[torch.Tensor]:
        obs = build_obs(env, self.device)
        coarse_targets, phase_intent = self.controller.step(env)
        coarse_targets = coarse_targets.to(self.device)
        phase_intent = phase_intent.to(self.device)

        obs_with_targets = torch.cat([obs, coarse_targets])

        with torch.no_grad():
            delta_q, _, _ = self.policy.act(obs_with_targets, phase_intent, deterministic=True)

        sim_action = map_to_sim(coarse_targets, delta_q, env, self.device)
        return sim_action.squeeze(0)  # (43,)

    def start(self):
        self.controller.reset()

    def cleanup(self):
        pass

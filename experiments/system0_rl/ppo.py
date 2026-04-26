"""
PPO implementation for System 0 MoE training.

RolloutBuffer + GAE advantage computation + clipped PPO update.

GAE fix (2026-04-26): Buffer now stores (T, N, ...) layout so advantages are
computed per-env independently. The old flat (T*N,) layout interleaved envs,
causing done signals and bootstrap values from env_i to corrupt the advantage
estimate for env_j — fatal for sparse-reward problems like tactile grasping.
"""

import torch


class RolloutBuffer:
    """Stores rollout data with (T, N, ...) layout for correct per-env GAE."""

    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int,
                 intent_dim: int, action_dim: int, device="cpu"):
        T, N = rollout_steps, num_envs
        self.T = T
        self.N = N
        self.device = device
        self.pos = 0  # current step index (0 .. T-1)

        self.observations = torch.zeros(T, N, obs_dim,    device=device)
        self.intents      = torch.zeros(T, N, intent_dim, device=device)
        self.actions      = torch.zeros(T, N, action_dim, device=device)
        self.log_probs    = torch.zeros(T, N,             device=device)
        self.rewards      = torch.zeros(T, N,             device=device)
        self.dones        = torch.zeros(T, N,             device=device)
        self.values       = torch.zeros(T, N,             device=device)
        self.advantages   = torch.zeros(T, N,             device=device)
        self.returns      = torch.zeros(T, N,             device=device)

    def add_step(self, obs: torch.Tensor, intent: torch.Tensor,
                 actions: torch.Tensor, log_probs: torch.Tensor,
                 rewards: torch.Tensor, dones: torch.Tensor,
                 values: torch.Tensor) -> None:
        """Store one full step for all envs. All inputs shape (N, ...)."""
        t = self.pos
        self.observations[t] = obs
        self.intents[t]      = intent
        self.actions[t]      = actions
        self.log_probs[t]    = log_probs
        self.rewards[t]      = rewards
        self.dones[t]        = dones
        self.values[t]       = values
        self.pos += 1

    def compute_advantages(self, last_values: torch.Tensor,
                           last_dones: torch.Tensor,
                           gamma: float = 0.99,
                           gae_lambda: float = 0.95) -> None:
        """Vectorized per-env GAE — no cross-env contamination.

        last_values: (N,) bootstrap values for the state AFTER the rollout
        last_dones:  (N,) bool/float — 1 if env just terminated at end of rollout
        """
        T = self.pos  # steps actually filled
        last_gae = torch.zeros(self.N, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values       = last_values
                next_nonterminal  = 1.0 - last_dones.float()
            else:
                next_values       = self.values[t + 1]         # (N,) same-env next step
                next_nonterminal  = 1.0 - self.dones[t + 1].float()

            delta    = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

        # Normalize advantages across the whole (T, N) batch
        adv_flat = self.advantages[:T].reshape(-1)
        self.advantages[:T] = (
            (self.advantages[:T] - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        )

    def get_batches(self, minibatch_size: int):
        """Flatten (T, N, ...) → (T*N, ...) and yield random minibatches."""
        T, N = self.pos, self.N
        total = T * N

        obs_flat       = self.observations[:T].reshape(total, -1)
        intents_flat   = self.intents[:T].reshape(total, -1)
        actions_flat   = self.actions[:T].reshape(total, -1)
        log_probs_flat = self.log_probs[:T].reshape(total)
        adv_flat       = self.advantages[:T].reshape(total)
        returns_flat   = self.returns[:T].reshape(total)

        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]
            yield (
                obs_flat[idx],
                intents_flat[idx],
                actions_flat[idx],
                log_probs_flat[idx],
                adv_flat[idx],
                returns_flat[idx],
            )

    def reset(self) -> None:
        self.pos = 0


def ppo_update(policy, optimizer, buffer, config):
    """Run PPO update on collected rollout data. Returns dict with metrics."""
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates         = 0

    for _ in range(config.ppo_epochs):
        for obs, intent, actions, old_log_probs, advantages, returns in \
                buffer.get_batches(config.minibatch_size):

            log_probs, entropy, values = policy.evaluate_actions(obs, intent, actions)

            ratio  = (log_probs - old_log_probs).exp()
            surr1  = ratio * advantages
            surr2  = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss  = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = (policy_loss
                    + config.value_coeff   * value_loss
                    + config.entropy_coeff * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += (-entropy_loss).item()
            n_updates         += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss":  total_value_loss  / max(n_updates, 1),
        "entropy":     total_entropy     / max(n_updates, 1),
    }

# Adaptation Module Design for System 0

**Date**: 2026-03-23
**Status**: Design document (not implemented)
**Dependencies**: Domain randomization must be implemented first. Specialists must work with DR.
**Reference**: Qi et al. "In-Hand Object Rotation via Rapid Motor Adaptation" (CoRL 2022); MoDE-VLA (Tang et al. 2026)

---

## Objective

Replace privileged simulator state (block position, velocity, contact normals) with a learned latent representation predicted from proprioceptive history. This enables deployment on real hardware where ground-truth object state is unavailable.

---

## Architecture Overview

```
Training Phase 1 (PPO with asymmetric critic):
  Actor:  [28D obs + 32D privileged_latent] = 60D -> MLP 128x128 -> 7D actions
  Critic: [28D obs + 18D raw_privileged]     = 46D -> MLP 256x256 -> 1D value

Training Phase 2 (Adaptation module, actor frozen):
  PrivilegedEncoder: MLP(18D sim_state -> 64 -> 32D latent)  [FROZEN from Phase 1]
  AdaptationModule:  LSTM(proprio_history T x 17D -> 32D predicted_latent)
  Loss: MSE(predicted_latent, privileged_latent)

Training Phase 3 (Fine-tune, actor unfrozen):
  Actor:  [28D obs + 32D LSTM_predicted_latent] = 60D -> MLP 128x128 -> 7D actions
  LSTM provides online latent estimates, actor adapts to imperfect predictions
```

---

## Privileged State (18D)

Information available in simulation but NOT on real hardware:

| Component | Dims | Source | Description |
|-----------|------|--------|-------------|
| block_xyz | 3 | block.data.root_pos_w | Block center-of-mass position |
| block_quat | 4 | block.data.root_quat_w | Block orientation quaternion |
| palm_xyz | 3 | robot.data.body_pos_w[:, palm_idx] | Palm link position |
| target_xyz | 3 | config.target_position | Target placement position |
| block_vel | 3 | block.data.root_lin_vel_w | Block linear velocity |
| contact_normal | 2 | contact_sensor.data.net_forces_w, projected | Dominant contact normal direction (2D projection) |

**Total: 18D**

### Normalization

Each component should be normalized to approximately [-1, 1]:
- Positions: Subtract workspace center, divide by workspace radius (~0.15m)
- Quaternion: Already in [-1, 1]
- Velocities: Divide by max expected velocity (~0.5 m/s)
- Contact normals: Already unit vectors (2D projection)

---

## PrivilegedEncoder

Encodes raw 18D privileged state into a compact 32D latent that captures task-relevant information.

```python
class PrivilegedEncoder(nn.Module):
    """Encode 18D privileged sim state into 32D latent."""

    def __init__(self, input_dim=18, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, privileged_state):
        # privileged_state: (batch, 18)
        return self.encoder(privileged_state)  # (batch, 32)
```

Training: Trained jointly with the actor during Phase 1 PPO. The actor receives `[obs_28D, encoder(privileged_18D)]` as its 60D input.

---

## AdaptationModule (LSTM)

Predicts the 32D latent from a history of proprioceptive observations.

### LSTM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| input_dim | 17 | finger_pos(7) + finger_vel(7) + force_mag(3) -- excludes phase_onehot and contact_binary |
| hidden_dim | 64 | Matches RECIPE (512 is overkill for 7-DOF hand; our obs is much smaller) |
| num_layers | 2 | Sufficient for temporal pattern extraction |
| sequence_length | 16 | ~0.5s at 30Hz control; captures grasp dynamics |
| output_dim | 32 | Must match PrivilegedEncoder latent_dim |
| dropout | 0.0 | Small model, regularization not needed |

### Why 17D input (not 28D)?

The LSTM should learn to infer hidden state from *proprioceptive* signals only:
- `finger_pos(7)`: Joint positions reveal grasp configuration
- `finger_vel(7)`: Joint velocities reveal dynamics (slip detection)
- `force_mag(3)`: Contact forces reveal interaction state

Excluded:
- `contact_binary(3)`: Redundant with force_mag (threshold of force)
- `phase_onehot(8)`: Known a priori, not informative for state estimation

### Implementation

```python
class AdaptationModule(nn.Module):
    """LSTM that predicts privileged latent from proprioceptive history."""

    def __init__(self, input_dim=17, hidden_dim=64, num_layers=2,
                 output_dim=32, seq_len=16):
        super().__init__()
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Bound output to match latent range
        )

        # History buffer (managed externally during rollouts)
        self._history = None

    def init_history(self, num_envs, device):
        """Initialize history buffer at episode start."""
        self._history = torch.zeros(
            num_envs, self.seq_len, 17, device=device
        )

    def update_history(self, proprio_obs, env_ids_to_reset=None):
        """Push new observation into history buffer."""
        # Shift left, append new
        self._history = torch.roll(self._history, -1, dims=1)
        self._history[:, -1] = proprio_obs

        # Reset history for specified envs
        if env_ids_to_reset is not None and len(env_ids_to_reset) > 0:
            self._history[env_ids_to_reset] = 0

    def forward(self, history=None):
        """Predict latent from history buffer.

        Args:
            history: (batch, seq_len, 17) or None (uses internal buffer)

        Returns:
            predicted_latent: (batch, 32)
        """
        if history is None:
            history = self._history

        lstm_out, _ = self.lstm(history)  # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1]     # (batch, hidden_dim)
        return self.output_head(last_hidden)  # (batch, 32)
```

---

## Actor Expansion: 28D -> 60D

### Phase 1 Actor (60D input)

The actor must be modified to accept 60D input instead of 28D:

```python
class System0ActorWithPrivileged(nn.Module):
    """Actor that takes [28D obs + 32D latent] = 60D input."""

    def __init__(self, obs_dim=28, latent_dim=32, hidden_dim=128, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),  # 60 -> 128
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, latent):
        x = torch.cat([obs, latent], dim=-1)  # (batch, 60)
        mean = self.net(x)
        return mean, self.log_std.expand_as(mean)
```

### Migration from Existing 28D Checkpoint

Option 1 (RECOMMENDED): Train from scratch with 60D
- Retrain both specialists with [28D obs + 32D privileged_latent]
- PrivilegedEncoder is trained end-to-end with the actor
- ~2 hours per specialist

Option 2: Expand existing checkpoint
- Initialize first linear layer: `W_new[:, :28] = W_old`, `W_new[:, 28:] = 0`
- Bias: `b_new = b_old`
- Fine-tune with small lr (1e-5) for ~200 iters
- Risk: Zero-initialized latent dims may take long to activate

---

## Training Procedure

### Phase 1: PPO with Privileged Information (~2 hours per specialist)

1. Create PrivilegedEncoder (18D -> 32D)
2. Create 60D actor, asymmetric critic (receives raw 18D privileged as extra input)
3. Train with standard PPO + domain randomization
4. Save: actor weights, critic weights, PrivilegedEncoder weights

```python
# Training loop pseudo-code
for iter in range(num_iters):
    # Collect rollout
    obs_28d = env.get_observation()  # (num_envs, 28)
    privileged_18d = env.get_privileged()  # (num_envs, 18)

    latent_32d = privileged_encoder(privileged_18d)  # (num_envs, 32)
    actor_input = torch.cat([obs_28d, latent_32d], dim=-1)  # (num_envs, 60)

    action_mean, action_log_std = actor(actor_input)
    # ... standard PPO rollout and update
```

### Phase 2: Train Adaptation Module (~30 min)

1. Freeze actor and PrivilegedEncoder
2. Collect dataset: run trained policy for ~500 episodes, record (proprio_history, privileged_latent) pairs
3. Train LSTM to predict PrivilegedEncoder output from proprio history

```python
# Data collection
dataset = []
for episode in range(500):
    history_buffer = torch.zeros(seq_len, 17)
    for step in range(episode_length):
        obs = env.get_observation()
        privileged = env.get_privileged()
        target_latent = privileged_encoder(privileged).detach()

        proprio = obs[:14]  # finger_pos(7) + finger_vel(7)
        force = obs[14:17]  # force_mag(3)
        proprio_full = torch.cat([proprio, force])  # 17D

        history_buffer = torch.roll(history_buffer, -1, dims=0)
        history_buffer[-1] = proprio_full

        dataset.append((history_buffer.clone(), target_latent))

# Training
optimizer = Adam(adaptation_module.parameters(), lr=3e-4)
for epoch in range(200):
    for history, target_latent in dataloader:
        predicted = adaptation_module(history)
        loss = F.mse_loss(predicted, target_latent)
        loss.backward()
        optimizer.step()
```

Target: MSE < 0.01 (latent predictions close to privileged encoder output)

### Phase 3: Fine-tune Actor with LSTM (~15 min)

1. Unfreeze actor
2. Replace PrivilegedEncoder with AdaptationModule in the loop
3. Fine-tune for ~200 PPO iterations with reduced lr (3e-5)

```python
for iter in range(200):
    obs_28d = env.get_observation()
    proprio_17d = extract_proprio(obs_28d)  # finger_pos + finger_vel + force_mag

    adaptation_module.update_history(proprio_17d)
    predicted_latent = adaptation_module()  # (num_envs, 32)

    actor_input = torch.cat([obs_28d, predicted_latent], dim=-1)
    # ... PPO update with actor and adaptation_module both trainable
```

---

## Optional: Depth Integration

For richer state estimation, add wrist camera depth to the LSTM input.

### CNN Depth Encoder

```python
class DepthEncoder(nn.Module):
    """Encode grayscale depth image to 64D embedding."""

    def __init__(self, embed_dim=64):
        super().__init__()
        # Simple CNN for 1-channel 64x64 depth image
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 64x1x1
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, depth_image):
        # depth_image: (batch, 1, 64, 64) grayscale
        x = self.conv(depth_image).flatten(1)
        return self.fc(x)  # (batch, 64)
```

### Modified LSTM with Depth

```python
class AdaptationModuleWithDepth(nn.Module):
    def __init__(self, proprio_dim=17, depth_embed_dim=64,
                 hidden_dim=128, num_layers=2, output_dim=32, seq_len=16):
        super().__init__()
        self.depth_encoder = DepthEncoder(embed_dim=depth_embed_dim)
        self.lstm = nn.LSTM(
            input_size=proprio_dim + depth_embed_dim,  # 17 + 64 = 81
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
```

**Note**: Depth integration is Phase 5+ work. Start with proprio-only LSTM first.

---

## Deployment Summary

Final deployed system (no privileged state needed):

```
Wrist camera depth (64x64 grayscale)
    |
    v
[DepthEncoder CNN] -> 64D depth_embed (optional)
    |
    v
Proprioception history (16 x 17D)  +  depth_embed
    |
    v
[AdaptationModule LSTM] -> 32D predicted_latent
    |
    v
[28D obs + 32D predicted_latent] = 60D
    |
    v
[Actor MLP 128x128] -> 7D finger actions
    |
    v
Dex3-1 hand
```

---

## File Organization

```
experiments/system0_skills/
    privileged_encoder.py    # PrivilegedEncoder MLP
    adaptation_module.py     # AdaptationModule LSTM + optional DepthEncoder
    policy_privileged.py     # System0ActorWithPrivileged (60D input)
    train_with_privileged.py # Phase 1: PPO with privileged state
    train_adaptation.py      # Phase 2: Supervised LSTM training
    finetune_with_lstm.py    # Phase 3: End-to-end fine-tuning
```

---

## Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 (PPO+privileged) | Single-block stacking with DR | > 70% |
| Phase 2 (LSTM supervised) | Latent prediction MSE | < 0.01 |
| Phase 2 (LSTM eval) | Stacking success (LSTM replacing encoder) | > 55% |
| Phase 3 (fine-tune) | Stacking success after fine-tuning | > 65% |
| With depth | Stacking success with depth encoder | > 70% |

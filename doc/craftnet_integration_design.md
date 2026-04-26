# CraftNet Integration Design: System 0 as Residual Correction

**Date**: 2026-03-23
**Status**: Design document (not implemented)
**Dependencies**: System 0 must be fully trained (with adaptation module) before integration.
**Target file**: `unitree_IL_lerobot/unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py`

---

## Objective

Integrate System 0 (RL-trained finger specialists) into CraftNet (GR00T N1.5 VLA) as a residual correction. During contact-rich phases (grasping, releasing), System 0 refines the VLA's finger actions. During free-space motion, the VLA controls everything.

This follows the MoDE-VLA hierarchical decision pattern: the VLA handles high-level vision-language planning while IMCopilot (our System 0) handles reactive low-level dexterous manipulation.

---

## Integration Architecture

```
CraftNet (GR00T N1.5 VLA)
    |
    v
[VLA Backbone: Eagle2.5 vision + Qwen3-VL-8B LLM + DiT action expert]
    |
    v
Action prediction: (B, H, action_dim)
    |
    +---> arm_actions[:, :21]     --> Pass through (VLA controls arm)
    |
    +---> finger_actions[:, 21:]  --> system1_fingers (7D right hand)
              |
              v
         [TactileGate] <--- contact_forces (6D from sensors)
              |
              v
         gate: per-finger sigmoid (7D, values in [0, 1])
              |
              v
         [System0 Actor] <--- [28D obs + 32D LSTM_latent] = 60D
              |
              v
         system0_delta: 7D raw finger correction
              |
              v
         final_fingers = system1_fingers + gate * system0_delta * 0.3
                                                                  ^
                                                          scale factor
```

---

## Where to Inject in modeling_groot.py

The injection point is in `predict_action_chunk()` at line ~303, after the VLA generates `action_pred`:

```python
# In GrootPolicy.predict_action_chunk():

# Line 301: outputs = self._groot_model.get_action(groot_inputs)
# Line 303: actions = outputs.get("action_pred")

# === INJECTION POINT ===
if self.system0_enabled and self._system0_actor is not None:
    actions = self._apply_system0_correction(actions, batch)
# === END INJECTION ===

# Line 305: original_action_dim = self.config.output_features[ACTION].shape[0]
# Line 306: actions = actions[:, :, :original_action_dim]
```

---

## Residual Correction Formula

```python
final_fingers = system1_fingers + gate * system0_delta * scale
```

Where:
- `system1_fingers`: VLA-predicted finger actions (7D), from DiT action expert
- `gate`: TactileGate output, per-finger sigmoid (7D), range [0, 1]
- `system0_delta`: System 0 actor output minus current finger positions (correction)
- `scale`: 0.3 (limits maximum correction to 30% of System 0's full output)

### Why scale = 0.3?

- Prevents System 0 from overwhelming the VLA's predictions
- Acts as a safety margin: even if System 0 produces maximum output, the correction is bounded
- Can be increased during Stage 4 RECAP fine-tuning if the gate learns to be conservative
- Inspired by residual learning literature: small residual corrections are more stable than full replacements

---

## TactileGate

The gate determines how much System 0 correction to apply, based on contact force feedback.

### Architecture

```python
class TactileGate(nn.Module):
    """MLP that maps contact forces to per-finger gate values.

    When no contact is detected (free-space motion), gate -> 0
    and System 0 correction is suppressed. During contact-rich
    phases, gate -> 1 and System 0 takes effect.
    """

    def __init__(self, force_dim=6, finger_dim=7, hidden_dim=32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(force_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, finger_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, contact_forces):
        """
        Args:
            contact_forces: (B, 6) -- 3 fingertip force magnitudes + 3 binary contacts
                            OR (B, 6) -- raw 6-DOF force/torque from sensors

        Returns:
            gate: (B, 7) -- per-finger gate values in [0, 1]
        """
        return self.gate_net(contact_forces)
```

### Input: Contact Forces (6D)

Two options for the 6D input:

**Option A**: Use existing System 0 observation components
- `force_mag(3)`: Clamped force magnitudes from 3 fingertip sensors
- `contact_binary(3)`: Binary contact indicators
- Total: 6D (matches our observation space)

**Option B**: Raw force/torque from real sensors
- 6-DOF force/wrench per sensor (if available on Dex3-1)
- Would need different preprocessing for sim vs. real

Recommendation: **Option A** -- matches existing observation pipeline, works in both sim and real.

### Gate Behavior

| Scenario | contact_forces | Expected gate | Effect |
|----------|---------------|---------------|--------|
| Free-space (no contact) | [0,0,0, 0,0,0] | ~[0,0,0,0,0,0,0] | VLA controls alone |
| Light contact (approaching) | [0.5,0,0, 1,0,0] | ~[0.3,0.1,0.1,...] | Mild correction |
| Firm grasp (all contacts) | [5,3,4, 1,1,1] | ~[0.9,0.8,0.9,...] | Strong System 0 |
| Release (contacts dropping) | [1,0,0, 1,0,0] | ~[0.5,0.2,0.2,...] | Moderate correction |

---

## Hierarchical Switching

Beyond the continuous gate, we also implement hard switching between VLA and System 0 based on the manipulation phase:

```python
class HierarchicalController:
    """Manages switching between VLA-only and System0-augmented control."""

    # Phase mapping to control mode
    PHASE_CONTROL = {
        "HOVER_ABOVE": "vla_only",        # Free-space, VLA handles
        "DESCEND_TO_GRASP": "vla_only",   # Approach, VLA handles
        "GRASP_HOLD": "system0",          # Contact-rich, System 0 active
        "LIFT": "system0",                # Maintaining grip, System 0 active
        "TRANSPORT": "system0",           # Maintaining grip, System 0 active
        "DESCEND_TO_PLACE": "system0",    # Maintaining grip, System 0 active
        "RELEASE_HOLD": "system0",        # Contact-rich, System 0 active
        "RETREAT": "vla_only",            # Free-space, VLA handles
    }

    def get_actions(self, phase, vla_actions, system0_actions, gate):
        if self.PHASE_CONTROL[phase] == "vla_only":
            return vla_actions  # gate is ignored
        else:
            # Apply residual correction
            delta = system0_actions - vla_actions  # correction
            return vla_actions + gate * delta * self.scale
```

### Phase Detection

On real hardware, the phase is determined by the VLA's high-level state machine or language instruction. In CraftNet, this could be:
1. **Explicit phase token**: VLA outputs a phase prediction alongside actions
2. **Contact-based**: Automatically detect phase from contact forces (simpler)
3. **Time-based**: Fixed phase schedule per task (simplest, but brittle)

Recommendation: Contact-based detection is most robust. If total contact force > threshold for N consecutive steps, switch to System 0 mode.

---

## System 0 Checkpoint to Load

The System 0 actor checkpoint must be the **post-adaptation-module** version:

- Input: 60D (28D obs + 32D LSTM-predicted latent)
- Architecture: MLP 128x128
- Output: 7D finger actions
- File: `logs/system0_pos_invariant/checkpoint_adapted.pt` (after Phase 3 fine-tuning)

The AdaptationModule LSTM must also be loaded:
- File: `logs/system0_pos_invariant/adaptation_module.pt`

Both are loaded into the GrootPolicy at initialization:

```python
class GrootPolicy(PreTrainedPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # ... existing init ...

        # System 0 integration (optional, loaded if config specifies)
        self.system0_enabled = config.system0_enabled  # default False
        self._system0_actor = None
        self._tactile_gate = None
        self._adaptation_module = None

        if self.system0_enabled:
            self._load_system0(config.system0_checkpoint_path)

    def _load_system0(self, checkpoint_path):
        """Load System 0 actor, adaptation module, and tactile gate."""
        from experiments.system0_skills.policy_privileged import System0ActorWithPrivileged
        from experiments.system0_skills.adaptation_module import AdaptationModule
        from experiments.system0_skills.tactile_gate import TactileGate

        # Load actor (60D input)
        self._system0_actor = System0ActorWithPrivileged(
            obs_dim=28, latent_dim=32, hidden_dim=128, action_dim=7
        )
        ckpt = torch.load(f"{checkpoint_path}/actor_adapted.pt", map_location="cpu")
        self._system0_actor.load_state_dict(ckpt)
        self._system0_actor.eval()

        # Load adaptation module
        self._adaptation_module = AdaptationModule(
            input_dim=17, hidden_dim=64, num_layers=2, output_dim=32, seq_len=16
        )
        ckpt = torch.load(f"{checkpoint_path}/adaptation_module.pt", map_location="cpu")
        self._adaptation_module.load_state_dict(ckpt)
        self._adaptation_module.eval()

        # Tactile gate (initialized fresh, trained during Stage 4 RECAP)
        self._tactile_gate = TactileGate(force_dim=6, finger_dim=7, hidden_dim=32)
```

---

## Training the Gate: Stage 4 RECAP RL

The TactileGate is the ONLY component trained during CraftNet fine-tuning. Everything else is frozen:

### What is frozen:
- VLA backbone (GR00T N1.5)
- System 0 actor weights
- System 0 adaptation module weights

### What is trained:
- TactileGate MLP (small: 6 -> 32 -> 32 -> 7 = ~1.3K parameters)
- Optionally: The scale factor (learnable, initialized at 0.3)

### Training objective:
During RECAP (Reinforcement Learning with Action Composition and Prediction):
1. CraftNet generates full action predictions (arm + fingers)
2. System 0 generates finger corrections
3. TactileGate modulates the correction
4. The combined actions are executed in sim
5. Task reward (stacking success) backpropagates through the gate only

```python
# RECAP training loop (simplified)
for batch in recap_dataloader:
    vla_actions = groot_model.get_action(batch)  # frozen
    finger_vla = vla_actions[:, :, 21:28]  # right hand fingers

    # System 0 forward (frozen)
    with torch.no_grad():
        obs_28d = extract_system0_obs(batch)
        adaptation_module.update_history(extract_proprio(obs_28d))
        latent = adaptation_module()
        system0_out = system0_actor(obs_28d, latent)

    # Gate (trainable)
    contact_forces = extract_contact_forces(batch)  # 6D
    gate = tactile_gate(contact_forces)

    # Compose
    delta = system0_out - finger_vla
    final_fingers = finger_vla + gate * delta * scale

    # Replace finger actions
    vla_actions[:, :, 21:28] = final_fingers

    # Execute and compute reward
    reward = env.step(vla_actions)

    # Update gate parameters only
    gate_loss = -reward.mean()  # maximize reward
    gate_optimizer.zero_grad()
    gate_loss.backward()
    gate_optimizer.step()
```

---

## Implementation in predict_action_chunk

```python
def _apply_system0_correction(self, actions, batch):
    """Apply System 0 residual correction to finger actions.

    Args:
        actions: (B, H, action_dim) -- VLA predicted actions
        batch: dict with state, contact forces, etc.

    Returns:
        actions: (B, H, action_dim) -- corrected actions
    """
    # Extract System 0 observations from batch
    # This requires the preprocessor to include finger_pos, finger_vel,
    # contact forces in the batch
    state = batch.get("state")  # (B, state_dim)

    # Extract relevant components for System 0
    # Indices depend on CraftNet's state vector ordering
    finger_pos = state[:, FINGER_POS_START:FINGER_POS_END]  # 7D
    finger_vel = state[:, FINGER_VEL_START:FINGER_VEL_END]  # 7D
    force_mag = state[:, FORCE_MAG_START:FORCE_MAG_END]      # 3D
    contact_bin = state[:, CONTACT_BIN_START:CONTACT_BIN_END] # 3D
    phase_onehot = self._get_current_phase_onehot()           # 8D

    obs_28d = torch.cat([finger_pos, finger_vel, force_mag, contact_bin, phase_onehot], dim=-1)

    # Run adaptation module
    proprio_17d = torch.cat([finger_pos, finger_vel, force_mag], dim=-1)
    self._adaptation_module.update_history(proprio_17d)
    predicted_latent = self._adaptation_module()  # (B, 32)

    # Run System 0 actor
    with torch.no_grad():
        system0_mean, _ = self._system0_actor(obs_28d, predicted_latent)  # (B, 7)

    # Compute gate
    contact_6d = torch.cat([force_mag, contact_bin], dim=-1)  # (B, 6)
    gate = self._tactile_gate(contact_6d)  # (B, 7)

    # Apply correction to each timestep in the action horizon
    finger_slice = slice(21, 28)  # right hand finger indices in CraftNet action vector
    for h in range(actions.shape[1]):
        vla_fingers = actions[:, h, finger_slice]  # (B, 7)
        delta = system0_mean - vla_fingers
        actions[:, h, finger_slice] = vla_fingers + gate * delta * self.system0_scale

    return actions
```

---

## Configuration

Add to GrootConfig:

```python
@dataclass
class GrootConfig:
    # ... existing fields ...

    # System 0 integration
    system0_enabled: bool = False
    system0_checkpoint_path: str = ""
    system0_scale: float = 0.3
    system0_finger_indices: list = field(default_factory=lambda: list(range(21, 28)))
```

---

## Testing Protocol

1. **Unit test**: Load System 0 actor + LSTM + gate in isolation, verify forward pass shapes
2. **Integration test**: Run GrootPolicy.predict_action_chunk with system0_enabled=True, verify actions are modified only in finger indices
3. **Sim eval**: Run CraftNet with System 0 on block stacking task in IsaacLab
4. **Ablation**: Compare VLA-only vs. VLA+System0 on stacking success rate
5. **Gate visualization**: Plot gate values over time to verify it activates during contact phases

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| System 0 correction destabilizes VLA | Scale factor (0.3) limits correction; gate suppresses during free-space |
| State vector index mismatch | Explicit index mapping in config, validated at init |
| LSTM history out of sync | Reset history on episode boundaries |
| Different action spaces | System 0 outputs position targets, VLA outputs position targets -- compatible |
| Real-time latency | System 0 is a small MLP (<1ms), negligible overhead |

---

## Full CraftNet Pipeline: System 2 -> System 1 -> System 0

**Added**: 2026-03-24 (Opus deep analysis)

### End-to-End Data Flow

```
Camera frames (3 views: base + left wrist + right wrist)
    |
    v
[System 2: Qwen3-VL-8B] (~10 Hz)
    |
    +---> Subtask instruction (text): "pick up red block"
    +---> Bounding box: [x1, y1, x2, y2] in image space
    |
    v
[Depth Camera Pipeline]
    |
    +---> Grayscale depth image (64x64, from wrist camera)
    +---> 3D position estimate: deproject bbox center using depth + camera intrinsics
    |         Code ref: unitree_sim_isaaclab/experiments/system0_skills/depth_to_3d.py (planned)
    |         Formula: xyz_world = K_inv @ [u*d, v*d, d] @ R_cam_to_world + t_cam
    |
    v
[IK Prior] (ik_prior_prob=0.4 from GR00T N1.5 checkpoint config)
    |
    +---> Base arm trajectory: 21D joint targets from analytical IK
    |         The IK prior provides the initial arm trajectory that gets the hand
    |         near the target 3D position. DiT refines this.
    |         In the GR00T N1.5 config, ik_prior_prob=0.4 means 40% of training
    |         samples use the IK prior as the initial noise for the diffusion process.
    |
    v
[System 1: DiT Action Head] (120 Hz)
    |
    +---> Full action prediction: (B, H=16, 28D) at each chunk
    |         21D arm joints + 7D right hand fingers
    |         DiT refines the IK prior through N=10 Euler denoising steps
    |
    +---> arm_actions[:, :21]    --> Applied directly to robot
    +---> finger_actions[:, 21:] --> system1_fingers (7D)
              |
              v
         [Distance Check]
              |
              +---> IF ee_to_target > 10cm: use system1_fingers directly (free-space)
              +---> IF ee_to_target <= 10cm: enter System 0 correction zone
              |
              v
         [System 0 Correction Zone]
              |
              +---> [AdaptationModule LSTM] <--- proprio_history (16 x 17D)
              |         Predicts 32D latent from proprioceptive history
              |         Replaces PrivilegedEncoder at deployment
              |
              +---> [System0 Actor MLP] <--- [28D obs + 32D LSTM_latent] = 60D
              |         Outputs 7D finger position targets
              |
              +---> [TactileGate MLP] <--- contact_forces (6D)
              |         Outputs per-finger gate (7D sigmoid)
              |
              v
         final_fingers = system1_fingers + gate * (system0_out - system1_fingers) * scale
                                                                                    ^
                                                                            0.3 (learnable)
```

### The 10cm Gate: When to Switch

The distance-based gate replaces the earlier phase-based switching design. Rationale:

1. **Phase detection requires a state machine**, which is fragile on real hardware.
2. **Distance is directly observable** from the 3D position estimate (System 2 bbox + depth).
3. **10cm threshold** is chosen because:
   - The arm IK is accurate to ~1cm at distances > 10cm (free-space trajectory is fine)
   - Below 10cm, contact forces become the dominant feedback signal
   - This matches the grasp specialist's training: the DESCEND_TO_GRASP phase starts ~5cm above the block and the GRASP_HOLD phase has the hand at block level

Implementation in `modeling_groot.py` at the injection point (~line 303):

```python
def _apply_system0_correction(self, actions, batch):
    # 1. Compute end-effector to target distance
    ee_pos = batch["state"][:, EE_POS_START:EE_POS_END]  # 3D
    target_pos = batch.get("target_3d")  # from System 2 + depth
    if target_pos is None:
        return actions  # no target, VLA controls alone

    distance = (ee_pos - target_pos).norm(dim=-1)  # (B,)

    # 2. Distance-based activation mask
    in_correction_zone = distance < 0.10  # 10cm threshold
    if not in_correction_zone.any():
        return actions  # all envs in free-space

    # 3. Run System 0 only for envs in correction zone
    active = in_correction_zone
    obs_28d = self._build_system0_obs(batch, active)
    
    self._adaptation_module.update_history(
        self._extract_proprio(obs_28d), 
        env_ids_to_reset=(~active).nonzero().squeeze()
    )
    predicted_latent = self._adaptation_module()  # (B, 32)

    with torch.no_grad():
        system0_mean, _ = self._system0_actor(obs_28d, predicted_latent)

    # 4. Compute tactile gate
    contact_6d = self._extract_contact_forces(batch)
    gate = self._tactile_gate(contact_6d)  # (B, 7)

    # 5. Apply correction only to active envs
    finger_slice = slice(21, 28)
    for h in range(actions.shape[1]):
        vla_fingers = actions[:, h, finger_slice]  # (B, 7)
        delta = system0_mean - vla_fingers
        correction = gate * delta * self.system0_scale
        # Only apply to envs in correction zone
        actions[active, h, finger_slice] = (
            vla_fingers[active] + correction[active]
        )

    return actions
```

### Action Space Alignment

**CRITICAL**: System 0 and CraftNet must use the same action parameterization.

| Property | System 0 | CraftNet (GR00T N1.5) |
|----------|----------|----------------------|
| Action type | Position targets | Position targets |
| Range | [-1, 1] * action_scale(1.5) | Normalized [0, 1] (mapped to joint limits) |
| Joint ordering | right_hand_indices [32,33,34,38,39,40,42] | Depends on URDF action mapping |

The residual `delta = system0_out - vla_fingers` requires both to be in the same space. Add a normalization layer:

```python
def _normalize_system0_to_craftnet(self, system0_action):
    """Convert System 0 action space to CraftNet action space.
    
    System 0: [-1, 1] * 1.5 = [-1.5, 1.5] (absolute position offset from default)
    CraftNet: [0, 1] normalized within joint limits [lower, upper]
    """
    # System 0 outputs position targets relative to default pose
    absolute_pos = self.default_finger_pos + system0_action * self.action_scale
    # Normalize to [0, 1] within CraftNet's joint limits
    normalized = (absolute_pos - self.joint_lower) / (self.joint_upper - self.joint_lower)
    return normalized.clamp(0, 1)
```

The exact mapping depends on CraftNet's action normalization, which must be verified from the checkpoint config.

### Phase-to-Specialist Mapping via System 2

On real hardware, the phase_onehot (part of System 0's 28D obs) must be provided by an external source since there is no scripted state machine. Options:

1. **System 2 predicts phase** as an auxiliary output alongside the bbox. The Qwen3-VL model can be prompted: "What manipulation phase is this? hover/descend/grasp/lift/transport/place/release/retreat". This adds ~0 latency (piggybacks on the System 2 inference).

2. **Contact-based phase estimation**: Simple rules based on contact force history:
   ```python
   def estimate_phase(contact_history, ee_height, ee_velocity):
       if max(contact_history[-5:]) < 0.1:  # no contact
           if ee_velocity[2] < -0.01: return Phase.DESCEND_TO_GRASP
           elif ee_velocity[2] > 0.01: return Phase.RETREAT
           else: return Phase.HOVER_ABOVE
       else:  # contact present
           if ee_velocity[2] > 0.01: return Phase.LIFT
           elif ee_velocity[2] < -0.01: return Phase.DESCEND_TO_PLACE
           elif abs(ee_velocity[1]) > 0.01: return Phase.TRANSPORT
           elif any(contact_decreasing): return Phase.RELEASE_HOLD
           else: return Phase.GRASP_HOLD
   ```

3. **Time-based within action chunk**: Each CraftNet action chunk (H=16 steps at 120Hz = 133ms) is short enough that the phase is approximately constant. The state machine runs at the System 2 rate (~10 Hz) and assigns phases to chunks.

**Recommendation**: Option 3 for initial integration. System 2 provides a coarse phase label every ~100ms. The System 0 phase_onehot is held constant within each action chunk. This is simple and leverages the existing state machine logic.

### Learnable Scale Factor

The scale factor (initialized at 0.3) should be learnable during Stage 4 RECAP:

```python
self.system0_scale = nn.Parameter(torch.tensor(0.3))
# During RECAP, include in gate_optimizer:
gate_optimizer = torch.optim.Adam(
    list(self._tactile_gate.parameters()) + [self.system0_scale],
    lr=1e-3
)
```

Alternatively, make scale per-finger (7D learnable vector):
```python
self.system0_scale = nn.Parameter(torch.full((7,), 0.3))
```

This allows different correction magnitudes for thumb vs. index vs. middle finger. The thumb typically needs larger corrections (wider range of motion, more influence on grasp stability).

### Pre-training the TactileGate

The gate has only ~1.3K parameters. Training from RECAP reward alone may be too weak a signal. Pre-train using supervised learning:

1. Run System 0 in sim with known phase labels
2. For each step, compute the "ideal gate":
   - gate=1.0 during GRASP_HOLD, LIFT, TRANSPORT, DESCEND_TO_PLACE
   - gate=0.0 during HOVER, DESCEND_TO_GRASP, RETREAT
   - gate=0.5 during phase transitions
3. Train: `loss = BCE(gate_pred, ideal_gate)`
4. Then fine-tune during RECAP

This gives the gate a sensible initialization before the more expensive RECAP training.

### Latency Budget

| Component | Latency | Device |
|-----------|---------|--------|
| System 2 (Qwen3-VL-8B) | ~100ms | RTX 6000 Ada |
| DiT denoising (10 steps) | ~15ms | RTX 6000 Ada |
| System 0 LSTM + MLP | <1ms | RTX 6000 Ada |
| TactileGate MLP | <0.1ms | RTX 6000 Ada |
| Total per action chunk | ~116ms | |

At 120Hz control (8.3ms per step), the action chunk (H=16) covers 133ms. System 2 runs asynchronously at ~10Hz. DiT generates chunks at the chunk rate. System 0 correction adds negligible latency (<1ms) on top of the DiT output.

On deployment hardware (e.g., Jetson Orin), the LSTM may take ~5ms instead of <1ms. Still within budget for 120Hz if overlapped with communication.

### Verification Checklist Before Integration

- [ ] System 0 trained with domain randomization (adaptation module works)
- [ ] Action space alignment verified (System 0 and CraftNet joint ordering matches)
- [ ] TactileGate pre-trained on sim rollouts
- [ ] Single-block stacking works in sim with System 0 only (>70% success)
- [ ] CraftNet generates reasonable finger actions for block stacking (baseline without System 0)
- [ ] Depth pipeline validated (bbox -> 3D position estimate within 2cm accuracy)
- [ ] LSTM real-time mode tested (single-step forward with persistent hidden state)

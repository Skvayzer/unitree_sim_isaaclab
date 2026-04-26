# Domain Randomization Design for System 0 Specialists

**Date**: 2026-03-23
**Status**: Design document (not implemented)
**Dependencies**: Both specialists must work in the stacking pipeline before DR is added.

---

## Objective

Add domain randomization (DR) to the block stacking specialists (grasp + release) so they transfer from IsaacLab simulation to the real Unitree G1 + Dex3-1 hardware. DR should cover dynamics uncertainty (mass, friction, stiffness), perception noise, and actuation delay.

---

## What to Randomize

### Physical Parameters

| Parameter | Default | DR Range | Rationale |
|-----------|---------|----------|-----------|
| Block mass (kg) | 0.05 | U(0.03, 0.15) | Real blocks vary; heavier objects need firmer grasp |
| Block friction | 2.0 | U(0.5, 3.0) | Surface finish varies; low friction = slip risk |
| Block size scale | 1.0 (4cm) | x U(0.9, 1.1) | Manufacturing tolerance, different block sizes |
| Table friction | 1.0 | U(0.3, 2.0) | Table surface varies |
| Hand friction | 1.0 | U(0.5, 2.0) | Finger pad wear/material variation |
| Joint stiffness | nominal | x U(0.8, 1.2) | Actuator compliance varies |
| Joint damping | nominal | x U(0.7, 1.3) | Mechanical wear |

### Observation Noise

| Parameter | DR Range | Rationale |
|-----------|----------|-----------|
| Joint position noise | + N(0, 0.02) rad | Encoder noise |
| Joint velocity noise | + N(0, 0.1) rad/s | Velocity estimation noise |
| Contact force noise | + N(0, 0.3) N | Tactile sensor noise |
| Contact force scale | x U(0.8, 1.2) | Sensor calibration drift |

### Actuation

| Parameter | DR Range | Rationale |
|-----------|----------|-----------|
| Action noise | + N(0, 0.05) | Motor noise |
| Action delay | 0-2 steps (uniform) | Communication latency |
| PD P gain | x U(0.8, 1.1) | Controller tuning mismatch |
| PD D gain | x U(0.7, 1.2) | Controller tuning mismatch |

### Environmental

| Parameter | DR Range | Rationale |
|-----------|----------|-----------|
| Block initial position (xy) | + U(-0.01, 0.01) m | Placement noise (already in pos-invariant training) |
| Block initial z-rotation | + U(-0.3, 0.3) rad | Block orientation variation |
| Gravity z | + U(-0.1, 0.1) m/s^2 | Sim-real mismatch |
| Random external force | 0.5 N, prob 0.05/step | Perturbation robustness |

---

## IsaacLab Implementation

### EventTerm API in configclass

Domain randomization in IsaacLab uses `EventTerm` within the environment configuration. Each randomization is specified as an event that fires at specific intervals.

```python
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.mdp import events

@configclass
class BlockStackDREventCfg:
    """Domain randomization events for block stacking."""

    # -- Physics randomization (at episode reset) --
    randomize_block_mass = EventTerm(
        func=events.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "mass_distribution_params": (0.03, 0.15),
            "operation": "abs",
        },
    )

    randomize_block_friction = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "static_friction_range": (0.5, 3.0),
            "dynamic_friction_range": (0.5, 3.0),
            "restitution_range": (0.0, 0.1),
        },
    )

    randomize_hand_friction = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (0.5, 2.0),
            "dynamic_friction_range": (0.5, 2.0),
        },
    )

    randomize_joint_stiffness = EventTerm(
        func=events.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="right_hand_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )

    randomize_block_scale = EventTerm(
        func=events.randomize_rigid_body_mass,  # placeholder -- actual scale randomization needs custom
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "scale_range": (0.9, 1.1),
        },
    )

    # -- Observation noise (every step) --
    add_joint_noise = EventTerm(
        func=events.add_noise_to_observation,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every step
        params={
            "noise_range": (-0.02, 0.02),
            "obs_key": "joint_pos",
        },
    )

    # -- Action noise (every step) --
    add_action_noise = EventTerm(
        func=events.add_noise_to_action,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "noise_range": (-0.05, 0.05),
        },
    )

    # -- External perturbation (probabilistic) --
    push_block = EventTerm(
        func=events.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.5, 2.0),  # random interval
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "force_range": (-0.5, 0.5),
        },
    )
```

### Custom EventTerm for Action Delay

Action delay requires a custom implementation (ring buffer):

```python
class ActionDelayWrapper:
    """Delays actions by 0-2 steps to simulate communication latency."""

    def __init__(self, num_envs, action_dim, max_delay=2, device="cuda"):
        self.max_delay = max_delay
        self.buffer = torch.zeros(num_envs, max_delay + 1, action_dim, device=device)
        self.delays = torch.randint(0, max_delay + 1, (num_envs,), device=device)

    def reset(self, env_ids):
        self.buffer[env_ids] = 0
        self.delays[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids),), device=self.buffer.device)

    def apply(self, actions):
        # Shift buffer
        self.buffer = torch.roll(self.buffer, 1, dims=1)
        self.buffer[:, 0] = actions
        # Return delayed actions
        delayed = self.buffer[torch.arange(len(actions)), self.delays]
        return delayed
```

---

## Training Strategy

### Option A: Train from scratch with DR (RECOMMENDED)

Rationale: Current specialists are trained without DR. Adding DR after the fact may destabilize learned behaviors. Training from scratch ensures the policy is robust from the start.

Steps:
1. Add DR events to block_stack_env.py / release_env.py
2. Train grasp specialist from scratch (1024 envs, ~2000 iters, ~2 hours)
3. Train release specialist from scratch (1024 envs, ~1000 iters, ~1 hour)
4. Evaluate in stacking pipeline with DR enabled

Expected impact: Lower peak performance (maybe 70% vs 82%) but much more robust to perturbations.

### Option B: Fine-tune existing specialists with DR

Rationale: Preserve existing learned behavior while adding robustness.

Steps:
1. Load existing checkpoint (e.g., grasp checkpoint_500.pt)
2. Start with MILD DR (50% of full ranges)
3. Train for 500 additional iters with lr=5e-5 (lower than initial training)
4. Gradually increase DR ranges over training

Risk: Policy may collapse if DR is too aggressive. Need careful monitoring.

### Recommendation

Use Option A. The specialists are small MLPs (128x128) that train in ~1 hour each. The cost of retraining is low compared to the risk of a destabilized fine-tune.

---

## Curriculum Strategy

### Phase 1: No DR (current state)
- Get stacking pipeline working with both specialists
- Establish baseline success rates

### Phase 2: Mild DR
- Mass: U(0.04, 0.08), Friction: U(1.0, 3.0)
- Joint noise: N(0, 0.01), Action noise: N(0, 0.02)
- No delays, no external forces
- Target: >60% stacking success

### Phase 3: Full DR
- All parameters at full ranges (table above)
- Action delay: 0-2 steps
- External forces: 0.5N, 5% probability
- Target: >40% stacking success (acceptable for sim-to-real)

### Phase 4: Aggressive DR (if Phase 3 succeeds easily)
- Mass: U(0.02, 0.25), Friction: U(0.3, 4.0)
- External forces: 1.0N, 10% probability
- Action delay: 0-3 steps
- For maximum real-world robustness

---

## Asymmetric Critic with DR

The critic should receive DR parameters as privileged observations (following RECIPE):

```python
# Privileged critic observations (added to existing 28D actor obs)
dr_privileged = torch.cat([
    block_mass_normalized,      # 1D
    block_friction_normalized,  # 1D
    hand_friction_normalized,   # 1D
    stiffness_scale,            # 1D
    damping_scale,              # 1D
    current_action_delay,       # 1D (integer, normalized)
], dim=-1)  # 6D total

# Critic input: 28D actor_obs + 18D sim_state + 6D dr_params = 52D
```

This helps the critic understand WHY certain actions fail (e.g., heavy block with low friction = slips), enabling better advantage estimation.

---

## Validation Protocol

After training with DR:
1. **Nominal eval**: DR disabled, default physics. Should still achieve >70% stacking.
2. **DR eval**: DR enabled at training ranges. Should achieve >40% stacking.
3. **Extreme eval**: DR at 150% of training ranges. Test generalization.
4. **Per-parameter sensitivity**: Sweep each DR parameter independently to find failure modes.

---

## References

- RECIPE (Lin et al., CoRL 2025) Table 4: Domain Randomization Setup
- MoDE-VLA (Tang et al., 2026): Domain randomization over "object scale, mass, friction, center-of-mass offset, gravity, and PD gains"
- Qi et al. (CoRL 2022): "In-Hand Object Rotation via Rapid Motor Adaptation" -- adaptation module + DR for sim-to-real
- DPPO (Ren et al., 2024): Shows diffusion policies have inherent robustness to noise, relevant if we move to diffusion-based action generation

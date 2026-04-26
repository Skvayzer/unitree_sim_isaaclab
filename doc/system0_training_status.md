# System 0 Training Status

**Last updated: 2026-03-23**
**Single authoritative source for System 0 training state.**

---

## Current Status (2026-03-23)

### Grasp Specialist — WORKING (82% at default position)

- **Checkpoint**: `logs/system0_pos_invariant/checkpoint_500.pt` (28D, best eval)
- Additional checkpoints up to 5000 iterations
- 28D obs: finger_pos(7) + finger_vel(7) + force_mag(3) + contact_binary(3) + phase_onehot(8)
- 7D actions, MLP 128->128, Gaussian policy with learnable log_std

### Position-Invariant Finetune — COMPLETED (81.2% single-block)

- **Checkpoint**: `logs/system0_pos_inv_finetune/checkpoint_5000.pt` (5000/5000 iterations, completed)
- Curriculum: +/-1cm -> +/-2cm -> +/-3cm -> +/-3.5cm
- Dramatically outperforms base pos_invariant in pipeline eval (81.2% vs 12.5%)
- Initialized from block_stack checkpoint_3000_v4

### Release Specialist — BLOCKER

- 22D checkpoint (`logs/system0_release/best_model.pt`): trained with old System0Config, INCOMPATIBLE with 28D pipeline
- 28D checkpoint (`logs/system0_release_28d/best_model.pt`, iter 722): works in pipeline
- 99% standalone success, 51.6% in full pipeline
- Must retrain with BlockStackConfig for best results

---

## Pipeline Eval Results

| Test | Block 0 Lift | Configuration |
|------|-------------|---------------|
| Single-block env | 79.7% | Baseline (grasp only) |
| Single-block pipeline | 73.4% | grasp + release specialists |
| Single-block finetune | 81.2% | pos_inv_finetune/checkpoint_5000 |
| Multi-block, 30cm spacing | 72.7% | Far blocks (proves env itself is OK) |
| Multi-block, 10cm spacing | 73.4% | Minimum safe spacing |
| Multi-block, 6cm behind | 10.9% | Blocks behind hand |
| Multi-block, 6cm symmetric | 7.8% | Original layout (arm collision) |
| Multi-block, 6cm + lift boost | 8.3% | Higher transport (didn't help) |
| Multi-block eval v7 (10cm, SM) | ~78% block 0, 0% blocks 1,2 | Full pipeline |

---

## State Machine — IMPLEMENTED

- `stacking_state_machine.py`: chains grasp + release with ParameterizedArmTrajectory
- Includes `lift_sp_boost` parameter (default=-0.2) for higher lift during LIFT/TRANSPORT
- `eval_stacking.py`: single-block and multi-block evaluation

---

## Bug Status

### Fixed

- Lifted gate added to `multi_block_rewards.py` (was causing reward hacking)
- Fingertip contact indices unified to [13, 14, 17]
- MoE policy obs dims fixed to [28, 28, 28]
- Multi-block config updated to 10cm spacing (was 6cm)

### NOT Fixed

- `lr=3e-4` in `block_stack_config.py` (should be 1e-4) — only affects `train_block_stack.py`
- `entropy_coeff=0.02` in `block_stack_config.py` (should be 0.03) — only affects `train_block_stack.py`
- No `log_std` clamping in `train_block_stack.py` or `train_multi_block.py`
- `release_env.py` still uses old System0Config (22D obs, wrong arm position, no table)
- No domain randomization anywhere
- No adaptation module
- `block_stack_rewards.py`: lifted gate not yet added (only in `multi_block_rewards.py`)

### Known Constraints

- Arm collision with adjacent blocks at <10cm spacing — FUNDAMENTAL
- Arm SR reachable range (~10cm) is approximately equal to minimum safe spacing (10cm)
- Blocks 1,2 at edge/beyond SR range -> 0% lift without expanded curriculum
- Plan was to fix with position-invariant training, but finetune training doesn't cover blocks at +/-10cm from center

---

## Roadmap

### GATE 1: Single-block stacking > 70% — PASSED

Best: 81.2% with pos_inv_finetune/checkpoint_5000

### GATE 2: Multi-block block 0 > 70% — PASSED (with 10cm spacing)

73.4% with 10cm spacing

### GATE 3: tower_rate > 0% — BLOCKED

Requires all 3 blocks to be graspable. Currently blocks 1,2 are at 0%.

Options:
- (a) Train policy directly in multi-block env to handle closer spacing (learn to avoid collisions)
- (b) Expand position-invariant curriculum to cover wider y-positions (+/-10cm)
- (c) Arrange blocks in 2D (different x positions, not just along y-axis)
- (d) Use retractable/movable block positions to avoid collision during non-target block phases

### Phase 3: Domain Randomization — NOT STARTED

- Design doc at `doc/domain_randomization_design.md`
- Depends on stacking working first
- Mass, friction, stiffness, noise, delay randomization

### Phase 4: Adaptation Module — NOT STARTED

- Design doc at `doc/adaptation_module_design.md`
- PrivilegedEncoder (18D -> 32D) + LSTM (proprio history -> 32D)
- Depends on domain randomization

### Phase 5: CraftNet Integration — NOT STARTED

- Design doc at `doc/craftnet_integration_design.md`
- Residual injection: `final_fingers = system1_fingers + gate * system0_delta * 0.3`
- Depends on adaptation module

---

## MoDE-VLA Gap Analysis

| Component | System 0 Status | MoDE-VLA/IMCopilot |
|-----------|----------------|-------------------|
| RL-trained atomic skills | Grasp 82%, release 99% | PPO in IsaacLab |
| Asymmetric actor-critic | In train_position_invariant.py only | Teacher phase |
| Specialist chaining | stacking_state_machine.py | Skill chaining |
| Domain randomization | Zero randomization | Mass, friction, CoM, PD gains |
| Adaptation module | Not implemented | LSTM teacher-student distillation |
| Proprioceptive history | Single timestep | 3-step history |
| Residual injection | Not implemented | MoDE token fusion |
| Depth in adaptation | Not implemented | N/A (VLA handles vision) |

**Estimated time to MoDE-VLA parity: 3-4 weeks of focused development.**

---

## Critical Lessons

1. Extended release is critical: need 100+ steps
2. NEVER snap fingers open during retreat
3. Slow arm lift preserves upright orientation
4. Lifted gate is MANDATORY for all downstream rewards
5. lr=1e-4, entropy_coeff=0.03, log_std clamp [-2.0, 0.5]
6. Config mismatch kills pipelines (22D vs 28D)
7. Arm collision at <10cm spacing — minimum 10cm required
8. Lift boost doesn't fix collision (collision during all phases)
9. NEVER train single policy for 930-step episodes
10. lift==place in eval means release is NOT the bottleneck

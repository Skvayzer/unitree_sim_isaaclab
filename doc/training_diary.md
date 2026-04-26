# System 0 Training Diary

Chronological log of experiments, findings, and decisions.

---

## 2026-03-23 — Code Review & MoDE-VLA Gap Analysis

### Full Codebase Review (9,249 lines, 27 files)

Reviewed all System 0 files against MoDE-VLA paper requirements.

**7 bugs found, 3 fixed by agent, 4 remaining:**

| # | Bug | Status |
|---|-----|--------|
| 1 | Lifted gate missing from multi_block_rewards.py | Fixed |
| 2 | Fingertip indices [12,14,15] in config.py vs [13,14,17] in block_stack_config | Fixed |
| 3 | lr=3e-4 in block_stack_config.py (should be 1e-4) | Not fixed |
| 4 | entropy_coeff=0.02 in block_stack_config.py (should be 0.03) | Not fixed |
| 5 | No log_std clamping in train_block_stack.py or train_multi_block.py | Not fixed |
| 6 | MoE obs dims [21,28,22] (should be [28,28,28]) | Fixed |
| 7 | release_env.py incompatible with block_stack pipeline (22D, wrong config) | BLOCKER |

### MoDE-VLA Paper Analysis

Key finding: **IMCopilot uses NO cameras/depth during RL training.** Pure proprioception + privileged sim state. Depth only used by VLA backbone. Adaptation module (LSTM over proprio history) handles sim-to-real.

**Decision**: Add depth to adaptation module (not PPO training). Keeps RL fast (30K FPS).

---

## 2026-03-23 — Pipeline Eval Sweep (4 checkpoints)

Evaluated single-block pipeline with different grasp checkpoints:

| Grasp Checkpoint | Release Checkpoint | Lift | Place |
|------------------|--------------------|------|-------|
| pos_invariant/best_model.pt | release_28d/best_model.pt | 12.5% | 12.5% |
| pos_invariant/checkpoint_5000.pt | release_28d/best_model.pt | 12.5% | 12.5% |
| pos_inv_finetune/best_model.pt (iter 37) | release_28d/best_model.pt | 71.9% | 71.9% |
| pos_inv_finetune/checkpoint_1000.pt | release_28d/best_model.pt | 73.4% | 73.4% |

**Key finding**: pos_inv_finetune (initialized from block_stack checkpoint_3000_v4) dramatically outperforms base pos_invariant (73.4% vs 12.5%). The finetune adapts to the pipeline env much better.

**Observation**: lift==place in ALL evals — if block is grasped, it's also placed. Release is NOT the bottleneck. Focus on grasp/transport.

---

## 2026-03-23 — Multi-Block Collision Investigation

### Problem Statement
Block 0 drops from ~80% lift in single-block to ~8% in multi-block environment despite identical control logic. Why?

### Isolation Tests

**Test 1: Multi-block env + direct trajectory (no state machine)**
- Result: 7.8% lift
- Conclusion: Multi-block ENV itself is the problem, not the state machine

**Test 2: Multi-block env + blocks 1,2 moved to 30cm away**
- Block 1: (0.321, -0.480, 0.819), Block 2: (0.321, 0.120, 0.819)
- Result: 72.7% lift
- Conclusion: Physical interference from adjacent blocks is the root cause

**Test 3: Lift height boost (SP-0.2 for LIFT/TRANSPORT)**
- Raises arm higher during transport phases
- Result: 8.3% (no improvement)
- Conclusion: Collision happens during ALL phases (HOVER, DESCEND too), not just transport

**Test 4: Block layout sweep**

| Layout | Description | Block 0 Lift |
|--------|-------------|-------------|
| 6cm symmetric | Original (blocks at ±6cm) | 7.8% |
| 6cm behind | Blocks behind hand (-y) | 10.9% |
| 10cm spread | Blocks at ±10cm | 73.4% |
| 15cm spread | Blocks at ±15cm | ~73% |
| 30cm spread | Blocks at ±30cm | 72.7% |

### Root Cause
The robot's arm (forearm/wrist links) physically collides with adjacent blocks during the pick-and-place trajectory. The arm's physical extent creates a collision zone of approximately ±8cm from the hand center. Any block within this zone gets hit, disrupting the grasp.

### Solution
**Minimum safe block spacing: 10cm.** Updated multi_block_config.py:
- Block 0 (red): (0.321, -0.180, 0.819) — center
- Block 1 (yellow): (0.321, -0.280, 0.819) — 10cm in -y
- Block 2 (green): (0.321, -0.080, 0.819) — 10cm in +y

### Fundamental Constraint Discovered
Arm SR reachable range (~10cm: y=-0.217 to y=-0.120) approximately equals minimum safe spacing (10cm). With 10cm spacing, blocks 1,2 are at the edges/beyond the arm's trained curriculum. Result: block 0 ~78% lift, blocks 1,2 at 0% lift. This is the primary blocker for GATE 3 (tower_rate > 0%).

---

## 2026-03-23 — Position-Invariant Finetune Training

### Training Run
- Script: train_pos_inv_finetune.py on remote PC (RTX 6000 Ada)
- Initialized from: block_stack checkpoint_3000_v4
- Curriculum: ±1cm → ±2cm → ±3cm → ±3.5cm
- 1024 envs, 5000 iterations

### Result
- Training completed: 5000/5000 iterations
- Checkpoint: logs/system0_pos_inv_finetune/checkpoint_5000.pt
- Eval: 81.2% single-block lift+place
- Significant improvement over base pos_invariant (12.5%)

---

## 2026-03-23 — Multi-Block Eval v7 (10cm spacing, state machine)

### Configuration
- Block spacing: 10cm (post-collision fix)
- State machine: BlockStackingStateMachine with lift_sp_boost=-0.2
- Grasp checkpoint: pos_inv_finetune
- Release checkpoint: release_28d

### Results
- Block 0: ~78% lift (recovered from 8% at 6cm spacing)
- Block 1: 0% lift (SR=-0.70 is beyond trained curriculum)
- Block 2: 0% lift (SR=-0.20 is beyond trained curriculum)
- Tower rate: 0%

### Analysis
10cm spacing recovers block 0 performance but blocks 1,2 are unreachable. The position-invariant training curriculum only covers ±3.5cm from center (y=-0.180). Blocks at y=-0.280 and y=-0.080 are ~10cm from center — far outside the curriculum.

---

## 2026-03-23 — MoDE-VLA Comparison Analysis

### Alignment
- Architecture: Aligned (specialist decomposition = IMCopilot atomic skills)
- Training: PPO in IsaacLab = same as IMCopilot
- Observation: 28D tactile/proprio = similar (IMCopilot uses ~165D with 3-step history)

### Critical Gaps
1. **No domain randomization** — IMCopilot randomizes mass, friction, CoM, gravity, PD gains
2. **No adaptation module** — IMCopilot uses LSTM teacher-student distillation
3. **No proprioceptive history** — IMCopilot uses 3-step history, we use single timestep
4. **No residual injection** — MoDE-VLA fuses tactile via MoE token routing

### Estimated timeline to parity: 3-4 weeks

### Reference Papers
- MoDE-VLA (Tang et al. 2026): IMCopilot architecture, MoE fusion
- RECIPE (Lin et al. CoRL 2025): DR ranges (Table 4), specialist distillation
- DPPO (Ren et al. 2024): Diffusion policy sim-to-real robustness

---

## 2026-03-23 — Documentation Consolidation

Consolidated 8 overlapping/outdated spec files into clean set:
- `doc/system0_project_spec.md` — Architecture, code map, obs/action, reward (THE reference)
- `doc/system0_training_status.md` — Current status, eval results, bugs, roadmap
- `doc/training_diary.md` — This file (chronological log)
- `doc/mode_vla_analysis.md` — Paper analysis (kept as-is)
- `doc/domain_randomization_design.md` — Future work design doc (kept as-is)
- `doc/adaptation_module_design.md` — Future work design doc (kept as-is)
- `doc/craftnet_integration_design.md` — Future work design doc (kept as-is)

Removed:
- `doc/system0_stacking_spec.md` — Merged into project_spec
- `doc/system0_skills_training.md` — Replaced by training_status
- `.auto-claude/specs/001-*` — Outdated MoE spec
- `.auto-claude/specs/003-*` — Outdated grasping spec

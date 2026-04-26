# System 0: Dexterous Block Stacking — Project Specification

## Overview

System 0 trains the Unitree G1 humanoid with Dex3-1 hands to stack 3 blocks into a tower using sim-to-real reinforcement learning in IsaacLab. The design draws inspiration from MoDE-VLA and IMCopilot architectures.

**Key design decision: Specialist decomposition.** Separate grasp and release policies are trained independently and chained by a scripted state machine. The arm follows an 8-phase analytical trajectory (computed via IK). Only the fingers (7 DOF) are learned by PPO.

**Why not end-to-end?** A full multi-block episode is 930 steps. With a 64-step rollout window, PPO sees only 7% of the episode per update — making credit assignment effectively impossible. End-to-end training was attempted 3 times and failed each time. The specialist approach gives each policy ~100 steps of directly relevant experience.

**Why not MoE?** A Mixture-of-Experts actor (3 experts + router) was explored but abandoned. Single MLP specialists outperform MoE for this task.

---

## Architecture

```
IsaacLab Env
    │
    ├── Robot State (finger pos/vel) ──┐
    ├── Contact Forces (fingertips)  ──┤
    └── Phase / Block Index ───────────┤
                                       ▼
                              28D Observation
                                       │
              ┌────────────────────────┬┘
              ▼                        ▼
   Phase ∈ {GRASP_HOLD,        Phase = RELEASE_HOLD
    LIFT, TRANSPORT,                   │
    DESCEND_TO_PLACE}                  ▼
              │               Release Specialist
              ▼               MLP 28→128→128→7
      Grasp Specialist
      MLP 28→128→128→7        Phase ∈ {HOVER,
              │                DESCEND, RETREAT}
              │                        │
              ▼                        ▼
      7D Finger Action          Fingers Open
      (position targets)        (action = -1.0)

   Arm: ParameterizedArmTrajectory (scripted, per-block IK)
   Sequencing: BlockStackingStateMachine (3 blocks x 310 steps)
```

---

## Specialist Details

### Grasp Specialist

- **Active phases:** GRASP_HOLD, LIFT, TRANSPORT, DESCEND_TO_PLACE
- **Checkpoint (base):** `logs/system0_pos_invariant/checkpoint_500.pt` — 28D obs, 82% single-block grasp
- **Checkpoint (finetuned):** `logs/system0_pos_inv_finetune/checkpoint_5000.pt` — 81.2% single-block, position curriculum up to +/-3.5cm

### Release Specialist

- **Active phase:** RELEASE_HOLD only
- **Checkpoint (OLD, incompatible):** `logs/system0_release/best_model.pt` — 22D obs (old config). INCOMPATIBLE with 28D pipeline. Do not use.
- **Checkpoint (current):** `logs/system0_release_28d/best_model.pt` — 28D obs, iter 722. 99% standalone success, 51.6% in full pipeline.

---

## Observation Space

### Single-block: 28D

| Component       | Dims | Range              | Source                                        |
|-----------------|------|--------------------|-----------------------------------------------|
| finger_pos      | 7    | ~[-0.5, 1.5] rad   | `robot.data.joint_pos[:, hand_idx]`           |
| finger_vel      | 7    | ~[-5, 5] rad/s     | `robot.data.joint_vel[:, hand_idx]`           |
| contact_force   | 3    | [0, 10] N clamped  | `fingertip_contacts`, indices [13, 14, 17]    |
| contact_binary  | 3    | {0, 1}             | force > 0.1N threshold                       |
| phase_onehot    | 8    | {0, 1}             | `traj.get_phase_onehot()`                     |

### Multi-block: 31D

All of the above, plus:

| Component         | Dims | Range  | Source              |
|-------------------|------|--------|---------------------|
| block_idx_onehot  | 3    | {0, 1} | Current block index |

Both specialists use identical observation structure. The phase one-hot tells the policy which trajectory phase is active.

---

## Action Space (7D)

- 7 right-hand finger joint position targets
- Network output in [-1, 1], scaled by `action_scale = 1.5`
- EMA smoothing: `action = 0.7 * new + 0.3 * prev`
- Joint indices (`right_hand_indices`): [32, 33, 34, 38, 39, 40, 42]

---

## 8-Phase Arm Trajectory (per block)

| Phase             | ID | Steps | Arm Motion           | Finger Control                 |
|-------------------|----|-------|----------------------|--------------------------------|
| HOVER_ABOVE       | 0  | 30    | Hold above block     | Open (action = -1)             |
| DESCEND_TO_GRASP  | 1  | 30    | Lower to block       | Open (action = -1)             |
| GRASP_HOLD        | 2  | 50    | Hold at block        | Grasp specialist               |
| LIFT              | 3  | 40    | Raise hand           | Grasp specialist (maintain)    |
| TRANSPORT         | 4  | 50    | Move to target       | Grasp specialist (maintain)    |
| DESCEND_TO_PLACE  | 5  | 40    | Lower to stack height| Grasp specialist (maintain)    |
| RELEASE_HOLD      | 6  | 40    | Hold at stack        | Release specialist             |
| RETREAT           | 7  | 30    | Lift away            | Open (action = -1)             |

**Total:** 310 steps per block. 3 blocks x 310 = 930 steps for full multi-block episode.

---

## ParameterizedArmTrajectory (SR Mapping)

Piecewise-linear mapping from block y-position to shoulder_roll, calibrated from a diagnostic sweep:

```
Diagnostic data (center_y → SR):
  y = -0.2315  →  SR = -0.70
  y = -0.2133  →  SR = -0.60
  y = -0.1944  →  SR = -0.50  (reference position)
  y = -0.1850  →  SR = -0.45
  y = -0.1761  →  SR = -0.40
  y = -0.1633  →  SR = -0.35
  y = -0.1531  →  SR = -0.30
  y = -0.1348  →  SR = -0.20
```

- Fingertip-to-block offset: 1.44cm (constant)
- Reachable y-range: approximately y = -0.217 to y = -0.120 (~10cm)

---

## Multi-Block Configuration

### Arm Collision Finding (2026-03-23)

The robot arm and forearm physically collide with adjacent blocks during trajectory phases. This is the primary blocker for multi-block stacking.

| Block Spacing | Block 0 Lift Rate | Notes                                      |
|---------------|-------------------|--------------------------------------------|
| 6cm           | 8%                | Collision destroys grasp                   |
| 10cm          | 73.4%             | Recovered — minimum safe distance          |
| 30cm          | 72.7%             | Same as 10cm; 10cm is sufficient           |

Lift height boost (SP-0.2) did NOT help — collision occurs during ALL phases, not just lift.

### Current Block Positions (10cm spacing)

| Block | Color  | Position (x, y, z)       | Notes                           |
|-------|--------|---------------------------|---------------------------------|
| 0     | Red    | (0.321, -0.180, 0.819)   | Center                          |
| 1     | Yellow | (0.321, -0.280, 0.819)   | 10cm in -y (SR clamped to -0.70)|
| 2     | Green  | (0.321, -0.080, 0.819)   | 10cm in +y (SR clamped to -0.20)|

### Fundamental Constraint

Arm reachable range (~10cm) is approximately equal to the minimum safe spacing (10cm). Blocks 1 and 2 sit at the edges of the reachable range. Current eval: block 0 ~78% lift, blocks 1 and 2 at 0% lift. This is the primary blocker for GATE 3 (tower_rate > 0%).

---

## Reward Structure

Defined in `block_stack_rewards.py`.

| Component        | Active Phase(s)                         | Lifted Gate? | Formula                                           |
|------------------|-----------------------------------------|--------------|----------------------------------------------------|
| lift_reward      | LIFT, TRANSPORT, DESC_PLACE, RELEASE    | No           | 3.0 x clamp(lift / 0.05, max=1)                   |
| place_reward     | RETREAT                                 | Yes          | 30.0 if xy < 5cm and z < 5cm                      |
| partial_place    | RETREAT                                 | Yes          | 10.0 if xy < 10cm                                 |
| approach_reward  | DESC_PLACE, RELEASE                     | Yes          | 5.0 x (1 - xy / 0.20)                             |
| release_reward   | RELEASE                                 | Yes          | 5.0 x (1 - contacts/3) x near_target              |
| contact_reward   | DESCEND, GRASP_HOLD                     | No           | 0.3 x n_contacts                                  |
| hold_reward      | LIFT, TRANSPORT, DESC_PLACE             | Yes          | 0.05 x (block above table)                        |
| drop_penalty     | LIFT, TRANSPORT                         | No           | -2.0 if block below table                         |
| knock_penalty    | HOVER, DESCEND                          | No           | -3.0 if block fell                                |
| action_smooth    | All                                     | No           | -0.002 x sum(action^2)                            |

**Lifted gate:** A sticky boolean set when `block_z > initial_z + 3cm` during LIFT. Prevents reward hacking by ensuring downstream rewards only fire after a genuine lift.

---

## PPO Hyperparameters

| Parameter         | Value   | Notes                                    |
|-------------------|---------|------------------------------------------|
| lr                | 1e-4    | NOT 3e-4 (causes gradient spikes)        |
| entropy_coeff     | 0.03    | NOT 0.02 (prevents policy collapse)      |
| clip_eps          | 0.2     |                                          |
| gamma             | 0.99    |                                          |
| gae_lambda        | 0.95    |                                          |
| ppo_epochs        | 5       |                                          |
| mini_batches      | 4       |                                          |
| action_ema_alpha  | 0.7     |                                          |
| log_std clamp     | [-2.0, 0.5] | Prevents collapse                   |
| max_grad_norm     | 1.0     |                                          |
| steps_per_rollout | 64      |                                          |

**WARNING:** `block_stack_config.py` still contains `lr=3e-4` and `entropy_coeff=0.02`. These values are incorrect but only affect `train_block_stack.py`. The scripts `train_position_invariant.py` and `train_pos_inv_finetune.py` use the correct values above.

---

## CraftNet Integration (System 2 / System 1 / System 0)

| System   | Model            | Rate    | Role                                          |
|----------|------------------|---------|-----------------------------------------------|
| System 2 | Qwen3-VL-8B     | ~10 Hz  | Vision + language -> subtask plan             |
| System 1 | DiT action head  | 120 Hz  | Visuomotor policy -> 28D joint targets        |
| System 0 | Finger specialist| 120 Hz  | Tactile correction (residual) on finger output|

Residual injection formula:

```
final_fingers = system1_fingers + gate * system0_delta * 0.3
```

System 0 is trained independently in simulation and loaded into CraftNet at deployment time.

---

## Code Map

All files under `experiments/system0_skills/`.

### Config

| File                    | Description                                           | Status |
|-------------------------|-------------------------------------------------------|--------|
| block_stack_config.py   | Main config: positions, indices, hyperparams          | lr/entropy not fixed |
| config.py               | OLD config (System0Config) — DO NOT USE               | Deprecated |
| multi_block_config.py   | Extends BlockStackConfig for 3-block, 10cm spacing    | OK     |

### Trajectory

| File                        | Description                                       | Status |
|-----------------------------|---------------------------------------------------|--------|
| arm_trajectory.py           | Phase enum, ARM_JOINT_NAMES, base ArmTrajectory   | OK     |
| parameterized_trajectory.py | Position-adaptive y->SR mapping (piecewise-linear) | OK     |
| multi_block_trajectory.py   | 3-cycle trajectory for multi-block                | OK     |

### Environment

| File                | Description                                                | Status     |
|---------------------|------------------------------------------------------------|------------|
| block_stack_env.py  | Single-block IsaacLab env                                  | OK         |
| multi_block_env.py  | 3-block IsaacLab env                                       | OK         |
| release_env.py      | Release env — INCOMPATIBLE (22D, old config, needs rewrite)| Deprecated |

### Rewards

| File                    | Description                              | Status |
|-------------------------|------------------------------------------|--------|
| block_stack_rewards.py  | Single-block rewards with lifted gate    | OK     |
| multi_block_rewards.py  | Multi-block rewards with lifted gate     | OK     |

### Policy

| File            | Description                                          | Status     |
|-----------------|------------------------------------------------------|------------|
| policy.py       | System0Actor, System0Critic, System0AsymmetricCritic | OK         |
| moe_policy.py   | MoE actor (3 experts + router) — defined, NOT used   | Abandoned  |

### Pipeline

| File                      | Description                                       | Status |
|---------------------------|---------------------------------------------------|--------|
| stacking_state_machine.py | Chains grasp + release specialists for multi-block| OK     |
| eval_stacking.py          | Single-block and multi-block evaluation           | OK     |

### Training

| File                         | Description                                    | Status     |
|------------------------------|------------------------------------------------|------------|
| train_position_invariant.py  | Position-invariant grasp with asymmetric critic| OK         |
| train_pos_inv_finetune.py    | Fine-tuning with position curriculum           | OK         |
| train_block_stack.py         | Single-block training                          | lr/entropy wrong |
| train_multi_block.py         | Multi-block end-to-end (ABANDONED approach)    | Abandoned  |

### Test Scripts

| File                            | Description                              | Status |
|---------------------------------|------------------------------------------|--------|
| test_multiblock_env_direct.py   | Multi-block env isolation test           | OK     |
| test_multiblock_far_blocks.py   | Multi-block with far blocks (30cm)       | OK     |
| test_block_layout.py            | Block layout sweep (behind/spread10/15)  | OK     |
| test_lift_boost.py              | Lift height boost test                   | OK     |
| diagnostic_sr_sweep.py          | SR diagnostic with full robot reset      | OK     |

---

## Key Numbers

| Parameter                  | Value                                    |
|----------------------------|------------------------------------------|
| Block size                 | 4cm cube                                 |
| Block mass                 | 50g                                      |
| Block friction             | 2.0                                      |
| Block initial z            | 0.819m (table surface at 0.797m)         |
| Stack heights              | [0.819, 0.859, 0.899] (each +4cm)       |
| Target position            | (0.295, -0.152, 0.819) local coords     |
| Table                      | 0.5 x 0.5m at (0.32, -0.18)             |
| Min safe block spacing     | 10cm (arm collision at closer distances) |
| Arm SR reachable range     | ~10cm (y = -0.217 to y = -0.120)        |
| Best single-block grasp    | 82%                                      |
| Best single-block finetune | 81.2%                                    |
| Best pipeline eval         | 73.4%                                    |
| FPS                        | ~30K (state-based, 1024 envs, RTX 6000 Ada) |

---

## Planned Components (not yet implemented)

| Component              | Description                                        | Design Doc                           |
|------------------------|----------------------------------------------------|--------------------------------------|
| Adaptation Module      | PrivilegedEncoder (18D->32D) + LSTM (proprio->32D) | doc/adaptation_module_design.md      |
| Domain Randomization   | Mass, friction, stiffness, noise, delay            | doc/domain_randomization_design.md   |
| CraftNet Integration   | Residual injection + tactile gate                  | doc/craftnet_integration_design.md   |

---

## Critical Lessons Learned

1. **Extended release is critical.** 50 steps is not enough; need 100+ for the release specialist.
2. **NEVER snap fingers open during retreat.** Causes 20-degree orientation degradation.
3. **Slow arm lift** (30-step smooth-step) gently breaks contact, preserves 79% upright rate.
4. **Don't load checkpoints into PPO** — `p_loss = 0` from clip ratio mismatch.
5. **Piecewise-linear SR mapping** — old linear slope was 3x wrong.
6. **Lifted gate is MANDATORY** — any reward downstream of grasping needs it to prevent reward hacking.
7. **lr = 1e-4, not 3e-4** — higher learning rate causes gradient spikes.
8. **entropy_coeff = 0.03 + log_std clamp [-2.0, 0.5]** — prevents policy collapse.
9. **Config mismatch kills pipelines** — release_env using config.py while block_stack uses block_stack_config.py produces incompatible checkpoints.
10. **Arm collision with adjacent blocks at <10cm spacing** — minimum 10cm required.
11. **Lift height boost doesn't fix collision** — collision occurs during ALL phases, not just lift.
12. **NEVER train a single policy for 930-step multi-block episodes** — credit assignment is impossible.
13. **Per-step metrics are noisy** — use deterministic eval for real success rates.

---

## Infrastructure

| Resource      | Host                                  | Hardware           | Use Case                |
|---------------|---------------------------------------|--------------------|-------------------------|
| Desktop PC    | cosmos (192.168.1.201)                | RTX 5080 16GB      | Dev/eval, max 512 envs  |
| Remote PC     | konstantinsmirnov@10.127.102.40       | RTX 6000 Ada 48GB  | Training, max 2048 envs |

- **Working directory:** `~/unitree_sim_isaaclab/experiments/system0_skills/`
- **Conda environment:** `unitree_sim_env`
- **WandB:** project `System0_MoE`, entity `skvayzer`

# System 0 Improvement Proposals

**Date**: 2026-03-26
**Author**: Research Agent (Agent 3)
**Status**: Proposals ranked by expected impact, ready for Agent 1 implementation.

---

## Priority Ranking

| Rank | Proposal | Expected Impact | Risk | Effort |
|------|----------|----------------|------|--------|
| 1 | Per-block arm trajectory (multi-object reach) | CRITICAL -- unblocks tower_rate > 0% | LOW | 1 day |
| 2 | Block-count curriculum | HIGH -- enables 2-block then 3-block | LOW | 0.5 day |
| 3 | Phased release (sequential finger retraction) | HIGH -- fixes toppling bottleneck | MEDIUM | 1 day |
| 4 | Asymmetric critic in all training scripts | MEDIUM -- faster convergence | LOW | 0.5 day |
| 5 | Pinch grasp reward shaping | MEDIUM -- better release outcomes | MEDIUM | 0.5 day |
| 6 | Reward hacking defenses | MEDIUM -- robustness | LOW | 0.5 day |
| 7 | MLP 128->256 upgrade | LOW-MEDIUM -- marginal gains | LOW | 0.25 day |
| 8 | Domain randomization Tier 1 | MEDIUM -- sim-to-real prep | MEDIUM | 1 day |
| 9 | LSTM temporal context | MEDIUM -- needed for DR | HIGH | 2 days |
| 10 | Domain randomization Tier 2-3 | HIGH -- sim-to-real | HIGH | 2 days |

---

## Proposal 1: Per-Block Arm Trajectory [CRITICAL]

**Problem**: Block 1 = 0% lift because placed block 0 collides with arm during approach.

**Solution**: Configure different shoulder_roll (SR) values per block to approach from different angles:
- Block 0: SR=-0.5 (current, approaches from right)
- Block 1: SR=-0.3 (approaches more centrally, avoids placed block 0 at target)
- Block 2: SR=-0.7 (approaches from far right)

Alternatively, increase pick-place separation from 3.8cm to >8cm.

**Files**:
- `experiments/system0_skills/parameterized_trajectory.py` -- `set_block_positions()`
- `experiments/system0_skills/stacking_state_machine.py` -- `_configure_traj_for_block()`

**Success Metric**: Block 1 lift > 50%. Tower_rate > 0%.

**Dependencies**: None. Can be tested immediately with existing checkpoints.

---

## Proposal 2: Block-Count Curriculum [HIGH]

**Problem**: Training on 3 blocks simultaneously is too hard. Policy cannot learn multi-block when single-block is the prerequisite.

**Solution**: Progressive curriculum:
- Phase 1 (iter 0-2000): 1 block, master pick-and-place
- Phase 2 (iter 2000-4000): 2 blocks, learn tower stacking
- Phase 3 (iter 4000+): 3 blocks, full tower
- Auto-promote at >70% success for 200 iters, auto-demote at <30% for 100 iters

**Files**:
- `experiments/system0_skills/train_multi_block.py` -- add curriculum scheduler
- `experiments/system0_skills/multi_block_rewards.py` -- add tower completion bonus

**Success Metric**: Monotonically increasing tower completion. 2-block tower > 50%.

**Dependencies**: Proposal 1 (arm trajectory must work for multiple blocks first).

---

## Proposal 3: Phased Release [HIGH]

**Problem**: Release specialist opens ALL fingers simultaneously, applying lateral force that topples blocks. Pipeline release success = 51.6%.

**Solution**: Three sub-phases within RELEASE_HOLD:
1. Steps 0-15: Open thumb only (it opposes other fingers; removing it lets block settle)
2. Steps 15-30: Open all fingers gradually
3. Steps 30-40: Retreat while maintaining finger opening

Add 3-bit sub-phase encoding to observation (obs_dim 28->31).

**Files**:
- `experiments/system0_skills/block_stack_rewards.py` -- phased release reward
- `experiments/system0_skills/release_env.py` -- sub-phase tracking
- `experiments/system0_skills/block_stack_config.py` -- obs_dim update

**Success Metric**: Pipeline release success > 75%. Block toppling rate < 10%.

**Dependencies**: None. Can be trained independently from other proposals.

---

## Proposal 4: Asymmetric Critic Everywhere [MEDIUM]

**Problem**: Asymmetric actor-critic only used in train_position_invariant.py. Other training scripts use basic System0Critic which cannot see privileged state, leading to noisy advantage estimates (value loss 35-162).

**Solution**: Switch ALL training scripts to System0AsymmetricCritic. Add 3D task-progress signals (block_lifted, block_above_target, n_stacked/total) to privileged obs (18D -> 21D).

**Files**:
- `experiments/system0_skills/policy.py` -- PRIVILEGED_DIM=21
- `experiments/system0_skills/train_block_stack.py` -- use asymmetric critic
- `experiments/system0_skills/train_multi_block.py` -- use asymmetric critic

**Success Metric**: Value loss variance reduced by >50%. Training convergence 20-30% faster.

**Dependencies**: None.

---

## Proposal 5: Pinch Grasp Reward [MEDIUM]

**Problem**: Uniform contact reward (n_contacts * weight) incentivizes power grasp (all fingers). Power grasp makes release harder because 3 fingers must retract simultaneously.

**Solution**: Finger-specific reward during GRASP_HOLD:
```python
reward = 2.0 * (thumb_contact + index_contact) - 1.0 * middle_contact
```

**Files**:
- `experiments/system0_skills/block_stack_rewards.py` -- line 92-93
- `experiments/system0_skills/multi_block_rewards.py` -- line 93-94

**Success Metric**: Grasp shifts to pinch pattern. Release success improves synergistically with Proposal 3.

**Dependencies**: Best combined with Proposal 3 (phased release). Can be done independently.

---

## Proposal 6: Reward Hacking Defenses [MEDIUM]

**Problem**: Known hacking vector: sliding blocks to target without lifting.

**Solution**: Three defenses:
1. Block tilt penalty during RETREAT: -5.0 when tilt > 15 deg
2. Sliding detection: penalize xy movement > 3cm/step when not lifted
3. Tower bonus requires blocks upright (quat w > 0.95)

**Files**:
- `experiments/system0_skills/multi_block_rewards.py`
- `compute_tower_bonus()` function

**Success Metric**: No reward hacking detected in adversarial eval scenarios.

**Dependencies**: None.

---

## Proposal 7: MLP 128->256 [LOW-MEDIUM]

**Problem**: Current 128-hidden MLP may be undersized for multi-position generalization. RECIPE uses 512x512x512.

**Solution**: Increase to 256x256. 4x parameters, negligible compute cost.

**Files**:
- `experiments/system0_skills/block_stack_config.py` -- hidden_dim=256
- `experiments/system0_skills/policy.py` -- no changes needed (reads from config)

**Success Metric**: 1-3% improvement in multi-position grasp. Training speed unchanged.

**Dependencies**: None. Can be combined with any retraining.

---

## Proposal 8: Domain Randomization Tier 1 [MEDIUM]

**Problem**: Zero domain randomization. Policy is brittle to any parameter variation.

**Solution**: Start with low-impact DR:
- Block mass: U(0.03, 0.08) kg (current: fixed 0.05)
- Block friction: U(1.0, 3.0) (current: fixed 2.0)
- Joint observation noise: N(0, 0.1)

**Files**:
- `experiments/system0_skills/block_stack_env.py` -- add randomize_on_reset()
- New: `experiments/system0_skills/domain_randomization.py`

**Success Metric**: Success drops to ~60% initially, recovers to >75% within 2000 iters.

**Dependencies**: Proposals 1-3 should be working first (base performance > 80%).

---

## Proposal 9: LSTM Temporal Context [MEDIUM, LATER]

**Problem**: Single-timestep observation cannot distinguish environment dynamics (mass, friction). Required for domain randomization to work at Tier 2+.

**Solution**: Add LSTM (256 hidden) processing 3-step proprioceptive history. Architecture: obs_history (3x28D) -> LSTM(256) -> MLP(256, 256) -> action(7D).

**Files**:
- `experiments/system0_skills/policy.py` -- new System0LSTMActor class
- Training scripts -- add history buffer and LSTM state management

**Success Metric**: Required for DR Tier 2. Enables online adaptation to randomized dynamics.

**Dependencies**: Proposal 8 (DR Tier 1) should be stable first.

---

## Proposal 10: Full Domain Randomization [HIGH, LATER]

**Problem**: Sim-to-real transfer requires robustness to real-world variation.

**Solution**: RECIPE Table 4 ranges: PD gain scaling, action delay, random forces (2N, 20% probability), observation noise N(0, 0.4), frame lag (10%).

**Files**:
- `experiments/system0_skills/domain_randomization.py` -- extend Tier 1
- `experiments/system0_skills/block_stack_env.py`

**Success Metric**: >70% success with full DR. Ready for adaptation module distillation.

**Dependencies**: Proposals 8, 9 (DR Tier 1 + LSTM).

---

## Implementation Order (Recommended)

```
Week 1: Proposals 1, 2, 4, 6 (arm trajectory + curriculum + critic + hacking defense)
         These are all low-risk, high-impact changes that can be done in parallel.

Week 2: Proposals 3, 5 (phased release + pinch grasp)
         Retrain release specialist with new reward structure.
         Combined: pinch grasp -> phased release -> better stacking.

Week 3: Proposals 7, 8 (MLP upgrade + DR Tier 1)
         Bundle the architecture upgrade with DR start.

Week 4+: Proposals 9, 10 (LSTM + full DR)
          Only after base performance is solid with DR Tier 1.
```

---

## Sources

- Tang et al., "MoDE-VLA" (arXiv 2603.08122)
- Ye et al., "From Power to Precision" (arXiv 2511.13710)
- Andrychowicz et al., "Learning Dexterous In-Hand Manipulation" (IJRR 2020)
- RECIPE / Lin et al. (CoRL 2025)
- "Achieving Goals Using Reward Shaping and Curriculum Learning" (arXiv 2206.02462)
- "Mastering Stacking of Diverse Shapes" (arXiv 2312.11374)
- Zhao et al., "Precise and dexterous robotic manipulation via HIL-RL" (Science Robotics 2024)
- Pinto et al., "Asymmetric Actor Critic" (RSS 2018)
- "Informed Asymmetric Actor-Critic" (arXiv 2509.26000)
- "Detecting and Mitigating Reward Hacking in RL" (arXiv 2507.05619)
- Lilian Weng, "Reward Hacking in RL" (blog, 2024)
- IsaacGymEnvs domain_randomization.md

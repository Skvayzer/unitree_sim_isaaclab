# MoDE-VLA / IMCopilot Paper Analysis

**Paper**: "Towards Human-Like Manipulation through RL-Augmented Teleoperation and Mixture-of-Dexterous-Experts VLA"
**Authors**: Tang, Ji, Xing et al. (Shanghai Jiao Tong University, Sharpa, NUS)
**ArXiv**: 2603.08122, March 2026

---

## 1. IMCopilot Observation Space

IMCopilot skills (stable grasp maintenance, in-hand object rotation) use:

- **Actor observation** `o_t`: 3-step history of proprioception, fingertip contact forces, and target rotation axis
  - Proprioception: relative joint positions (22-DOF SharpaWave hand)
  - Contact forces: 6-DOF force/wrench per fingertip (5 fingers x 6D = 30D per hand)
  - Target: rotation axis specification
  - Total per timestep is not explicitly stated, but with 3-step history the observation is approximately 3 x (22 + 30 + 3) = ~165D

- **Critic observation** (teacher phase, asymmetric): Actor observation `o_t` + privileged information `e_t`:
  - Object pose (position + orientation)
  - Object velocities (linear + angular)
  - Mass of the object
  - Center of mass offset
  - Friction coefficients

The privileged information `e_t` is encoded into a "compact latent embedding" and concatenated with `o_t`.

**Relevance to System 0**: Our 28D obs (finger_pos 7 + finger_vel 7 + force_mag 3 + contact_binary 3 + phase_onehot 8) is much simpler but appropriate for our 7-DOF Dex3-1 hand.

---

## 2. IMCopilot Reward Function

The reward for **in-hand rotation** is:

```
r = lambda_rot * r_rot + lambda_vel * r_vel + lambda_work * r_work + lambda_torq * r_torq + lambda_diff * r_diff
```

Where:
- `r_rot`: Angular velocity around the target axis (positive reward)
- `r_vel`: Penalizes undesired linear velocity (keep object stable)
- `r_work`: Penalizes excessive joint work (energy efficiency)
- `r_torq`: Penalizes excessive torque (smooth control)
- `r_diff`: Penalizes joint deviation from reference (prevents wild poses)

**Stable grasp maintenance**: Implicitly encoded -- the object must remain grasped during rotation. If dropped, the rotation reward disappears.

**Relevance to System 0**: Our reward structure (block_stack_rewards.py) is already more complex with phase-gated components and the lifted gate. The MoDE-VLA reward is simpler because each skill has a narrower scope.

---

## 3. Asymmetric Critic: Privileged Observations

During the **teacher phase**, both actor and critic receive privileged information `e_t`:
- Object pose (xyz + quaternion)
- Object velocities (linear + angular)
- Mass
- Center of mass offset
- Friction coefficients

This is encoded into a compact latent embedding and concatenated with the standard observation `o_t`.

The **student policy** (for deployment) learns to regress this embedding directly from `o_t` -- i.e., the standard observations without privileged state access. This is the teacher-student distillation mechanism (reference [38]: Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation", CoRL 2022).

**Relevance to System 0**: This is exactly the adaptation module pattern we plan to use. Our PrivilegedEncoder maps 18D sim state to 32D latent. The LSTM adaptation module replaces it at deployment.

---

## 4. Domain Randomization Ranges

MoDE-VLA does **not provide explicit DR ranges** in the paper. It states:

> "To facilitate zero-shot sim-to-real transfer, we apply domain randomization over object scale, mass, friction, center-of-mass offset, gravity, and PD gains."

No exact numbers are given. The paper references IsaacLab [37] for the simulation environment.

**From RECIPE (Lin et al., CoRL 2025)** -- Table 4 in appendix provides exact ranges:

| Parameter | Range |
|-----------|-------|
| Object Mass (kg) | [0.03, 0.1] |
| Object Friction | [0.5, 1.5] |
| Object Shape Scale | x U(0.95, 1.05) |
| Object Initial Position (cm) | + U(-0.02, 0.02) |
| Object Initial z-orientation | + U(-0.75, 0.75) rad |
| Hand Friction | [0.5, 1.5] |
| PD Controller P Gain | x U(0.8, 1.1) |
| PD Controller D Gain | x U(0.7, 1.2) |
| Random Force Scale | 2.0 N |
| Random Force Probability | 0.2 per step |
| Random Force Decay | 0.99 every 0.1s |
| Object Pos Observation Noise | 0.02 m |
| Joint Observation Noise | + N(0, 0.4) |
| Action Noise | + N(0, 0.1) |
| Frame Lag Probability | 0.1 |
| Action Lag Probability | 0.1 |
| Depth Camera Pos Noise (cm) | 0.005 |
| Depth Camera Rot Noise (deg) | 5.0 |
| Depth Camera FoV Noise (deg) | 5.0 |

---

## 5. Adaptation Module Architecture

MoDE-VLA uses teacher-student distillation (ref [38]: Qi et al., CoRL 2022 -- "Rapid Motor Adaptation"):

**Teacher phase**:
1. Train actor + critic with full privileged state `e_t` (object pose, vel, mass, friction, CoM)
2. `e_t` is encoded by a learned MLP into a compact latent
3. Both actor and critic receive `[o_t; latent(e_t)]`

**Student phase**:
1. Freeze the actor weights
2. Train an adaptation module (from [38], this is typically an LSTM or TCN) to predict the latent from a history of proprioceptive observations only
3. LSTM input: sequence of `o_t` over the last N steps
4. LSTM output: predicted latent matching the privileged encoder output
5. Loss: MSE between predicted latent and teacher's privileged latent

**Fine-tuning**:
1. After adaptation module converges, optionally unfreeze actor
2. Fine-tune end-to-end with LSTM providing online latent estimates
3. Short fine-tuning period (~200 iterations)

The paper does not specify exact LSTM dimensions. Based on [38] (Qi et al.):
- LSTM hidden dim: 128-256
- Sequence length: 50-100 timesteps of proprioceptive history
- Latent dim: 32-64
- Training: ~500 epochs on collected teacher rollout data

---

## 6. Residual Injection Mechanism

MoDE-VLA uses **residual injection** to combine force/tactile corrections with the base VLA output:

### Architecture (Equation 5 in paper):
```
v_theta(x_t, t) = [W1(Z_f + Z_suffix) || W2(Z_g + Z_suffix)]
```

Where:
- `Z_f`: Force token refined by MoDE layer (R^{H x d_pali})
- `Z_g`: Tactile token refined by MoDE layer (R^{H x d_pali})
- `Z_suffix`: Action expert suffix output from backbone
- `W1, W2`: Separate linear projection layers for arm and hand actions
- `||`: Concatenation

### Key design principles:
1. **Residual structure**: MoDE functions strictly as a *refinement* over the base VLA prediction. When force/tactile signals carry little information (free-space motion), the correction naturally diminishes toward zero.
2. **Modality-specific routing**: Arm-level torque information does not interfere with finger control, and vice versa. `W1` projects arm actions, `W2` projects hand actions.
3. **Hierarchical decision mechanism**: The VLA outputs a scalar trigger signal `c in [0, 1]`. When `c > 0.5`, IMCopilot takes full control of hand actions (Option 2). Otherwise, the VLA with MoDE generates hand actions directly (Option 1).

### MoDE Module internals:
- Input: 4 concatenated token sequences: `[Z_prefix || Z_suffix || Z_f_tilde || Z_g_tilde]`
- Self-attention layer processes the concatenated sequence
- Token-level MoE: E=8 expert MLPs with top-k=1 scatter routing
- Sparse routing allows different experts to specialize per manipulation phase (contact onset vs. steady-state vs. free-space)

**Relevance to System 0 / CraftNet**: Our planned integration is simpler:
- `final_fingers = system1_fingers + gate * system0_delta * scale`
- Tactile gate: MLP(contact 6D -> 7D per-finger, sigmoid)
- This is a direct residual correction, not the full MoDE token-level architecture

---

## 7. Training Hyperparameters

### IMCopilot (PPO in IsaacLab):
- Training environment: IsaacLab [37]
- Architecture: Asymmetric actor-critic with teacher-student distillation
- Actor/Critic: MLP (dimensions not specified -- RECIPE uses 3-layer 512x512x512)
- Policy: Outputs relative joint position offsets `a_t = Delta_theta_t`
- Integration: `q_t = q_{t-1} + lambda_scale * Delta_theta_t`, tracked by low-level PD controllers
- Initial states: Sampled around default pose; only physically feasible grasps accepted

### MoDE-VLA:
- Base backbone: pi_0 (OpenPI) -- SigLIP vision tokenizer + PaLiGemma VLM + Gemma-300M action expert
- MoDE module: E=8 expert MLPs, top-k=1 routing
- Action horizon: H=50 steps
- Denoising: N=10 Euler steps at inference
- Camera views: 3 (base + left wrist + right wrist), 224x224, 256 patch tokens each (768 total image tokens)
- Training: 60,000 steps with AdamW, cosine learning rate decay, color jitter augmentation
- Flow matching objective (Equation 1)

### From RECIPE (more detailed PPO):
- Actor/Critic: 3-layer MLP (512, 512, 512) with ELU activation
- Optimizer: AdamW, lr=0.0001, weight_decay=0.00001
- Batch size: 128
- Specialist training: 5000 steps over 100 environments
- Generalist: Diffusion Policy with 100 denoising steps, cosine noise schedule

---

## 8. Sim-to-Real Transfer

### MoDE-VLA approach:
1. **IMCopilot trained in IsaacLab** with domain randomization (mass, friction, CoM, gravity, PD gains)
2. **Teacher-student distillation**: Remove privileged state dependency via adaptation module
3. **VLA handles high-level**: Vision and language processing remains with the VLA backbone (pre-trained on real data), so the sim-to-real gap is only in the low-level finger control
4. **Hierarchical switching**: During free-space motion the VLA controls everything (good real-world performance from pre-training). IMCopilot only activates during contact-rich phases where sim training excels.

### RECIPE approach (more detailed):
1. **Automated real-to-sim tuning**: Algorithm 1 -- optimize sim physics parameters (friction, damping, inertia, joint limits) to minimize tracking error against real robot calibration data. Requires only 4 minutes of real data.
2. **Approximate object modeling**: Simple geometric primitives (cubes, cylinders, spheres) with randomized physical parameters. Works surprisingly well for sim-to-real.
3. **Hybrid object representation**: Combine sparse (3D object CoM from third-view camera) + dense (segmented depth from egocentric view) for robust transfer.
4. **Extensive domain randomization**: See Table 4 above.
5. **Divide-and-conquer distillation**: Train specialist policies per sub-task, collect successful rollouts, distill into generalist via Diffusion Policy behavioral cloning.

### DPPO approach (complementary):
1. **Diffusion Policy + PPO fine-tuning**: Pre-train diffusion policy on demonstrations, then fine-tune with PPO
2. **Zero-shot sim-to-real**: Achieved 80% real-world success on FurnitureBench One-leg (vs. 87% sim), demonstrating strong transfer
3. **Key insight**: Diffusion policies are naturally robust to noise (multi-step denoising acts as a filter), enabling better sim-to-real than Gaussian policies

---

## Key Takeaways for System 0

1. **Our specialist decomposition is architecturally aligned** with MoDE-VLA (IMCopilot is exactly this: RL-trained atomic skills chained by higher-level control).

2. **RECIPE's DR ranges (Table 4) are our primary reference** since MoDE-VLA doesn't publish exact numbers. RECIPE is also closer to our setup (humanoid, multi-fingered, sim-to-real).

3. **The adaptation module pattern is well-established**: Teacher-student distillation from Qi et al. (CoRL 2022) is used by both MoDE-VLA and RECIPE. Our planned 18D -> 32D privileged encoder + LSTM is standard.

4. **Residual injection for VLA integration**: MoDE-VLA's approach is more complex (MoE with sparse routing). Our simpler gate * delta * scale approach is appropriate for our setup where System 0 handles a single specialist behavior (block stacking fingers) rather than general contact-rich manipulation.

5. **DPPO shows diffusion policies transfer better to real**: If CraftNet moves to diffusion-based action generation, DPPO-style fine-tuning could be used instead of RECAP RL.

---

## 9. Additional Research Findings (2026-03-26)

### 9.1 Multi-Object Sequential Manipulation

MoDE-VLA/IMCopilot handles multi-object tasks by using the **same** RL-trained atomic skill (e.g., stable grasp maintenance) repeatedly, with the VLA backbone selecting different arm configurations for each object via vision+language. There are NO per-object specialist policies. The VLA decides WHERE to reach; IMCopilot decides HOW to grasp. This maps directly to our architecture: arm trajectory sets position, finger specialist handles contact. Our 0% success on blocks 1,2 is confirmed as an **arm trajectory problem** (collision with placed blocks), not a finger policy problem.

**Implication**: We need per-block arm approach angles or increased pick-place separation. The finger specialist can be reused as-is.

### 9.2 Precision Pinch vs Power Grasp

"From Power to Precision" (Ye et al., arXiv 2511.13710, UCSD) demonstrates that RL-trained dexterous policies default to power grasps when rewards incentivize maximum contact. Precision pinch (thumb + one finger) produces more controlled placement because fewer fingers retract during release. Their system achieves 82.5% zero-shot sim-to-real for precision grasping by jointly optimizing fingertip geometry and control.

**Implication**: Our uniform contact reward (`n_contacts * weight`) naturally produces power grasp. Switching to finger-specific weights (reward thumb+index, penalize middle) should improve release quality at the cost of slightly lower grasp stability.

### 9.3 Release Strategy

No published RL work explicitly trains sequential finger retraction for release. HIL-SERL (Zhao et al., Science Robotics 2024) achieves gentle release in assembly tasks via human-in-the-loop corrections. OpenAI's dexterous hand work shows emergent finger gaiting and coordinated gravity use, suggesting that with the right reward structure, sequential release can emerge. The key design: reward thumb opening first (it provides opposition force), then remaining fingers.

**Implication**: Our release specialist needs sub-phase rewards. A 3-sub-phase structure (thumb first -> all fingers -> retreat) within RELEASE_HOLD should reduce toppling.

### 9.4 Reward Shaping for Tower Stacking

Literature consensus (arXiv 2206.02462, arXiv 2312.11374): combine sparse milestone rewards with block-count curriculum. Key design principles:
- Rewards should have an **upper bound** (prevents exploitation)
- **Rapid growth, slow convergence** (prevents reward hacking near optimum)
- Curriculum from 1-block to N-block (not all blocks simultaneously)
- Binary success/failure per subtask works surprisingly well (SimpleVLA-RL: 17.1% -> 91.7%)

**Implication**: Our multi_block_rewards.py is well-designed but missing curriculum. Adding block-count progression (1 -> 2 -> 3 blocks) is the single most impactful change for tower completion.

### 9.5 Policy Architecture: MLP vs LSTM

OpenAI dexterous hand (IJRR 2020): LSTM (512 hidden) + MLP (1024) achieved ~2x performance over feedforward MLP because memory enables online system identification (infer mass, friction from interaction history). RECIPE uses 3-layer MLP 512x512x512 for specialists (no LSTM). Our 2-layer MLP 128x128 is adequate for single-block grasp (82%) but will become a bottleneck when:
1. Domain randomization is added (policy needs temporal context to adapt)
2. Multi-block sequencing (policy needs to know which block/phase it is in)

**Implication**: Increase to 256x256 immediately (free improvement). Add LSTM when starting domain randomization.

### 9.6 Domain Randomization

RECIPE Table 4 remains our primary reference. IsaacLab v2.3 added native support for student-teacher distillation and DR via `EventManager`. Critical lesson: DR should only be applied AFTER base policy exceeds 80% without DR, then gradually increased. The most impactful DR parameters for grasping (ranked): (1) friction, (2) object mass, (3) PD gains, (4) observation noise, (5) action delay.

### 9.7 Asymmetric Actor-Critic: Best Privileged Signals

"Informed Asymmetric Actor-Critic" (arXiv 2509.26000) shows that task-progress signals in the critic significantly help multi-phase tasks. Beyond our current 18D privileged obs, adding 3D task-progress signals (block_lifted, block_above_target, n_stacked/total) to the critic should accelerate convergence. Our System0AsymmetricCritic exists but is only used in train_position_invariant.py -- it should be enabled everywhere.

---

## 10. Updated Gap Analysis (2026-03-26)

| Component | System 0 Status | MoDE-VLA/IMCopilot | Gap | Priority |
|-----------|----------------|-------------------|-----|----------|
| RL-trained atomic skills | Grasp 82%, release 99% standalone | PPO in IsaacLab | SMALL | -- |
| Asymmetric actor-critic | In train_position_invariant.py only | Teacher phase (all scripts) | MEDIUM | P4 |
| Specialist chaining | stacking_state_machine.py | Skill chaining via VLA | OK | -- |
| Multi-object reach | 0% for blocks 1,2 (arm collision) | VLA selects approach angle | CRITICAL | P1 |
| Block-count curriculum | Not implemented | N/A (VLA handles sequencing) | HIGH | P2 |
| Grasp type control | Power grasp (uniform reward) | Task-specific grasp selection | MEDIUM | P5 |
| Release strategy | Simultaneous open (51.6% pipeline) | Implicit via grasp maintenance | HIGH | P3 |
| Domain randomization | Zero | Mass, friction, CoM, PD gains | HIGH | P8 |
| Adaptation module | Not implemented | LSTM teacher-student distillation | HIGH | P9 |
| Proprioceptive history | Single timestep | 3-step history | HIGH | P9 |
| Reward hacking defense | Lifted gate only | N/A | MEDIUM | P6 |
| Network capacity | MLP 128x128 (~50K params) | MLP 512x512x512 | LOW | P7 |

**Estimated time to close all gaps: 4-5 weeks** (revised from 3-4 weeks due to multi-object reach being harder than anticipated).

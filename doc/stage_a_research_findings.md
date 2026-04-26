# Stage A RL Training: Research Findings

**Date**: 2026-03-25
**Agent**: Senior Research Agent (Agent 3)
**Status**: BLOCKING ANSWER for Agent 1

---

## 1. How to Backprop Through the DiT Denoising Loop for PPO

### The Problem

GR00T N1.5's DiT runs 4-10 flow matching denoising steps (Euler integration) to produce a 28D action. For PPO we need policy gradients. Three options were evaluated:

### Option A: Full Backprop Through All Denoising Steps

- Exact policy gradients via backpropagation through time (BPTT)
- Memory cost: O(K) where K = number of denoising steps; stores all intermediate activations
- 10x memory overhead for 10 denoising steps
- Risk of exploding gradients (confirmed by DRaFT-K paper findings)
- Will NOT fit 48GB with Qwen backbone + sim + 32 envs

**Verdict: REJECTED** -- memory infeasible, gradient instability

### Option B: REINFORCE / Score Function Estimator on Final Action

- Treats entire denoising chain as black box
- Constant memory
- High variance, slow convergence, requires many samples
- DDPO_SF variant uses this (Black et al., 2023) -- shown to work but significantly slower

**Verdict: FALLBACK ONLY** -- works but convergence too slow for our timeline

### Option C: DPPO Approach (RECOMMENDED)

**Paper**: "Diffusion Policy Policy Optimization" (Allen Z. Ren et al., ICLR 2025)
- arXiv: 2409.00588
- Code: https://github.com/irom-princeton/dppo

#### Core Mechanism: Two-Layer MDP

DPPO formulates the problem as a **two-layer MDP**:
- **Outer MDP**: Environment interactions (obs -> action -> reward -> next obs)
- **Inner MDP**: Denoising process (x_K -> x_{K-1} -> ... -> x_0 = action)

Each denoising step is treated as an action in the inner MDP with a **tractable Gaussian log-likelihood**. Since each denoising step applies:

```
x_{t-1} = mu_theta(x_t, t) + sigma_t * epsilon
```

The log-probability of each step is simply:

```
log pi(x_{t-1} | x_t) = log N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)
```

This is **analytically computable** -- no backprop through the denoising chain needed.

#### How PPO Is Applied

1. Collect rollouts: run full denoising chain forward (no grad), get actions, execute in env, collect rewards
2. For each denoising step, compute the Gaussian log-likelihood ratio:
   ```
   ratio_t = pi_new(x_{t-1} | x_t) / pi_old(x_{t-1} | x_t)
   ```
3. Apply PPO clipping objective at each denoising step
4. Two discount factors:
   - `gamma_env` (0.99): standard environment discount
   - `gamma_denoise` (0.9-1.0): downweights noisier (earlier) denoising steps, stabilizes training
5. Advantage estimation uses GAE across both layers

#### Key DPPO Best Practices

1. **Fine-tune fewer denoising steps**: Only fine-tune the last K' steps (e.g., last 2-4 of 10), freeze earlier steps. This reduces GPU memory and speeds up training without sacrificing performance.
2. **Denoising discount**: Use `gamma_denoise < 1.0` to downweight contributions from noisier steps.
3. **Modified noise schedule**: Adjust the diffusion noise schedule during fine-tuning for better exploration.
4. **Structured exploration**: Diffusion parameterization naturally provides multi-modal exploration in both action space and temporal horizon.

#### Memory and Compute

- **No backprop through denoising chain** -- only need forward pass + log-likelihood computation
- Memory is O(1) per denoising step for the policy gradient (same as a single forward pass)
- Approximately 2-3x overhead vs standard PPO (due to K forward passes per action), NOT 10x
- Fits comfortably in 48GB with 32 envs

**Verdict: RECOMMENDED** -- constant memory, stable gradients, proven on robotics tasks

---

### Related Approaches (for reference)

#### Flow-GRPO (Liu et al., NeurIPS 2025)
- arXiv: 2505.05470
- Applies GRPO (critic-free) to flow matching models
- Uses ODE-to-SDE conversion for exploration + denoising reduction for efficiency
- Potentially simpler than DPPO (no value network needed)
- Relevant since GR00T N1.5 uses flow matching, not DDPM

#### pi_RL (arXiv: 2510.25889)
- Specifically targets flow-based VLAs (pi_0, pi_0.5) -- closest to our GR00T N1.5
- Two variants: Flow-Noise (learnable noise network) and Flow-SDE (ODE-to-SDE conversion)
- Achieves strong results on LIBERO with pi_0 and pi_0.5 models
- Open-source framework at https://github.com/RLinf/RLinf
- **Most directly applicable** to GR00T N1.5's flow matching architecture

#### SimpleVLA-RL (ICLR 2026, arXiv: 2509.09674)
- Uses GRPO with **binary (0/1) trajectory-level rewards**
- Treats VLA as autoregressive token generator, applies GRPO loss on action tokens
- Trajectory-level rewards uniformly propagated to all action tokens
- Does NOT handle diffusion/flow denoising -- applies to autoregressive VLAs only
- Not directly applicable to our DiT-based flow matching architecture

---

## 2. BC Regularization to Prevent Catastrophic Forgetting

### Recommendation: YES, use KL regularization

**Loss formulation:**
```
loss = ppo_loss + beta * KL(pi_current || pi_pretrained)
```

### Rationale

1. **DPPO paper findings**: DPPO uses PPO's clipping mechanism as implicit regularization (limits policy change per update). They found this sufficient for their benchmarks. However, their benchmarks start from BC pretraining on limited demos -- our case is different (we have a strong pretrained CraftNet).

2. **Literature consensus**: Strong regularization preserves pretrained capabilities but limits reward optimization. Weak regularization enables greater reward gains but risks catastrophic forgetting and mode collapse.

3. **Our specific case**: CraftNet was pretrained on real-world Dex3 data. We want to adapt it to sim (domain shift), NOT throw away its manipulation knowledge. KL regularization is critical.

### Beta Value Recommendation

| Phase | Beta | Rationale |
|-------|------|-----------|
| Early (steps 0-2K) | 0.1 | Allow larger policy changes for sim adaptation |
| Mid (steps 2K-8K) | 0.01 | Standard regularization |
| Late (steps 8K+) | 0.001 | Allow fine-tuning of details |

**Alternative: Adaptive beta** (from ADRPO, arXiv 2510.18053):
- Reduce regularization for high-advantage samples (good actions should be reinforced freely)
- Increase regularization for low-advantage samples (bad actions should stay close to pretrained)

### Implementation

For DPPO-style per-step KL:
```python
# At each denoising step t, compute KL between current and pretrained Gaussians
kl_t = 0.5 * ((sigma_pretrained/sigma_current)**2
              + (mu_current - mu_pretrained)**2 / sigma_current**2
              - 1 + 2*log(sigma_current/sigma_pretrained))

# Sum over denoising steps with denoising discount
kl_total = sum(gamma_denoise**t * kl_t for t in range(K))

# Final loss
loss = ppo_clip_loss - entropy_coeff * entropy + beta * kl_total
```

### How MoDE-VLA Handles This

MoDE-VLA (Tang et al., 2026) uses a different approach:
- RL-trained specialists (IMCopilot) are **frozen** after RL training
- The VLA backbone (with tactile injection) learns to blend specialist outputs
- No direct KL regularization -- instead, architectural separation prevents forgetting
- The HierarchicalSwitch gates between VLA output and specialist output

For our Stage A, we use the DPPO approach with explicit KL since we are fine-tuning the DiT directly, not freezing it.

---

## 3. Reward Scale Across Sequential Subtasks

### Current Proposed Scales
- Pick (grasp + lift): ~10
- Place (move + release at target): ~30
- Tower bonus (successful stack): 100

### Recommendation: Per-subtask normalization with running statistics

#### Problem with Raw Scales

1. **Advantage estimation bias**: If Place rewards dominate, the value function overfits to predicting Place outcomes, making Pick advantages noisy.
2. **Credit assignment**: With action chunking and denoising MDP, credit assignment is already hard. Unequal reward scales compound this.
3. **Learning dynamics**: The policy will prioritize Place quality over Pick quality because Place gradients are 3x larger.

#### Recommended Approach

**Option 1: Normalize rewards per phase (RECOMMENDED)**
```python
# During rollout collection, track running mean/std per phase
pick_reward_normalized = (pick_reward - pick_running_mean) / (pick_running_std + eps)
place_reward_normalized = (place_reward - place_running_mean) / (place_running_std + eps)
tower_bonus_normalized = (tower_bonus - tower_running_mean) / (tower_running_std + eps)
```

This ensures each subtask contributes equally to gradient signal regardless of raw scale.

**Option 2: Outcome-based binary rewards (SimpleVLA-RL style)**
```python
# Simplest possible: did the subtask succeed?
pick_reward = 1.0 if block_lifted else 0.0
place_reward = 1.0 if block_placed_correctly else 0.0
tower_bonus = 1.0 if tower_complete else 0.0
```

SimpleVLA-RL showed that binary 0/1 rewards work remarkably well (17.1% -> 91.7% on LIBERO-Long). This eliminates the scale problem entirely.

**Option 3: Hierarchical advantage (for later)**
Use separate value heads per subtask phase, compute advantages independently per phase, then combine. More complex but theoretically cleaner.

#### Practical Recommendation for Stage A

Start with **Option 2 (binary rewards)** for initial proof-of-concept:
- Lift success: +1.0
- Place success: +3.0 (weighted higher since it depends on lift succeeding)
- Tower complete: +10.0

Keep the ratios (1:3:10) but use much smaller absolute values. The relative weighting matters more than absolute scale. If training is unstable, switch to pure binary (1:1:1) and let the natural task structure provide the curriculum.

---

## Summary of Recommendations

| Question | Answer | Confidence |
|----------|--------|------------|
| Denoising backprop method | **Option C: DPPO** (per-step Gaussian likelihood, no backprop through chain) | HIGH |
| Specific variant for flow matching | Consider **pi_RL** or **Flow-GRPO** since GR00T uses flow matching not DDPM | HIGH |
| BC regularization | YES, KL penalty with beta=0.01 (adaptive schedule) | HIGH |
| Reward normalization | Per-phase normalization OR binary rewards | MEDIUM |

---

## Sources

- [DPPO: Diffusion Policy Policy Optimization (arXiv 2409.00588)](https://arxiv.org/abs/2409.00588)
- [DPPO Official Implementation](https://github.com/irom-princeton/dppo)
- [DPPO Project Page](https://diffusion-ppo.github.io/)
- [SimpleVLA-RL (arXiv 2509.09674)](https://arxiv.org/abs/2509.09674)
- [SimpleVLA-RL GitHub](https://github.com/PRIME-RL/SimpleVLA-RL)
- [Flow-GRPO (arXiv 2505.05470)](https://arxiv.org/abs/2505.05470)
- [pi_RL: Online RL for Flow-based VLAs (arXiv 2510.25889)](https://arxiv.org/abs/2510.25889)
- [D2PPO: DPPO with Dispersive Loss (arXiv 2508.02644)](https://arxiv.org/html/2508.02644v1)
- [Fine-tuning Diffusion Policies with Backprop Through Timesteps (arXiv 2505.10482)](https://arxiv.org/html/2505.10482)
- [ADRPO: Adaptive Divergence Regularized Policy Optimization (arXiv 2510.18053)](https://arxiv.org/html/2510.18053v1)
- [BDPO: Behavior-Regularized Diffusion Policy Optimization (arXiv 2502.04778)](https://arxiv.org/html/2502.04778v2)

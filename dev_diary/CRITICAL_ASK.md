# CRITICAL ASK — Tier 3 Architectural Decisions
**Written:** 2026-04-26 ~14:00 GMT+4
**Status:** Awaiting human approval before any action is taken.

---

## Issue 1: MoE Architecture Drift — 8-Expert Top-2 Sparse vs 4-Expert Dense Soft Gating

### What was specified
4 experts, dense soft gating (all experts always evaluated), gate weights via softmax (no top-k drop).
Rationale: matches 4 reward phases (search/palpation/closure/lift), safe under PPO's surrogate objective.

### What was implemented
8 experts, top-2 sparse routing in `System0PPOWrapper` (`system0_moe.py`).

**Why it drifted:** Based on git/diary search, the 8-expert top-2 configuration appears to have been present from the start of the RL training experiments. No diary entry explains the increase from 4→8 experts. Likely copied from a generic MoE reference implementation.

### Why it matters (expert's assessment)
1. **PPO gradient flow:** Sparse top-2 routing means 6/8 experts receive zero gradient per forward pass. With lift rate ≤1%, the few real reward signals get diluted across expert routing decisions. Dense gating gives every expert gradient on every step.
2. **DDP safety:** Top-k routing creates unused parameter warnings under DistributedDataParallel (currently single-GPU so silent, but future multi-GPU runs will break).
3. **Router input is wrong:** The intent vector (one-hot curriculum stage) is fed to the router, but curriculum is always Stage 0. The router is learning to gate on a constant input — effectively random gating.
4. **8 experts on a 4-phase task** dilutes specialization without benefit.

### Proposed fix
Replace `System0PPOWrapper` with `RLSystem0Policy` which implements 4-expert dense soft gating:
- `n_experts=4, top_k=4` (all experts, no dropping)
- Router: `Linear(obs+intent, 4)` → softmax → weighted sum of all expert outputs
- Each expert: `Linear(220, 256) → ReLU → Linear(256, 7)`
- `RLSystem0Policy` is already implemented in `system0_moe.py` (Phase 3 spec)

**Obs dim change required:** `RLSystem0Policy` uses `obs_dim=92` (different layout from current 92D input). This is compatible with the current train.py obs layout IF coarse_targets are kept. Verify `RLSystem0Policy.OBS_DIM=92` matches before switching.

### Impact
- Requires **fresh start** (new checkpoint) — architecture change means cannot load Run 9/10/11 weights
- OR: Keep current architecture for smoke test, switch only if smoke test fails
- Estimated effort: 30 minutes of code changes + 5M step smoke test

### Decision needed
**Option A:** Keep 8-expert top-2 for the GAE-fix smoke test (5M steps). If lift rate reaches >5%, declare success and THEN refactor to 4-expert dense for production run.
**Option B:** Refactor to 4-expert dense now, run smoke test with clean architecture. Adds 30min overhead but gets cleaner signal.

---

## Issue 2: Coarse Targets — Remove Zero-Padded 7D Input

### What was specified
`coarse_targets` were planned as right-hand finger position targets fed from System 1 (the high-level planner) to the RL policy. They provide context about the intended finger configuration.

### Current state
`coarse_targets = torch.zeros(N, config.target_dim, device=device)` — always zero. The policy observes 7 zeros that carry no information, occupying 7 of 220 input dims.

### Why it matters
- Wastes input capacity (minor)
- Creates a sim-real gap: real deployment would need non-zero targets from System 1, but training only sees zeros → policy learns to ignore this channel
- More importantly: removing it changes `input_dim` from 220 to 213, **breaking checkpoint compatibility**

### Proposed fix
Two options:
**Option A:** Remove coarse_targets from obs, change `target_dim=0` in `System0Config`, accept checkpoint incompatibility (fresh start). Input becomes 85D obs + 128D intent = 213D.  
**Option B:** Keep zeros permanently until System 1 integration arrives. Document as "reserved channel."

### Decision needed
This is a fresh-start decision. Recommend combining with Issue 1 (MoE refactor) into a single "clean architecture restart" if approved.

---

## Recommended Combined Action (if approved)

1. Run 5M-step smoke test with CURRENT architecture + GAE fix (no MoE refactor yet)
2. If smoke test shows lift_rate > 5%: proceed to full architectural clean-up (Issue 1 + Issue 2 combined)
3. If smoke test still stuck at ≤1%: escalate — something else is broken

**Do not implement Issues 1 or 2 without explicit user approval.**

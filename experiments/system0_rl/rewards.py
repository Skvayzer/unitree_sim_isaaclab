"""Dense per-step reward for System 0 blind tactile grasping — phase-aware v2.

Replaces the monolithic gentle_contact signal with four phase-gated signals
that create a continuous gradient: search → palpation → force-closure → lift.

Key fixes vs v1
───────────────
1. The 1.5 N gentle ceiling was physically wrong for grasping. Dex3 needs
   ~5 N/finger for a tripod grasp (μ ≈ 0.5 → mg/μ spread across 3 contacts).
   v1 penalised the policy for doing the physically necessary thing.
   New overforce threshold: 5 N (allows real grasp force, punishes crushing).

2. No opposition signal meant the policy couldn't distinguish "thumb opposing
   index+middle" from "pat block from above". r_closure fixes this via
   min(thumb_force, opposing_force) through a smooth tanh.

3. Palpation reward gates out when ≥3 right-hand pads engage. This forces a
   continuous gradient: adding a third pad switches from r_palpation to
   r_closure, which rewards thumb/finger opposition.

4. Lift rewards are gated on has_grasp to prevent the sky-hook exploit (wrist
   raises block by friction without a real pinch).

Right-hand pad indices in the 18D tactile observation
──────────────────────────────────────────────────────
See DEX3_PAD_LINKS in tasks/common_observations/tactile_state.py:
  Left hand:  0=palm_0 1=palm_1 2=palm_2 3=thumb_0 4=thumb_1 5=middle_0 6=middle_1 7=index_0 8=index_1
  Right hand: 9=palm_0 10=palm_1 11=palm_2 12=thumb_0 13=thumb_1 14=middle_0 15=middle_1 16=index_0 17=index_1
  (palm_0/1/2 are equal-split zones of palm_link; no thumb_2 on real hardware)
"""

import torch
from tasks.common_observations.tactile_state import get_tactile_obs

# ── Right-hand pad slices in the 18D tactile obs ───────────────────────────
_R_ALL    = [9, 10, 11, 12, 13, 14, 15, 16, 17]   # all 9 right-hand pads
# Relative indices within f_right (shape N×9):
# [0]=palm_0  [1]=palm_1  [2]=palm_2
# [3]=thumb_0  [4]=thumb_1
# [5]=middle_0  [6]=middle_1
# [7]=index_0  [8]=index_1

# ── Reward tuning constants ────────────────────────────────────────────────
# Phase 1 — search
SEARCH_COEFF        = 0.05   # small reward when no right-hand contact; keeps exploration

# Phase 2 — palpation (1–2 pads active, block not yet rising)
PAL_FORCE_MIN       = 0.10   # N — start of gentle window
PAL_FORCE_MAX       = 1.50   # N — top of gentle window (only used in palpation phase)
PAL_COEFF           = 0.30   # per-step; was 0.50 in v1

# Phase 3 — force closure (≥3 pads active)
CLOSURE_COEFF       = 1.00   # scales tanh(opposition / CLOSURE_SAT)
CLOSURE_SAT         = 3.00   # N total — opposition at which tanh saturates

# Grasp gate thresholds for has_grasp (used to gate lift reward)
GRASP_THUMB_THR     = 0.30   # N — minimum thumb differential force (2 modules now, was 0.50 for 3)
GRASP_OTHER_THR     = 0.50   # N — minimum opposing-finger differential force

# Phase 4 — lift (gated on has_grasp)
LIFT_DELTA          = 0.03   # m — height threshold for binary bonus
LIFT_PROP_COEFF     = 20.00  # was 3.0
LIFT_BONUS          = 50.00  # was 5.0

# Overforce — physical ceiling; ~5 N/finger is normal for a real pinch grasp
OVERFORCE_THR       = 5.00   # N (was 2.0 — that was blocking necessary grasp force)
OVERFORCE_COEFF     = 0.0    # disabled: baseline residual on middle_0 (~57N) makes this term
                             # always-negative regardless of policy behavior; tanh in r_closure
                             # already saturates at high force — this is redundant and harmful

# Block geometry
BLOCK_INIT_Z        = 0.819  # m — nominal block top

# Smoothness
SMOOTH_COEFF        = 0.002

# ── Per-env idle contact baseline ─────────────────────────────────────────
_CONTACT_BASELINE: "torch.Tensor | None" = None

# ── Diagnostic logging (stdout every N reward calls) ──────────────────────
_LOG_EVERY    = 500
_reward_calls = 0


def set_contact_baseline(env, device) -> None:
    """Capture idle table-contact forces right after env.reset() to tare sensor."""
    global _CONTACT_BASELINE
    try:
        _CONTACT_BASELINE = get_tactile_obs(env).detach().clone().to(device)
    except Exception:
        _CONTACT_BASELINE = None


def _check_force_closure(f_right: torch.Tensor) -> torch.Tensor:
    """Batched has_grasp: thumb AND at least one opposing finger engaged.

    f_right: (N, 9) right-hand differential forces, relative indices:
      [3:5] = thumb_0/1,  [5:7] = middle_0/1,  [7:9] = index_0/1
    Returns: (N,) bool tensor
    """
    thumb_max  = f_right[:, 3:5].max(dim=1).values
    index_max  = f_right[:, 7:9].max(dim=1).values
    middle_max = f_right[:, 5:7].max(dim=1).values
    return (thumb_max > GRASP_THUMB_THR) & (
        (index_max > GRASP_OTHER_THR) | (middle_max > GRASP_OTHER_THR)
    )


def compute_reward_blind(
    env,
    prev_hand_vel: torch.Tensor,             # (num_envs, 7) right-hand joint vels, previous step
    cur_hand_vel: torch.Tensor,              # (num_envs, 7) right-hand joint vels, current step
    device,
    block_init_z: "torch.Tensor | None" = None,  # (num_envs,) per-env block resting Z; falls back to BLOCK_INIT_Z
) -> torch.Tensor:
    """Vectorized per-step reward. Returns (num_envs,) tensor."""
    global _reward_calls
    _reward_calls += 1
    N = env.num_envs
    reward = torch.zeros(N, device=device)

    has_grasp = torch.zeros(N, dtype=torch.bool, device=device)

    try:
        f_raw = get_tactile_obs(env).to(device)          # (N, 18)

        if _CONTACT_BASELINE is not None:
            f = (f_raw - _CONTACT_BASELINE).clamp(min=0.0)
        else:
            f = f_raw

        f_right = f[:, _R_ALL]   # (N, 9) right-hand differential forces

        # ── Phase detection ────────────────────────────────────────────────
        active_r   = (f_right > PAL_FORCE_MIN).sum(dim=1).float()   # (N,)
        no_contact = (active_r == 0)
        palpating  = (active_r >= 1) & (active_r <= 2)

        # ── Phase 1: search reward — tiny incentive to keep exploring ──────
        r_search = SEARCH_COEFF * no_contact.float()
        reward  += r_search

        # ── Phase 2: palpation — gentle touch rewarded ONLY for 1–2 pads ──
        # Gates out when third pad engages → gradient continuously points
        # toward adding more fingers (into r_closure territory)
        in_window   = (f_right - PAL_FORCE_MIN).clamp(min=0.0)
        normed      = (in_window / (PAL_FORCE_MAX - PAL_FORCE_MIN)).clamp(max=1.0)
        best_gentle = normed.max(dim=1).values             # (N,)
        r_palpation = PAL_COEFF * best_gentle * palpating.float()
        reward     += r_palpation

        # ── Phase 3: force closure — reward thumb opposing index/middle ────
        # min(thumb, opposing) via tanh: both sides must engage; smooth gradient
        thumb_force    = f_right[:, 3:5].sum(dim=1)   # thumb_0 + thumb_1
        opposing_force = f_right[:, 5:9].sum(dim=1)   # middle_0+1 + index_0+1
        opposition     = torch.minimum(thumb_force, opposing_force)
        r_closure      = CLOSURE_COEFF * torch.tanh(opposition / CLOSURE_SAT)
        reward        += r_closure

        # ── Overforce — new 5 N ceiling allows real grasp force ────────────
        over   = (f_right - OVERFORCE_THR).clamp(min=0.0)
        r_over = -OVERFORCE_COEFF * (over ** 2).sum(dim=1)
        reward += r_over

        # ── has_grasp gate for lift reward ─────────────────────────────────
        has_grasp = _check_force_closure(f_right)

        if _reward_calls % _LOG_EVERY == 0:
            print(
                f"[reward_diag] calls={_reward_calls} "
                f"r_pal={r_palpation.mean():.4f} "
                f"r_clo={r_closure.mean():.4f} "
                f"has_grasp={has_grasp.float().mean():.3f} "
                f"active_pads={active_r.mean():.2f} "
                f"thumb_f={thumb_force.mean():.3f} "
                f"opp_f={opposing_force.mean():.3f}"
            )

    except Exception:
        pass

    # ── Phase 4: lift — gated on has_grasp ────────────────────────────────
    try:
        block_z    = env.scene["block"].data.root_pos_w[:, 2].to(device)
        ref_z      = block_init_z.to(device) if block_init_z is not None else \
                     torch.full_like(block_z, BLOCK_INIT_Z)
        lift_delta = (block_z - ref_z).clamp(min=0.0)
        g          = has_grasp.float()
        r_lift_prop  = LIFT_PROP_COEFF * lift_delta * g
        r_lift_bonus = LIFT_BONUS * (lift_delta > LIFT_DELTA).float() * g
        reward      += r_lift_prop + r_lift_bonus

        if _reward_calls % _LOG_EVERY == 0:
            print(
                f"[reward_diag] r_lift_prop={r_lift_prop.mean():.4f} "
                f"r_lift_bonus={r_lift_bonus.mean():.4f}"
            )
    except (KeyError, AttributeError):
        pass

    # ── Smoothness ─────────────────────────────────────────────────────────
    accel   = (cur_hand_vel - prev_hand_vel).abs().sum(dim=1)
    reward -= SMOOTH_COEFF * accel

    return reward


def is_lift_success(env, device, block_init_z: "torch.Tensor | None" = None) -> torch.Tensor:
    """Return bool (num_envs,) — True when block is above LIFT_DELTA threshold."""
    try:
        block_z = env.scene["block"].data.root_pos_w[:, 2].to(device)
        ref_z   = block_init_z.to(device) if block_init_z is not None else \
                  torch.full_like(block_z, BLOCK_INIT_Z)
        return (block_z - ref_z) > LIFT_DELTA
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=device)

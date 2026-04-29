# Tactile Grasp Demo — Unitree G1 Dex3-1 (right hand)

Training-free, vision-free grasp on a real G1. The operator hands an object
to the right Dex3-1 hand; tactile pads (or, as a fallback, joint torque)
trigger a compliant close that wraps around the object. Each finger stops
independently when its joint torque exceeds a per-finger threshold.

A state machine, conservative gains, and per-tick slew limits keep things
boring. No learning, no simulation, no ROS, no vision.

---

## Files

- `config.py` — single source of truth for every tunable.
- `sdk_check.py` — Step A. Verifies the Unitree SDK and the Dex3 DDS topics. Read-only.
- `hand_monitor.py` — Step B. Live torque + tactile dashboard. Read-only.
- `calibrate_poses.py` — Step C. Records OPEN and CLOSED joint angles → `poses.json`.
- `tactile_grasp.py` — Step D. The demo. Publishes commands.
- `poses.json` — output of Step C, input to Step D.
- `logs/` — per-run CSVs from Step D.

---

## Install

The Unitree Python SDK must be importable as `unitree_sdk2py`:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python
pip install -e .
```

If a conda env on the operator machine already has it, activate that env
instead.

---

## Safety preconditions

**These are non-negotiable. Read before plugging in.**

1. **E-stop accessible.** The operator's non-dominant hand must be on the
   G1 power kill or remote E-stop at all times during testing.
2. **Conservative gains.** `kp ≤ 2.0`, `kd ≤ 0.15` everywhere. Hard-capped
   in `config.py` and asserted at startup.
3. **Slow motion.** Per-tick joint-angle delta capped at
   `MAX_DQ_PER_TICK = 0.02 rad ≈ 1.15°` per 10 ms tick (≈ 2 rad/s).
4. **Torque ceiling.** `EMERGENCY_TAU = 0.6 N·m`. Any joint above → soft
   release, exit.
5. **Session timeout.** `MAX_SESSION_SEC = 120`. Hard kill after that.
6. **Signal handling.** Ctrl+C and SIGTERM trigger an open-and-exit ramp,
   not a process kill mid-grasp.
7. **First-test object.** First **5 trials** use a soft object (sponge,
   plush, foam ball). Only graduate to rigid objects after sponge runs are
   stable.

If any of these is in doubt: stop and reset.

---

## Testing protocol — run in this order

Do not skip steps. The earlier steps catch problems that would be much
worse at later steps.

### 1. Bench check
Robot powered on, arm pre-positioned so the right palm faces up. Hand not
yet near anything.
```
python sdk_check.py --iface <iface>
```
Expect: green OKs and `TACTILE_AVAILABLE_RIGHT=True`. Note the printed
schemas — if the SDK exposes different field names than this demo expects,
fix `config.py`/`tactile_grasp.py` and document the deviation here.

### 2. Monitor + finger poke
```
python hand_monitor.py --iface <iface> --side right
```
Press each fingertip in turn. Confirm:
- The `tau` row that lights up matches the expected name in
  `RIGHT_FINGER_NAMES` (poke index tip → `index_1` rises, etc.).
- Tactile pad sums rise on the matching finger.

If the names do not match the rows, **edit
`config.RIGHT_FINGER_NAMES`** before continuing. The whole demo
depends on this mapping.

### 3. Calibrate poses
```
python calibrate_poses.py --iface <iface> --side right
```
The hand becomes back-drivable (kp=0, kd=0). Move it to the OPEN palm-up
ready-to-receive pose, press Enter. Then close it around your thumb
(typical cylindrical grasp), press Enter. Inspect `poses.json` — open
should have small flexion values, closed should have larger flexion.
Re-run if either looks wrong.

### 4. Dry run
```
python tactile_grasp.py --iface <iface> --side right --dry-run
```
Walks through the full state machine, prints transitions, but **does not
publish**. Manually poke a fingertip during WAIT_CONTACT — confirm the
state machine transitions to CLOSING. Press Ctrl+C — confirm the signal
handler triggers a soft release path (not a freeze).

### 5. First live run, sponge
```
python tactile_grasp.py --iface <iface> --side right
```
Hand opens. Place a sponge in the palm. The hand should detect contact
and close gently around it, hold for 5 s, then open. If anything looks
wrong, hit Ctrl+C — that triggers EMERGENCY_RELEASE.

### 6. Repeat 5 times with the sponge
Look at the CSVs in `logs/`. Confirm `peak_tau` is well below
`EMERGENCY_TAU` (0.6 N·m). Acceptance criterion: 5/5 successful grasps,
zero emergency releases.

### 7. Graduate
Move on to a plush toy, then a stress ball or small foam block.

### 8. Only after all of the above
Try a thin or hard object. This is where the per-finger torque limits
matter most. Watch `peak_tau` carefully.

---

## Constants — what they do

All in `config.py`. Touch only if you've read this section.

| Name | Default | Purpose |
|---|---|---|
| `MAX_DQ_PER_TICK` | 0.02 rad | Per-tick slew limit. Caps joint speed at ~2 rad/s. |
| `EMERGENCY_TAU` | 0.6 N·m | Per-joint torque ceiling. Above → emergency release. |
| `KP_MAX` / `KD_MAX` | 2.0 / 0.15 | Hard caps on stiffness/damping. Asserted at startup. |
| `MAX_SESSION_SEC` | 120 | Hard kill of any single run. |
| `NOISE_TAU` | 0.10 N·m | Noise floor for `tau_est`; below this is ignored. |
| `CONTACT_TAU` | 0.15 N·m | Trigger threshold (per-joint, vs baseline) for WAIT_CONTACT → CLOSING. |
| `CONTACT_DEBOUNCE` | 5 ticks | Consecutive ticks above CONTACT_TAU before firing (50 ms at 100 Hz). |
| `TACTILE_TRIGGER` | 100 ADC | Tactile-pad trigger (max delta-from-baseline across modules). |
| `STOP_TAU` | 0.30 N·m | Per-joint freeze during CLOSING. |
| `STOP_DEBOUNCE` | 3 ticks | Ticks above STOP_TAU before freezing that joint. |
| `KP_READY/KD_READY` | 1.0 / 0.10 | Stiffness in INIT, READY, WAIT_CONTACT. |
| `KP_CLOSING/KD_CLOSING` | 1.5 / 0.10 | Stiffness while closing. |
| `KP_HOLD/KD_HOLD` | 2.0 / 0.15 | Stiffness during HOLDING (firmest setting). |
| `KP_RELEASE/KD_RELEASE` | 1.0 / 0.10 | Stiffness during RELEASING and EMERGENCY_RELEASE. |
| `DT_SEC` | 0.01 | Control loop period (100 Hz). |
| `T_INIT_RAMP` / `T_RELEASE_RAMP` | 1.5 s | Ramp durations to/from open pose. |
| `T_HOLD` | 5.0 s | Default hold duration (override with `--hold-sec`). |
| `WAIT_CONTACT_TIMEOUT` | 30 s | Give up if no contact in this long. |
| `EMERGENCY_RELEASE_TIMEOUT` | 1.0 s | Hard cap on emergency-release ramp. |
| `RIGHT_FINGER_NAMES` | thumb_0..middle_1 | SDK motor index → finger name. **Verify by poking.** |
| `PRESS_LABELS_RIGHT` | thumb_0..palm_2 | Tactile module labels (9 modules). |

The constants `CONTACT_TAU < STOP_TAU < EMERGENCY_TAU` form an escalation
ladder. Keep it ordered.

---

## State machine summary

```
INIT
  └─▶ ramp from current q to pose_open over 1.5 s
READY
  └─▶ hold pose_open at low stiffness, capture tau / tactile baseline
WAIT_CONTACT
  └─▶ same hold, watching for tactile delta > TACTILE_TRIGGER
      OR any |Δτ_est| > CONTACT_TAU for CONTACT_DEBOUNCE ticks
CLOSING
  └─▶ slew toward pose_closed at MAX_DQ_PER_TICK; per-joint freeze when
      |Δτ_est_i| > STOP_TAU for STOP_DEBOUNCE ticks. Done when all
      frozen OR all arrived.
HOLDING
  └─▶ hold the freeze targets at firm stiffness for hold_sec, then RELEASING
RELEASING
  └─▶ ramp to pose_open over 1.5 s, then DONE
EMERGENCY_RELEASE   (any |τ| > EMERGENCY_TAU, signal, or session timeout)
  └─▶ low-stiffness ramp to pose_open, then DONE
```

`tactile_grasp.py` prints state transitions live and writes a CSV per run
to `logs/run_<timestamp>.csv` (columns: t, state, q*, q_cmd*, tau*,
tactile_max).

At end of run, one-line summary:

```
RUN_SUMMARY  result=GRASPED  contact_t=2.31s  close_t=0.84s
             frozen_joints=[1,3,5]  peak_tau=0.42N·m  emergency=False
```

`result` is one of `GRASPED`, `NO_CONTACT_TIMEOUT`, `EMERGENCY_RELEASE`,
`USER_ABORT`.

---

## Troubleshooting

- **`sdk_check.py` reports no message on `rt/dex3/right/state`.** Wrong
  network interface. Run `ip -br link` and pick the one connected to the
  G1 (commonly `eth0`, sometimes `enp...`).
- **`hand_monitor.py` rows do not match expected names.** Edit
  `config.RIGHT_FINGER_NAMES` to match the indices that respond when each
  fingertip is pressed. The Unitree convention may differ on your unit.
- **Hand drops when `calibrate_poses.py` exits.** The script ramps a soft
  hold for ~1 s before exiting. If you Ctrl+C it ungracefully and the
  hand still falls, the OS killed Python before the soft hold finished —
  re-run and let it exit cleanly.
- **Contact never fires in WAIT_CONTACT.** Either tactile is unavailable
  (run with `--no-tactile`) or `CONTACT_TAU` is too high for your sensor
  noise. Check `hand_monitor.py` first to see what the noise floor
  actually looks like.
- **Joint freezes too early during CLOSING (small object slips).** Lower
  `STOP_TAU` only after confirming the joints can actually push harder
  without exceeding `EMERGENCY_TAU`. Bench-test first.

---

## Out of scope

- Wrist or arm motion (the arm is pre-positioned manually or by a separate
  teleop layer).
- Bimanual coordination.
- Vision / object recognition.
- ROS bridges.
- Logging anywhere but the local CSVs.

"""All tunables for the tactile-grasp demo. Single source of truth.

Every magic number used by the other scripts in this directory comes from
here. If a value needs changing, change it here and document why in git.

The hard-cap values (MAX_DQ_PER_TICK, EMERGENCY_TAU, KP_MAX, KD_MAX,
MAX_SESSION_SEC) are safety limits. They are asserted at startup. Do not
raise them without a code review.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Safety hard-caps (do not raise without review)
# ----------------------------------------------------------------------------
# Per-tick joint-angle slew limit. At 100 Hz this is ~2 rad/s — well below
# any teleop command rate the Dex3 sees in practice.
MAX_DQ_PER_TICK = 0.02       # rad per DT_SEC tick

# Per-joint absolute |Δtau| ceiling above baseline. Above this -> release.
#
# UNITS NOTE: the Dex3 firmware on this unit reports tau_est in raw sensor
# counts, NOT N·m. Idle noise floor differs by hand (calibrated 2026-04-30
# from hand_monitor.py readings):
#   - right hand: worst |Δtau| ~1700 raw across joints
#   - left  hand: worst |Δtau| ~7300 raw on thumb_0, ~1800 on the rest.
#     thumb_0 dominates so the global threshold must size around it. As a
#     consequence, real contact below ~10k raw will not register on the
#     quieter joints. If contact never fires later, switch to per-joint
#     thresholds.
EMERGENCY_TAU = 500000.0     # raw |Δtau| above baseline (left-hand sized)

# Stiffness ceilings. Anything above is rejected at startup.
KP_MAX = 2.0
KD_MAX = 0.15

# Hard kill of any single run.
MAX_SESSION_SEC = 120

# ----------------------------------------------------------------------------
# Contact detection (all values in raw |Δ-from-baseline| units)
# ----------------------------------------------------------------------------
# Below this, ignore as drift / sensor noise.
NOISE_TAU = 100.0

# Trigger threshold for WAIT_CONTACT -> CLOSING (per-joint, vs baseline).
# Sized at ~1.5× worst-joint idle noise on the left hand (thumb_0 ≈ 7300).
CONTACT_TAU = 20000.0

# Consecutive ticks above CONTACT_TAU before firing the trigger.
# At 100 Hz, 5 ticks == 50 ms.
CONTACT_DEBOUNCE = 5

# Tactile pad trigger. We use |Δ| (firmware reports both signs depending
# on the pad). Measured 2026-04-30: idle ≈ 0, firm tap on a fingertip
# moves the pad sum by 200-3500. 500 sits cleanly above noise.
TACTILE_TRIGGER = 500.0

# Per-joint freeze threshold during CLOSING (raw |Δtau| vs baseline).
# Joint pushing back hard against an obstacle should hit this before
# anything dangerous happens, so it must sit comfortably below
# EMERGENCY_TAU. Sized at ~3× worst-joint idle noise on the left hand.
STOP_TAU = 350000.0

# Consecutive ticks above STOP_TAU before freezing a joint. Lowered from 3
# to 1 so a joint freezes within 10 ms of the first STOP_TAU crossing —
# otherwise a fast wedge (e.g. firm finger / rigid object resisting the
# CLOSING ramp) outruns the debounce window and EMERGENCY_TAU fires before
# the joint can be frozen.
STOP_DEBOUNCE = 1

# ----------------------------------------------------------------------------
# Stiffness per phase
# ----------------------------------------------------------------------------
KP_READY,    KD_READY    = 1.0, 0.10
KP_CLOSING,  KD_CLOSING  = 1.5, 0.10
KP_HOLD,     KD_HOLD     = 2.0, 0.15
KP_RELEASE,  KD_RELEASE  = 1.0, 0.10
KP_LIMP,     KD_LIMP     = 0.0, 0.0     # for calibrate_poses backdrive mode

# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
DT_SEC = 0.01                 # 100 Hz control loop
T_INIT_RAMP = 1.5             # seconds to ramp from current q to pose_open
# Hard cap on the INIT phase. KP_READY=1.0 may not be enough to drive a
# deeply-curled hand all the way to pose_open against gravity / spring,
# so the strict arrival check (|q - pose_open| < ARRIVAL_TOL_RAD on every
# joint) can hang forever. After this many seconds we exit INIT into READY
# regardless of arrival, so contact detection becomes reachable even when
# a joint or two are stuck a little short of pose_open.
T_INIT_MAX = 4.0
T_HOLD = 5.0                  # default hold duration
T_RELEASE_RAMP = 1.5
WAIT_CONTACT_TIMEOUT = 30.0   # give up if no contact in this long
EMERGENCY_RELEASE_TIMEOUT = 1.0  # hard limit on emergency-release ramp
# Hard cap on the CLOSING phase. Without this, an unblocked-but-slow joint
# (e.g. thumb_2 has 1.2 rad of travel and may never freeze if it's not
# touching the object) can hang CLOSING indefinitely while the other fingers
# are already wrapped around the object. After this many seconds we exit
# CLOSING into HOLDING with whatever freeze/arrival state we have.
T_CLOSING_MAX = 3.0

# ----------------------------------------------------------------------------
# Joint layout
# ----------------------------------------------------------------------------
# Dex3-1 right hand index ordering (from Unitree's Dex3_1_Right_JointIndex
# enum). VERIFY by physically poking each tip and watching tau_est rise on
# the matching index in hand_monitor.py. If it doesn't match, fix this list
# and re-run from Step B.
RIGHT_FINGER_NAMES = [
    "thumb_0",   # 0
    "thumb_1",   # 1
    "thumb_2",   # 2
    "index_0",   # 3
    "index_1",   # 4
    "middle_0",  # 5
    "middle_1",  # 6
]

# Left hand has a different ordering in Unitree's enum: thumb, middle, index
# (instead of thumb, index, middle). Provided for completeness; demo only
# uses the right hand.
LEFT_FINGER_NAMES = [
    "thumb_0",   # 0
    "thumb_1",   # 1
    "thumb_2",   # 2
    "middle_0",  # 3
    "middle_1",  # 4
    "index_0",   # 5
    "index_1",   # 6
]

NUM_FINGER_JOINTS = 7

# Tactile press_sensor_state module ordering (from
# real_robot_tactile_smoke_test.py, mapping established 2026-04-26).
# m0..m5 are six finger sensors (no thumb-proximal sensor exists).
# m6..m8 are three palm sensors; the palm strip is wired mirrored, so
# left and right enumerate in opposite directions.
PRESS_LABELS_RIGHT = (
    "thumb_0", "thumb_1",
    "middle_0", "middle_1",
    "index_0", "index_1",
    "palm_0", "palm_1", "palm_2",
)
PRESS_LABELS_LEFT = (
    "thumb_0", "thumb_1",
    "middle_0", "middle_1",
    "index_0", "index_1",
    "palm_2", "palm_1", "palm_0",
)

# ----------------------------------------------------------------------------
# DDS topics
# ----------------------------------------------------------------------------
TOPIC_RIGHT_STATE = "rt/dex3/right/state"
TOPIC_RIGHT_CMD   = "rt/dex3/right/cmd"
TOPIC_LEFT_STATE  = "rt/dex3/left/state"
TOPIC_LEFT_CMD    = "rt/dex3/left/cmd"

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------
# Tolerance for "we have arrived at the target pose" per joint.
ARRIVAL_TOL_RAD = 0.05
CLOSING_DONE_TOL_RAD = 0.02

# Rolling-window length (in ticks) for tau_baseline averaging at READY entry.
BASELINE_WINDOW_TICKS = 50    # 0.5 s at 100 Hz


def assert_safety_caps() -> None:
    """Sanity-check the values in this file at startup. Called from every
    script that publishes commands."""
    assert MAX_DQ_PER_TICK <= 0.05, "MAX_DQ_PER_TICK is dangerously large"
    assert EMERGENCY_TAU > NOISE_TAU, "EMERGENCY_TAU must exceed noise floor"
    assert STOP_TAU < EMERGENCY_TAU, "STOP_TAU must be below EMERGENCY_TAU"
    assert CONTACT_TAU < STOP_TAU, "CONTACT_TAU must be below STOP_TAU"
    for kp in (KP_READY, KP_CLOSING, KP_HOLD, KP_RELEASE):
        assert kp <= KP_MAX, f"kp {kp} exceeds KP_MAX={KP_MAX}"
    for kd in (KD_READY, KD_CLOSING, KD_HOLD, KD_RELEASE):
        assert kd <= KD_MAX, f"kd {kd} exceeds KD_MAX={KD_MAX}"
    assert MAX_SESSION_SEC > 0
    assert DT_SEC > 0

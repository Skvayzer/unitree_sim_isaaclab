#!/usr/bin/env python3
"""Step B — live read-only TUI for one Dex3 hand.

Use this to:
  1. Watch resting torques and confirm the noise floor.
  2. Poke each fingertip in turn and verify the SDK motor index -> finger
     name mapping in config.RIGHT_FINGER_NAMES is correct.
  3. Sanity-check the tactile module ordering.

READ-ONLY: never publishes.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

import config


def import_sdk():
    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_  # type: ignore
    except ImportError as e:
        print(
            f"[ERROR] {e}\n\nRun sdk_check.py first to confirm the SDK is "
            f"installed.",
            file=sys.stderr,
        )
        sys.exit(2)
    return ChannelFactoryInitialize, ChannelSubscriber, HandState_


class StateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None
        self._count = 0

    def on_message(self, sample) -> None:
        with self._lock:
            self._latest = sample
            self._count += 1

    def latest(self):
        with self._lock:
            return self._latest, self._count


def render_panel(
    side: str,
    finger_names: list[str],
    sample,
    elapsed: float,
    has_tactile: bool,
    press_labels: tuple[str, ...],
    tau_baseline: list[float],
    tactile_baseline: list[float],
    tau_spike_thr: float,
    tactile_spike_thr: float,
    peak_dtau: list[float],
    peak_dtactile: list[float],
) -> str:
    """Render the live panel.

    Tau and tactile are shown as Δ-from-baseline because the firmware on
    this Dex3 reports both in raw sensor units with large nonzero idle
    offsets (tau_est sits at hundreds–thousands per joint, tactile sums
    at ~640k per module). Absolute values do not visibly change on a tap;
    deltas do. Same approach as real_robot_tactile_smoke_test.py.
    """
    motors = list(sample.motor_state)
    n = max(len(motors), len(finger_names))
    lines = []
    lines.append(f"\033[H\033[J{side} hand   t={elapsed:6.2f}s   "
                 f"motors={len(motors)}   "
                 f"(tau/tactile shown as Δ-from-baseline)")
    lines.append("")
    # Make absolutely clear which table this is: 7 actuated joints. The
    # thumb has 3 motors (thumb_0/1/2), so the row count is 7 and there is
    # no palm row — palm is not actuated. This is NOT the tactile mapping.
    lines.append(f"  ── JOINT TORQUES ({len(motors)} motors; "
                 f"thumb=3 joints, middle/index=2 each, no palm joint) ──")
    lines.append(
        f"  {'idx':<4}{'name':<10}{'q(rad)':>10}{'dq(rad/s)':>12}"
        f"{'tau(raw)':>12}{'Δtau':>12}{'peak|Δtau|':>12}   contact?"
    )
    for i in range(n):
        name = finger_names[i] if i < len(finger_names) else f"#{i}"
        if i < len(motors):
            m = motors[i]
            q = float(m.q)
            dq = float(m.dq)
            tau = float(m.tau_est)
            base = tau_baseline[i] if i < len(tau_baseline) else 0.0
            dtau = tau - base
            peak = peak_dtau[i] if i < len(peak_dtau) else 0.0
            mark = "▓" if abs(dtau) > tau_spike_thr else "."
            lines.append(
                f"  {i:<4}{name:<10}{q:+10.3f}{dq:+12.3f}"
                f"{tau:+12.1f}{dtau:+12.1f}{peak:+12.1f}     {mark}"
            )
        else:
            lines.append(f"  {i:<4}{name:<10}      —          —          —"
                         f"           —           —     —")

    if has_tactile:
        press = list(getattr(sample, "press_sensor_state", []))
        sums = [float(sum(p.pressure)) for p in press]
        deltas = [
            sums[i] - (tactile_baseline[i] if i < len(tactile_baseline) else 0.0)
            for i in range(len(sums))
        ]
        lines.append("")
        # Separate, explicit section so the 9-pad tactile mapping is not
        # confused with the 7-motor joint table above. Layout follows
        # SYSTEM0_FACTS.md (verified 2026-04-26): 2 thumb + 2 middle +
        # 2 index + 3 palm pads.
        lines.append(f"  ── TACTILE PADS ({len(press)} modules; "
                     f"thumb=2 pads, middle/index=2 each, palm=3) ──")
        # Print one pad per line for readability instead of one long row.
        for i, d in enumerate(deltas):
            lbl = press_labels[i] if i < len(press_labels) else f"m{i}"
            peak = peak_dtactile[i] if i < len(peak_dtactile) else 0.0
            mark = "▓" if abs(d) > tactile_spike_thr else "."
            lines.append(f"  m{i:<2} {lbl:<10}  Δ={d:+8.0f}   peak|Δ|="
                         f"{peak:6.0f}   {mark}")
        max_d = max((abs(d) for d in deltas), default=0.0)
        max_peak = max(peak_dtactile, default=0.0)
        lines.append(f"  max|Δ|={max_d:.0f}   peak|Δ|max={max_peak:.0f}   "
                     f"thr={tactile_spike_thr:.0f}")
    lines.append("")
    lines.append("  Poke each fingertip and confirm Δtau row matches its name.")
    lines.append("  Reported scales let you set CONTACT_TAU/STOP_TAU/TACTILE_TRIGGER")
    lines.append("  in config.py — they are in the same raw units shown here.")
    lines.append("  Ctrl-C to exit.")
    return "\n".join(lines)


def capture_baseline(cache: "StateCache", duration_s: float, n_motors: int):
    """Average tau_est per motor and pressure-sum per tactile module over
    ``duration_s``. Hand must be still and untouched."""
    tau_acc = [0.0] * n_motors
    tac_acc: Optional[list[float]] = None
    n = 0
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        s, _ = cache.latest()
        if s is not None:
            for i in range(min(n_motors, len(s.motor_state))):
                tau_acc[i] += float(s.motor_state[i].tau_est)
            press = getattr(s, "press_sensor_state", None)
            if press is not None and len(press) > 0:
                sums = [float(sum(p.pressure)) for p in press]
                if tac_acc is None:
                    tac_acc = list(sums)
                else:
                    for i in range(min(len(tac_acc), len(sums))):
                        tac_acc[i] += sums[i]
            n += 1
        time.sleep(0.01)
    if n == 0:
        return [0.0] * n_motors, []
    tau_base = [v / n for v in tau_acc]
    tac_base = [v / n for v in tac_acc] if tac_acc is not None else []
    return tau_base, tac_base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only Dex3 hand monitor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0", help="DDS network interface.")
    p.add_argument("--side", choices=("left", "right"), default="right")
    p.add_argument("--rate", type=float, default=20.0,
                   help="Print rate in Hz.")
    p.add_argument("--baseline-sec", type=float, default=1.0,
                   help="Seconds of idle data to average for tau/tactile "
                        "baseline at startup.")
    p.add_argument(
        "--tau-spike", type=float, default=50.0,
        help="|Δtau| above this raw value lights the contact mark. "
             "Adjust to match the noise floor you observe — Dex3 firmware "
             "reports tau_est in raw sensor units, not N·m.",
    )
    p.add_argument(
        "--tactile-spike", type=float, default=200.0,
        help="|Δtactile-sum| (max across modules) above this lights the "
             "tactile contact mark. Calibrate to your hand.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        print("[ERROR] --rate must be positive", file=sys.stderr)
        return 2

    ChannelFactoryInitialize, ChannelSubscriber, HandState_ = import_sdk()
    ChannelFactoryInitialize(0, args.iface)

    if args.side == "right":
        topic = config.TOPIC_RIGHT_STATE
        finger_names = list(config.RIGHT_FINGER_NAMES)
        press_labels = config.PRESS_LABELS_RIGHT
    else:
        topic = config.TOPIC_LEFT_STATE
        finger_names = list(config.LEFT_FINGER_NAMES)
        press_labels = config.PRESS_LABELS_LEFT

    cache = StateCache()
    sub = ChannelSubscriber(topic, HandState_)
    sub.Init(cache.on_message, 0)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        s, _ = cache.latest()
        if s is not None:
            break
        time.sleep(0.05)
    sample, _ = cache.latest()
    if sample is None:
        print(f"[ERROR] no message on {topic} within 3s", file=sys.stderr)
        return 3

    has_tactile = (
        hasattr(sample, "press_sensor_state")
        and len(getattr(sample, "press_sensor_state")) > 0
    )
    n_motors = len(sample.motor_state)

    print(f"[..] capturing {args.baseline_sec:.2f}s baseline — keep the hand "
          f"still and untouched...")
    tau_baseline, tactile_baseline = capture_baseline(
        cache, args.baseline_sec, n_motors,
    )
    print(f"[OK] tau_baseline (raw): "
          + ", ".join(f"{v:+.1f}" for v in tau_baseline))
    if tactile_baseline:
        print(f"[OK] tactile_baseline (sum/module): "
              + ", ".join(f"{v:.0f}" for v in tactile_baseline))

    period = 1.0 / args.rate
    t0 = time.monotonic()
    next_tick = t0

    # Running peak |Δ-from-baseline| per channel. Updated every cache
    # sample (not just every render tick) so we don't miss spikes that
    # land between frames.
    peak_dtau = [0.0] * n_motors
    n_pads = len(getattr(sample, "press_sensor_state", []) or [])
    peak_dtactile = [0.0] * n_pads
    last_count = 0
    try:
        while True:
            now = time.monotonic()
            sample, count = cache.latest()
            if sample is not None and count != last_count:
                last_count = count
                for i in range(min(n_motors, len(sample.motor_state))):
                    base = tau_baseline[i] if i < len(tau_baseline) else 0.0
                    a = abs(float(sample.motor_state[i].tau_est) - base)
                    if a > peak_dtau[i]:
                        peak_dtau[i] = a
                if has_tactile and tactile_baseline:
                    press = list(getattr(sample, "press_sensor_state", []))
                    for i, p in enumerate(press):
                        if i >= len(peak_dtactile):
                            break
                        s_i = float(sum(p.pressure))
                        b_i = (tactile_baseline[i]
                               if i < len(tactile_baseline) else 0.0)
                        a = abs(s_i - b_i)
                        if a > peak_dtactile[i]:
                            peak_dtactile[i] = a

            if now < next_tick:
                time.sleep(min(period, next_tick - now))
                continue
            next_tick += period
            if sample is None:
                continue
            panel = render_panel(
                args.side, finger_names, sample, now - t0,
                has_tactile, press_labels,
                tau_baseline, tactile_baseline,
                args.tau_spike, args.tactile_spike,
                peak_dtau, peak_dtactile,
            )
            sys.stdout.write(panel)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        try:
            sub.Close()
        except Exception:
            pass

    # Exit summary: per-joint peak |Δtau|, plus suggested config.py thresholds.
    overall_peak = max(peak_dtau, default=0.0)
    print()
    print(f"=== {args.side} hand peak |Δtau| over session ===")
    for i, p in enumerate(peak_dtau):
        name = finger_names[i] if i < len(finger_names) else f"#{i}"
        print(f"  {i}  {name:<10}  peak|Δtau| = {p:+10.1f} raw")
    print(f"  overall max = {overall_peak:.1f} raw")
    if peak_dtactile:
        overall_tac = max(peak_dtactile, default=0.0)
        print()
        print(f"=== {args.side} hand peak |Δtactile-sum| over session ===")
        for i, p in enumerate(peak_dtactile):
            lbl = press_labels[i] if i < len(press_labels) else f"m{i}"
            print(f"  m{i}  {lbl:<10}  peak|Δ| = {p:+10.0f}")
        print(f"  overall max = {overall_tac:.0f}")
    if overall_peak > 0:
        print()
        print("Suggested config.py thresholds (idle-noise calibration):")
        print(f"  CONTACT_TAU   = {1.5 * overall_peak:>8.0f}")
        print(f"  STOP_TAU      = {3.0 * overall_peak:>8.0f}")
        print(f"  EMERGENCY_TAU = {5.0 * overall_peak:>8.0f}")
        print("  (multipliers 1.5/3/5 over observed idle peak; raise for "
              "active-contact runs.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

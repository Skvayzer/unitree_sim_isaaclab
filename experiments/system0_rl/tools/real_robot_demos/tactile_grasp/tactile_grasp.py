#!/usr/bin/env python3
"""Step D — tactile-triggered compliant grasp on the Unitree G1 Dex3-1.

State machine:
    INIT -> READY -> WAIT_CONTACT -> CLOSING -> HOLDING -> RELEASING -> DONE
                                       \\
                                        EMERGENCY_RELEASE -> DONE

Per-finger torque caps freeze each joint independently during CLOSING so
the grasp wraps around objects of arbitrary shape without overloading any
single finger. EMERGENCY_TAU triggers an immediate soft release.

CLI:
    python tactile_grasp.py --iface eth0 --side right \\
                            --poses poses.json [--no-tactile] [--dry-run] \\
                            [--hold-sec 5.0]

See README.md for safety preconditions and testing protocol.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import config


# ---------------------------------------------------------------------------
# SDK
# ---------------------------------------------------------------------------

def import_sdk():
    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # type: ignore
            HandCmd_, HandState_,
        )
        from unitree_sdk2py.idl.default import (  # type: ignore
            unitree_hg_msg_dds__HandCmd_,
        )
    except ImportError as e:
        print(f"[ERROR] {e}\n\nRun sdk_check.py first.", file=sys.stderr)
        sys.exit(2)
    return (ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
            HandCmd_, HandState_, unitree_hg_msg_dds__HandCmd_)


def ris_mode(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
    """Pack the RIS mode byte for a Dex3 motor (id|status<<4|timeout<<7)."""
    return ((motor_id & 0x0F)
            | ((status & 0x07) << 4)
            | ((timeout & 0x01) << 7))


# ---------------------------------------------------------------------------
# Thread-safe state slot
# ---------------------------------------------------------------------------

class StateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None

    def on_message(self, sample) -> None:
        with self._lock:
            self._latest = sample

    def latest(self):
        with self._lock:
            return self._latest


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    INIT = "INIT"
    READY = "READY"
    WAIT_CONTACT = "WAIT_CONTACT"
    CLOSING = "CLOSING"
    HOLDING = "HOLDING"
    RELEASING = "RELEASING"
    EMERGENCY_RELEASE = "EMERGENCY_RELEASE"
    DONE = "DONE"


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    result: str = "UNKNOWN"          # GRASPED / NO_CONTACT_TIMEOUT / EMERGENCY_RELEASE / USER_ABORT
    contact_t: Optional[float] = None  # seconds from session start
    close_t: Optional[float] = None    # CLOSING duration
    frozen_joints: list[int] = field(default_factory=list)
    peak_tau: float = 0.0
    emergency: bool = False

    def line(self) -> str:
        ct = f"{self.contact_t:.2f}s" if self.contact_t is not None else "—"
        kt = f"{self.close_t:.2f}s" if self.close_t is not None else "—"
        return (f"RUN_SUMMARY  result={self.result}  contact_t={ct}  "
                f"close_t={kt}  frozen_joints={self.frozen_joints}  "
                f"peak_|Δtau|={self.peak_tau:.0f} raw  "
                f"emergency={self.emergency}")


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_poses(path: Path, expected_side: str, n_joints: int):
    if not path.exists():
        print(f"[ERROR] poses file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        data = json.load(f)
    if data.get("side") != expected_side:
        print(
            f"[ERROR] poses.json side='{data.get('side')}', "
            f"requested side='{expected_side}'.",
            file=sys.stderr,
        )
        sys.exit(2)
    p_open = list(data["pose_open"])
    p_closed = list(data["pose_closed"])
    if len(p_open) != n_joints or len(p_closed) != n_joints:
        print(
            f"[ERROR] poses.json has {len(p_open)} joints, expected "
            f"{n_joints}.",
            file=sys.stderr,
        )
        sys.exit(2)
    for label, pose in (("open", p_open), ("closed", p_closed)):
        for i, q in enumerate(pose):
            if not (-math.pi <= q <= math.pi):
                print(f"[ERROR] poses.json {label}[{i}]={q} out of range",
                      file=sys.stderr)
                sys.exit(2)
    return p_open, p_closed


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

def detect_contact(
    sample,
    tau_baseline: list[float],
    use_tactile: bool,
    tactile_baseline: Optional[list[float]],
) -> tuple[bool, list[float], float]:
    """Return (any_contact, per_finger_load, tactile_max_delta).

    per_finger_load is a 7-vector of |tau_est[i] - tau_baseline[i]|, used
    by the per-joint freeze logic during CLOSING regardless of which signal
    fires the trigger.
    """
    motors = sample.motor_state
    per_load = [abs(float(motors[i].tau_est) - tau_baseline[i])
                for i in range(len(tau_baseline))]

    tactile_delta_max = 0.0
    tactile_hit = False
    if use_tactile and tactile_baseline is not None:
        press = getattr(sample, "press_sensor_state", None)
        if press is not None and len(press) > 0:
            # Real firmware reports tap deltas with both signs depending on
            # the pad (e.g. thumb tap reads negative, palm tap positive on
            # the unit calibrated 2026-04-30). Use |Δ| so direction is
            # irrelevant.
            for i, p in enumerate(press):
                if i >= len(tactile_baseline):
                    break
                s = float(sum(p.pressure))
                d = abs(s - tactile_baseline[i])
                if d > tactile_delta_max:
                    tactile_delta_max = d
            if tactile_delta_max > config.TACTILE_TRIGGER:
                tactile_hit = True

    torque_hit = any(L > config.CONTACT_TAU for L in per_load)
    return (tactile_hit or torque_hit, per_load, tactile_delta_max)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GraspController:
    def __init__(self, args, sym):
        self.args = args
        self.sym = sym
        self.shutdown = False  # set by signal handler

        # Resolve side-specific config.
        if args.side == "right":
            self.topic_state = config.TOPIC_RIGHT_STATE
            self.topic_cmd = config.TOPIC_RIGHT_CMD
            self.finger_names = list(config.RIGHT_FINGER_NAMES)
        else:
            self.topic_state = config.TOPIC_LEFT_STATE
            self.topic_cmd = config.TOPIC_LEFT_CMD
            self.finger_names = list(config.LEFT_FINGER_NAMES)

        # SDK channels.
        sym["ChannelFactoryInitialize"](0, args.iface)
        self.cache = StateCache()
        self.sub = sym["ChannelSubscriber"](self.topic_state, sym["HandState_"])
        self.sub.Init(self.cache.on_message, 0)
        self.pub = sym["ChannelPublisher"](self.topic_cmd, sym["HandCmd_"])
        self.pub.Init()

        # Wait for first state.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and self.cache.latest() is None:
            time.sleep(0.05)
        s = self.cache.latest()
        if s is None:
            raise RuntimeError(f"no state on {self.topic_state}")
        self.n = len(s.motor_state)
        if self.n != config.NUM_FINGER_JOINTS:
            print(f"[WARN] {self.n} motors reported; config expects "
                  f"{config.NUM_FINGER_JOINTS}")

        # Poses.
        p_open, p_closed = load_poses(
            Path(args.poses), args.side, self.n
        )
        self.pose_open = p_open
        self.pose_closed = p_closed

        # Tactile.
        self.use_tactile = (not args.no_tactile) and (
            getattr(s, "press_sensor_state", None) is not None
            and len(s.press_sensor_state) > 0
        )

        # Mutable state.
        self.q_cmd_prev = list([float(m.q) for m in s.motor_state])
        self.tau_baseline = [0.0] * self.n
        self.tactile_baseline: Optional[list[float]] = None
        self.frozen = [False] * self.n
        self.q_freeze = [0.0] * self.n
        self.contact_streak = 0
        self.stop_streaks = [0] * self.n

        # Accumulators for the READY-phase re-baseline. The pre-INIT idle
        # baseline does not include the gravity / holding torques the
        # joints carry while KP_READY holds the hand at pose_open, so we
        # average a fresh window during READY and overwrite tau_baseline
        # before WAIT_CONTACT begins.
        self._ready_tau_acc = [0.0] * self.n
        self._ready_n = 0

        # Logging.
        self.t0 = time.monotonic()
        self.log_path = self._open_log()
        self.log_writer = csv.writer(self.log_path) if self.log_path else None
        if self.log_writer is not None:
            self.log_writer.writerow([
                "t", "state",
                *[f"q{i}" for i in range(self.n)],
                *[f"q_cmd{i}" for i in range(self.n)],
                *[f"tau{i}" for i in range(self.n)],
                "tactile_max",
            ])

        self.summary = RunSummary()
        self._t_contact: Optional[float] = None
        self._t_close_start: Optional[float] = None

    def _open_log(self):
        try:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            return open(log_dir / f"run_{stamp}.csv", "w", newline="")
        except Exception as e:
            print(f"[WARN] could not open log file: {e}", file=sys.stderr)
            return None

    # ----------- publishing -----------

    def publish_cmd(self, q_target: list[float], kp: float, kd: float) -> list[float]:
        """Apply slew limit, optionally publish, return the actual cmd sent."""
        q_cmd: list[float] = []
        for i in range(self.n):
            tgt = q_target[i]
            prev = self.q_cmd_prev[i]
            lo = prev - config.MAX_DQ_PER_TICK
            hi = prev + config.MAX_DQ_PER_TICK
            q_cmd.append(max(lo, min(hi, tgt)))
        self.q_cmd_prev = q_cmd

        if self.args.dry_run:
            return q_cmd

        msg = self.sym["make_handcmd"]()
        for i in range(self.n):
            msg.motor_cmd[i].mode = ris_mode(i)
            msg.motor_cmd[i].q = float(q_cmd[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = float(kp)
            msg.motor_cmd[i].kd = float(kd)
        try:
            self.pub.Write(msg)
        except Exception as e:
            print(f"[WARN] publish: {e}", file=sys.stderr)
        return q_cmd

    # ----------- baselines -----------

    def capture_baselines(self) -> None:
        """Average tau and tactile pad sums over BASELINE_WINDOW_TICKS."""
        ticks = config.BASELINE_WINDOW_TICKS
        tau_acc = [0.0] * self.n
        tac_acc: Optional[list[float]] = None
        n = 0
        period = config.DT_SEC
        next_tick = time.monotonic()
        for _ in range(ticks):
            if self.shutdown:
                break
            s = self.cache.latest()
            if s is not None:
                for i in range(self.n):
                    tau_acc[i] += float(s.motor_state[i].tau_est)
                if self.use_tactile:
                    press = getattr(s, "press_sensor_state", None)
                    if press is not None and len(press) > 0:
                        sums = [float(sum(p.pressure)) for p in press]
                        if tac_acc is None:
                            tac_acc = list(sums)
                        else:
                            for i in range(min(len(tac_acc), len(sums))):
                                tac_acc[i] += sums[i]
                n += 1
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        if n > 0:
            self.tau_baseline = [v / n for v in tau_acc]
            if tac_acc is not None:
                self.tactile_baseline = [v / n for v in tac_acc]

    # ----------- emergency check -----------

    def check_emergency(self, sample) -> bool:
        # Compare |Δtau - baseline|, not absolute tau. The Dex3 firmware
        # reports tau_est in raw sensor units with a large nonzero idle
        # offset (e.g. thumb_0 idles at ±5861 raw on this unit), so an
        # absolute-value check would fire before the hand even moves.
        for i in range(self.n):
            base = self.tau_baseline[i] if i < len(self.tau_baseline) else 0.0
            if abs(float(sample.motor_state[i].tau_est) - base) > config.EMERGENCY_TAU:
                return True
        return False

    # ----------- main loop -----------

    def run(self) -> RunSummary:
        period = config.DT_SEC

        # Capture baseline BEFORE we publish any commands. The Dex3 firmware
        # reports tau_est in raw units with a large idle offset per joint,
        # so the emergency check (and STOP/CONTACT thresholds) all need a
        # baseline to subtract. Capturing during READY is too late — INIT
        # would publish without a baseline and instantly hit the emergency
        # ceiling on idle drift alone.
        print(f"[..] capturing tau/tactile baseline before INIT "
              f"(hand still, no contact)...")
        self.capture_baselines()
        print(f"[..] tau_baseline = "
              + ", ".join(f'{v:+.1f}' for v in self.tau_baseline))
        if self.use_tactile and self.tactile_baseline is not None:
            print(f"[..] tactile_baseline = "
                  + ", ".join(f'{v:.0f}' for v in self.tactile_baseline))
        self._baseline_pre_captured = True

        next_tick = time.perf_counter()

        state = State.INIT
        # INIT phase: linear ramp from current q to pose_open over T_INIT_RAMP.
        s_now = self.cache.latest()
        q_init = [float(m.q) for m in s_now.motor_state]
        self.q_cmd_prev = list(q_init)

        t_state_entered = time.monotonic()
        t_hold_start: Optional[float] = None

        while True:
            now = time.perf_counter()
            if now < next_tick:
                # Busy-wait so we don't drift under load. The remaining
                # time is short (<10 ms) so a brief sleep is cheap.
                slack = next_tick - now
                if slack > 0.001:
                    time.sleep(slack - 0.0005)
                continue
            next_tick += period
            t = time.monotonic() - self.t0

            if t > config.MAX_SESSION_SEC and state != State.EMERGENCY_RELEASE:
                print(f"[..] session timeout ({config.MAX_SESSION_SEC}s) — "
                      f"emergency release.")
                state = State.EMERGENCY_RELEASE
                self.summary.emergency = True
                t_state_entered = time.monotonic()

            if self.shutdown and state not in (
                State.EMERGENCY_RELEASE, State.RELEASING, State.DONE
            ):
                print("[..] shutdown requested — soft release.")
                state = State.EMERGENCY_RELEASE
                self.summary.emergency = True
                self.summary.result = "USER_ABORT"
                t_state_entered = time.monotonic()

            sample = self.cache.latest()
            if sample is None:
                continue

            # Track peak |Δtau| across the run for the summary. Δ-from-
            # baseline matches what STOP_TAU / EMERGENCY_TAU mean.
            for i, m in enumerate(sample.motor_state):
                base = (self.tau_baseline[i]
                        if i < len(self.tau_baseline) else 0.0)
                a = abs(float(m.tau_est) - base)
                if a > self.summary.peak_tau:
                    self.summary.peak_tau = a

            # Emergency torque check (any state except DONE).
            if state not in (State.EMERGENCY_RELEASE, State.DONE):
                if self.check_emergency(sample):
                    print(
                        "[!!!] EMERGENCY torque exceeded — soft release."
                    )
                    state = State.EMERGENCY_RELEASE
                    self.summary.emergency = True
                    self.summary.result = "EMERGENCY_RELEASE"
                    t_state_entered = time.monotonic()

            q_now = [float(m.q) for m in sample.motor_state]
            # In dry-run nothing is published, so the hand never moves and
            # the arrival checks (q_now vs pose_open / pose_closed) never
            # pass — INIT would never advance to READY, so WAIT_CONTACT and
            # the contact-detection path are unreachable on the bench.
            # Substitute the commanded trajectory as a perfect tracker so
            # we can exercise the rest of the state machine. Real runs
            # always compare against measured q_now.
            q_eff = list(self.q_cmd_prev) if self.args.dry_run else q_now
            q_target = list(self.q_cmd_prev)
            kp, kd = config.KP_READY, config.KD_READY

            # ---- per-state logic ----
            if state == State.INIT:
                kp, kd = config.KP_READY, config.KD_READY
                elapsed = time.monotonic() - t_state_entered
                alpha = min(1.0, elapsed / config.T_INIT_RAMP)
                q_target = [q_init[i] + alpha * (self.pose_open[i] - q_init[i])
                            for i in range(self.n)]
                init_elapsed = time.monotonic() - t_state_entered
                init_arrived = (alpha >= 1.0
                                and all(abs(q_eff[i] - self.pose_open[i])
                                        < config.ARRIVAL_TOL_RAD
                                        for i in range(self.n)))
                init_timed_out = init_elapsed >= config.T_INIT_MAX
                if init_arrived or init_timed_out:
                    print(f"[t={t:5.2f}s] INIT -> READY  "
                          f"(arrived={init_arrived}, "
                          f"timed_out={init_timed_out})")
                    state = State.READY
                    t_state_entered = time.monotonic()

            elif state == State.READY:
                kp, kd = config.KP_READY, config.KD_READY
                q_target = list(self.pose_open)
                # Accumulate samples for the holding-pose re-baseline. See
                # __init__ for why this matters: holding torques at
                # pose_open swamp the pre-INIT idle baseline on some joints,
                # which would otherwise fire spurious CONTACT / STOP / and
                # (briefly observed) EMERGENCY trips immediately on entry
                # to WAIT_CONTACT and CLOSING.
                for i in range(self.n):
                    self._ready_tau_acc[i] += float(sample.motor_state[i].tau_est)
                self._ready_n += 1
                if time.monotonic() - t_state_entered >= 0.5:
                    if self._ready_n > 0:
                        new_base = [v / self._ready_n
                                    for v in self._ready_tau_acc]
                        print("[..] re-baseline at READY (holding "
                              "pose_open):")
                        print("     tau_baseline_holding = "
                              + ", ".join(f"{v:+.1f}" for v in new_base))
                        self.tau_baseline = new_base
                    print(f"[t={t:5.2f}s] READY -> WAIT_CONTACT  "
                          f"(tactile={self.use_tactile})")
                    print("    Place an object in the palm.")
                    state = State.WAIT_CONTACT
                    t_state_entered = time.monotonic()
                    self.contact_streak = 0

            elif state == State.WAIT_CONTACT:
                kp, kd = config.KP_READY, config.KD_READY
                q_target = list(self.pose_open)
                hit, _, _ = detect_contact(
                    sample, self.tau_baseline,
                    self.use_tactile, self.tactile_baseline,
                )
                if hit:
                    self.contact_streak += 1
                else:
                    self.contact_streak = 0
                if self.contact_streak >= config.CONTACT_DEBOUNCE:
                    print(f"[t={t:5.2f}s] CONTACT — WAIT_CONTACT -> CLOSING")
                    state = State.CLOSING
                    t_state_entered = time.monotonic()
                    self._t_contact = t
                    self.summary.contact_t = t
                    self._t_close_start = t
                    self.frozen = [False] * self.n
                    self.q_freeze = list(q_now)
                    self.stop_streaks = [0] * self.n
                elif (time.monotonic() - t_state_entered
                      > config.WAIT_CONTACT_TIMEOUT):
                    print(f"[t={t:5.2f}s] WAIT_CONTACT timed out — releasing.")
                    self.summary.result = "NO_CONTACT_TIMEOUT"
                    state = State.RELEASING
                    t_state_entered = time.monotonic()

            elif state == State.CLOSING:
                kp, kd = config.KP_CLOSING, config.KD_CLOSING
                # Per-joint freeze on torque excursion.
                for i in range(self.n):
                    if self.frozen[i]:
                        continue
                    load_i = abs(float(sample.motor_state[i].tau_est)
                                 - self.tau_baseline[i])
                    if load_i > config.STOP_TAU:
                        self.stop_streaks[i] += 1
                    else:
                        self.stop_streaks[i] = 0
                    if self.stop_streaks[i] >= config.STOP_DEBOUNCE:
                        self.frozen[i] = True
                        self.q_freeze[i] = q_now[i]
                        if i not in self.summary.frozen_joints:
                            self.summary.frozen_joints.append(i)
                        print(f"[t={t:5.2f}s]   freeze joint {i} "
                              f"({self.finger_names[i]}) at q={q_now[i]:+.3f}")

                # Build per-joint target.
                for i in range(self.n):
                    if self.frozen[i]:
                        q_target[i] = self.q_freeze[i]
                    else:
                        q_target[i] = self.pose_closed[i]

                all_frozen = all(self.frozen)
                arrived = all(abs(q_eff[i] - self.pose_closed[i])
                              < config.CLOSING_DONE_TOL_RAD
                              for i in range(self.n))
                closing_elapsed = time.monotonic() - t_state_entered
                timed_out = closing_elapsed >= config.T_CLOSING_MAX
                if all_frozen or arrived or timed_out:
                    print(f"[t={t:5.2f}s] CLOSING -> HOLDING  "
                          f"(frozen={self.summary.frozen_joints}, "
                          f"arrived={arrived}, timed_out={timed_out})")
                    if self._t_close_start is not None:
                        self.summary.close_t = t - self._t_close_start
                    # Snapshot the freeze targets so HOLDING uses the actual
                    # contact pose for any joints that arrived (not frozen).
                    for i in range(self.n):
                        if not self.frozen[i]:
                            self.q_freeze[i] = q_now[i]
                    state = State.HOLDING
                    t_state_entered = time.monotonic()
                    t_hold_start = t_state_entered

            elif state == State.HOLDING:
                kp, kd = config.KP_HOLD, config.KD_HOLD
                q_target = list(self.q_freeze)
                if t_hold_start is not None and (
                    time.monotonic() - t_hold_start >= self.args.hold_sec
                ):
                    print(f"[t={t:5.2f}s] HOLDING -> RELEASING")
                    if self.summary.result == "UNKNOWN":
                        self.summary.result = "GRASPED"
                    state = State.RELEASING
                    t_state_entered = time.monotonic()

            elif state == State.RELEASING:
                kp, kd = config.KP_RELEASE, config.KD_RELEASE
                elapsed = time.monotonic() - t_state_entered
                alpha = min(1.0, elapsed / config.T_RELEASE_RAMP)
                q_start = self.q_cmd_prev  # ramp from current cmd
                q_target = [q_start[i]
                            + alpha * (self.pose_open[i] - q_start[i])
                            for i in range(self.n)]
                if alpha >= 1.0:
                    print(f"[t={t:5.2f}s] RELEASING -> DONE")
                    state = State.DONE

            elif state == State.EMERGENCY_RELEASE:
                # Low stiffness so we don't yank against an object.
                kp, kd = config.KP_RELEASE, config.KD_RELEASE
                elapsed = time.monotonic() - t_state_entered
                alpha = min(1.0, elapsed / config.T_RELEASE_RAMP)
                q_start_em = list(q_now)  # release from where we are
                q_target = [q_start_em[i]
                            + alpha * (self.pose_open[i] - q_start_em[i])
                            for i in range(self.n)]
                if (elapsed >= config.T_RELEASE_RAMP
                    or elapsed >= config.EMERGENCY_RELEASE_TIMEOUT):
                    print(f"[t={t:5.2f}s] EMERGENCY_RELEASE complete — DONE")
                    if self.summary.result == "UNKNOWN":
                        self.summary.result = "EMERGENCY_RELEASE"
                    state = State.DONE

            elif state == State.DONE:
                break

            q_cmd = self.publish_cmd(q_target, kp, kd)

            # Log row.
            if self.log_writer is not None:
                tactile_max = 0.0
                if self.use_tactile and self.tactile_baseline is not None:
                    press = getattr(sample, "press_sensor_state", None)
                    if press is not None and len(press) > 0:
                        sums = [float(sum(p.pressure)) for p in press]
                        for i, s_i in enumerate(sums):
                            if i < len(self.tactile_baseline):
                                d = abs(s_i - self.tactile_baseline[i])
                                if d > tactile_max:
                                    tactile_max = d
                row = [f"{t:.4f}", state.value]
                row += [f"{v:.4f}" for v in q_now]
                row += [f"{v:.4f}" for v in q_cmd]
                row += [f"{float(sample.motor_state[i].tau_est):.4f}"
                        for i in range(self.n)]
                row.append(f"{tactile_max:.1f}")
                self.log_writer.writerow(row)

        return self.summary

    def close(self) -> None:
        try:
            self.sub.Close()
        except Exception:
            pass
        if self.log_path is not None:
            try:
                self.log_path.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tactile-triggered Dex3 grasp demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0")
    p.add_argument("--side", choices=("left", "right"), default="right")
    p.add_argument(
        "--poses",
        default=str(Path(__file__).parent / "poses.json"),
        help="Path to poses.json from calibrate_poses.py.",
    )
    p.add_argument("--no-tactile", action="store_true",
                   help="Use joint torque only for contact detection.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the state machine but never publish HandCmd.")
    p.add_argument("--hold-sec", type=float, default=config.T_HOLD,
                   help="HOLDING duration before automatic release.")
    return p.parse_args()


def main() -> int:
    config.assert_safety_caps()
    args = parse_args()
    (CFI, CPub, CSub, HandCmd_, HandState_, make_cmd) = import_sdk()
    sym = {
        "ChannelFactoryInitialize": CFI,
        "ChannelPublisher": CPub,
        "ChannelSubscriber": CSub,
        "HandCmd_": HandCmd_,
        "HandState_": HandState_,
        "make_handcmd": make_cmd,
    }

    if args.dry_run:
        print("[..] DRY RUN — no HandCmd will be published.")
    print(f"[..] iface={args.iface}  side={args.side}  "
          f"poses={args.poses}  hold={args.hold_sec}s  "
          f"tactile={'off (--no-tactile)' if args.no_tactile else 'auto'}")

    ctrl = GraspController(args, sym)

    def on_sig(signum, frame):
        print(f"\n[..] caught signal {signum}; flagging shutdown.")
        ctrl.shutdown = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    try:
        summary = ctrl.run()
    finally:
        ctrl.close()

    print()
    print(summary.line())
    return 0 if summary.result in ("GRASPED", "USER_ABORT",
                                   "NO_CONTACT_TIMEOUT") else 1


if __name__ == "__main__":
    sys.exit(main())

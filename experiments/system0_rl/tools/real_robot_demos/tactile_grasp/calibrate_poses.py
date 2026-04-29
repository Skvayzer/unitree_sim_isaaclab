#!/usr/bin/env python3
"""Step C — capture the OPEN and CLOSED joint poses and save to poses.json.

Joint angles vary across Dex3 units and mounts, so we record them per-robot
instead of hard-coding. The script:

  1. Publishes a kp=0, kd=0 hold (back-drivable) so the operator can move
     the hand by hand.
  2. Prompts for the OPEN palm-up pose, captures mean q over 1.0 s.
  3. Prompts for the CLOSED grasp pose, captures mean q over 1.0 s.
  4. Sanity-checks and writes to poses.json.
  5. On exit, holds the current q for ~1 s with low stiffness so the hand
     does not drop when the back-drive command stream stops.

The kp/kd cap from config.py is enforced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import config


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
    return ((motor_id & 0x0F)
            | ((status & 0x07) << 4)
            | ((timeout & 0x01) << 7))


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


def build_cmd(make_msg, n: int, q: list[float],
              kp: float, kd: float):
    msg = make_msg()
    for i in range(n):
        msg.motor_cmd[i].mode = ris_mode(i)
        msg.motor_cmd[i].q = float(q[i])
        msg.motor_cmd[i].dq = 0.0
        msg.motor_cmd[i].tau = 0.0
        msg.motor_cmd[i].kp = float(kp)
        msg.motor_cmd[i].kd = float(kd)
    return msg


def read_q(state) -> list[float]:
    return [float(m.q) for m in state.motor_state]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record OPEN and CLOSED Dex3 hand poses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0")
    p.add_argument("--side", choices=("left", "right"), default="right")
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "poses.json"),
        help="Where to write the JSON file.",
    )
    p.add_argument(
        "--capture-sec",
        type=float, default=1.0,
        help="Duration over which to average q for each pose.",
    )
    return p.parse_args()


def main() -> int:
    config.assert_safety_caps()
    args = parse_args()
    (ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
     HandCmd_, HandState_, make_handcmd) = import_sdk()

    if args.side == "right":
        topic_state = config.TOPIC_RIGHT_STATE
        topic_cmd = config.TOPIC_RIGHT_CMD
        names = list(config.RIGHT_FINGER_NAMES)
    else:
        topic_state = config.TOPIC_LEFT_STATE
        topic_cmd = config.TOPIC_LEFT_CMD
        names = list(config.LEFT_FINGER_NAMES)

    ChannelFactoryInitialize(0, args.iface)
    cache = StateCache()
    sub = ChannelSubscriber(topic_state, HandState_)
    sub.Init(cache.on_message, 0)

    pub = ChannelPublisher(topic_cmd, HandCmd_)
    pub.Init()

    # Wait for first state.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and cache.latest() is None:
        time.sleep(0.05)
    state = cache.latest()
    if state is None:
        print(f"[ERROR] no state on {topic_state}", file=sys.stderr)
        return 3
    n = len(state.motor_state)
    if n != config.NUM_FINGER_JOINTS:
        print(
            f"[WARN] state reports {n} motors, config expects "
            f"{config.NUM_FINGER_JOINTS}. Using {n}."
        )

    # Background thread: stream a back-drive command (kp=0, kd=0) at 100 Hz
    # so the controller stays alive but the hand is fully limp.
    stop_flag = threading.Event()

    def backdrive_loop() -> None:
        period = config.DT_SEC
        next_tick = time.monotonic()
        while not stop_flag.is_set():
            s = cache.latest()
            if s is None:
                time.sleep(period); continue
            q_now = read_q(s)
            msg = build_cmd(make_handcmd, len(q_now), q_now,
                            config.KP_LIMP, config.KD_LIMP)
            try:
                pub.Write(msg)
            except Exception as e:
                print(f"[WARN] publish failed: {e}", file=sys.stderr)
            now = time.monotonic()
            next_tick += period
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = now

    bg = threading.Thread(target=backdrive_loop, daemon=True)
    bg.start()

    # Graceful exit: ramp a soft hold, then stop the background thread.
    def soft_hold_and_exit():
        try:
            s = cache.latest()
            if s is not None:
                q_now = read_q(s)
                msg = build_cmd(make_handcmd, len(q_now), q_now,
                                config.KP_READY, config.KD_READY)
                t_end = time.monotonic() + 1.0
                while time.monotonic() < t_end:
                    pub.Write(msg)
                    time.sleep(config.DT_SEC)
        except Exception as e:
            print(f"[WARN] soft hold on exit: {e}", file=sys.stderr)
        finally:
            stop_flag.set()

    def on_signal(signum, frame):
        print(f"\n[..] caught signal {signum}; soft hold + exit.")
        soft_hold_and_exit()
        sys.exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def capture_pose(label: str) -> list[float]:
        print(f"\n>>> {label}")
        input("    Move the hand into the pose, then press Enter to capture: ")
        print(f"[..] averaging q over {args.capture_sec:.2f}s...")
        t_end = time.monotonic() + args.capture_sec
        acc: Optional[list[float]] = None
        count = 0
        while time.monotonic() < t_end:
            s = cache.latest()
            if s is None:
                time.sleep(0.005); continue
            q = read_q(s)
            if acc is None:
                acc = list(q)
            else:
                for i in range(min(len(acc), len(q))):
                    acc[i] += q[i]
            count += 1
            time.sleep(0.01)
        if acc is None or count == 0:
            print("[ERROR] no samples captured", file=sys.stderr)
            sys.exit(4)
        mean = [v / count for v in acc]
        print(f"[OK] captured ({count} samples): "
              + ", ".join(f"{names[i] if i < len(names) else f'#{i}'}="
                          f"{mean[i]:+.3f}" for i in range(len(mean))))
        return mean

    try:
        print("Hand is now back-drivable (kp=0, kd=0).")
        print(f"Side: {args.side}.  Joint names: {names}")
        pose_open = capture_pose(
            "OPEN POSE — set the hand palm-up, fingers gently extended, "
            "ready to receive an object."
        )
        pose_closed = capture_pose(
            "CLOSED POSE — close the hand around your thumb in a typical "
            "cylindrical grasp (do NOT crush)."
        )

        # Sanity checks.
        for label, pose in (("open", pose_open), ("closed", pose_closed)):
            for i, q in enumerate(pose):
                if not (-math.pi <= q <= math.pi):
                    print(f"[ERROR] {label}[{i}]={q:.3f} out of [-π, π]",
                          file=sys.stderr)
                    return 5
        diffs = [abs(pose_open[i] - pose_closed[i])
                 for i in range(len(pose_open))]
        n_distinct = sum(1 for d in diffs if d > 0.05)
        if n_distinct < 4:
            print(
                f"[ERROR] open and closed differ on only {n_distinct} of "
                f"{len(diffs)} joints (per-joint diffs: "
                f"{[f'{d:.3f}' for d in diffs]}). Expected at least 4. "
                f"Re-run.",
                file=sys.stderr,
            )
            return 6

        out = {
            "side": args.side,
            "joint_names": names[:len(pose_open)],
            "pose_open": pose_open,
            "pose_closed": pose_closed,
            "captured_at": dt.datetime.now().astimezone().isoformat(),
        }
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[OK] wrote {out_path}")
    finally:
        soft_hold_and_exit()
        try:
            sub.Close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

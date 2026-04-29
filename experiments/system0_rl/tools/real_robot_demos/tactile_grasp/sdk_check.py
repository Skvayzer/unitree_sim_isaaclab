#!/usr/bin/env python3
"""Step A — verify the Unitree SDK and Dex3 DDS topics on a real G1.

READ-ONLY: never publishes to any DDS topic.

Outcome on success:
    * Imports `unitree_sdk2py` and prints its install path.
    * Resolves and prints the schemas of HandCmd_, HandState_, MotorCmd_,
      MotorState_, and (if present) PressSensorState_.
    * Initialises the DDS factory on the requested interface.
    * Confirms `rt/dex3/right/state` (and optionally left) is alive within
      2 seconds and prints the first frame's q / dq / tau_est.
    * Detects whether tactile data is available and on which channel.
    * Exits 0.

Anything else exits non-zero with a clear error.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import fields, is_dataclass
from typing import Any, Optional


SDK_INSTALL_HINT = """\
The Unitree SDK could not be imported.

Install the official Python SDK:
    git clone https://github.com/unitreerobotics/unitree_sdk2_python
    cd unitree_sdk2_python
    pip install -e .

Or, if a system install is already on this machine, activate the conda
env that has it (e.g. `conda activate unitree_py`) and re-run.
"""


def import_sdk() -> dict[str, Any]:
    """Import the SDK and return the symbols we need. Hard-exit on failure."""
    out: dict[str, Any] = {}
    try:
        import unitree_sdk2py  # type: ignore
    except ImportError as e:
        print(f"[ERROR] {e}\n\n{SDK_INSTALL_HINT}", file=sys.stderr)
        sys.exit(2)

    out["unitree_sdk2py"] = unitree_sdk2py
    print(f"[OK]   unitree_sdk2py imported from {unitree_sdk2py.__file__}")
    version = getattr(unitree_sdk2py, "__version__", "(no __version__ attr)")
    print(f"       version: {version}")

    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
    except ImportError as e:
        print(f"[ERROR] cannot import core.channel symbols: {e}", file=sys.stderr)
        sys.exit(2)
    out["ChannelFactoryInitialize"] = ChannelFactoryInitialize
    out["ChannelSubscriber"] = ChannelSubscriber

    try:
        import unitree_sdk2py.idl.unitree_hg.msg.dds_ as hg_dds  # type: ignore
    except ImportError as e:
        print(f"[ERROR] unitree_hg msg namespace not found: {e}", file=sys.stderr)
        sys.exit(2)
    out["hg_dds"] = hg_dds

    available = sorted(n for n in dir(hg_dds) if not n.startswith("_"))
    print(f"[OK]   unitree_hg.msg.dds_ exposes: {available}")

    for required in ("HandCmd_", "HandState_", "MotorCmd_", "MotorState_"):
        if not hasattr(hg_dds, required):
            print(
                f"[ERROR] required class {required!r} missing from "
                f"unitree_hg.msg.dds_",
                file=sys.stderr,
            )
            sys.exit(2)
        out[required] = getattr(hg_dds, required)

    out["PressSensorState_"] = getattr(hg_dds, "PressSensorState_", None)
    return out


def describe_dataclass(cls) -> str:
    if not is_dataclass(cls):
        return f"{cls.__name__} (not a dataclass — fields unknown)"
    rows = [f"{cls.__name__} fields:"]
    for f in fields(cls):
        rows.append(f"    {f.name}: {f.type}")
    return "\n".join(rows)


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


def wait_for_first(cache: StateCache, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample, _ = cache.latest()
        if sample is not None:
            return True
        time.sleep(0.05)
    return False


def report_first_frame(name: str, sample) -> None:
    """Print q, dq, tau_est for each motor in motor_state."""
    motors = list(sample.motor_state)
    print(f"[OK]   {name}: {len(motors)} motor entries")
    for i, m in enumerate(motors):
        print(
            f"       m{i}: q={float(m.q):+.4f}  "
            f"dq={float(m.dq):+.4f}  tau_est={float(m.tau_est):+.4f}"
        )


def report_tactile(name: str, sample) -> bool:
    """Return True iff this hand exposes a usable tactile array."""
    press = getattr(sample, "press_sensor_state", None)
    if press is None:
        print(f"[..] {name}: no press_sensor_state field on HandState_")
        return False
    pads = list(press)
    if not pads:
        print(f"[..] {name}: press_sensor_state is empty (length 0)")
        return False
    print(f"[OK]   {name}: press_sensor_state has {len(pads)} modules")
    sample_module = pads[0]
    sample_pressure = list(getattr(sample_module, "pressure", []))
    print(
        f"       module[0]: pressure has {len(sample_pressure)} cells, "
        f"first values = {sample_pressure[:6]}"
    )
    sums = [float(sum(p.pressure)) for p in pads]
    print(f"       per-module sums: {[f'{v:.0f}' for v in sums]}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Step A: verify Unitree SDK + Dex3 DDS topics. "
                     "READ-ONLY — never publishes."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iface",
        default="eth0",
        help="Network interface DDS will bind to (e.g. eth0, enp0s31f6).",
    )
    p.add_argument(
        "--no-left",
        action="store_true",
        help="Skip the left hand (right-only check is enough for the demo).",
    )
    p.add_argument(
        "--state-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for the first state message.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    syms = import_sdk()
    HandState_ = syms["HandState_"]
    HandCmd_ = syms["HandCmd_"]
    MotorState_ = syms["MotorState_"]
    MotorCmd_ = syms["MotorCmd_"]
    PressSensorState_ = syms.get("PressSensorState_")

    print()
    print(describe_dataclass(HandState_))
    print()
    print(describe_dataclass(MotorState_))
    print()
    print(describe_dataclass(HandCmd_))
    print()
    print(describe_dataclass(MotorCmd_))
    print()
    if PressSensorState_ is not None:
        print(describe_dataclass(PressSensorState_))
    else:
        print("PressSensorState_: not exposed by SDK")
    print()

    ChannelFactoryInitialize = syms["ChannelFactoryInitialize"]
    ChannelSubscriber = syms["ChannelSubscriber"]

    print(f"[..] ChannelFactoryInitialize(0, {args.iface!r})")
    try:
        ChannelFactoryInitialize(0, args.iface)
    except Exception as e:
        print(f"[ERROR] DDS init on iface={args.iface}: {e}", file=sys.stderr)
        print(
            "        Common fix: pass the right interface name "
            "(e.g. enp4s0). Run `ip -br link` to list candidates.",
            file=sys.stderr,
        )
        return 3
    print(f"[OK]   DDS initialized on iface={args.iface}")

    right_cache = StateCache()
    right_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
    right_sub.Init(right_cache.on_message, 0)

    left_cache: Optional[StateCache] = None
    left_sub = None
    if not args.no_left:
        left_cache = StateCache()
        left_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        left_sub.Init(left_cache.on_message, 0)

    exit_code = 0
    try:
        if not wait_for_first(right_cache, args.state_timeout):
            print(
                f"[ERROR] no message on rt/dex3/right/state within "
                f"{args.state_timeout}s",
                file=sys.stderr,
            )
            return 4

        right_sample, _ = right_cache.latest()
        print()
        print("=== Right hand ===")
        report_first_frame("rt/dex3/right/state", right_sample)
        right_tactile = report_tactile("rt/dex3/right/state", right_sample)

        left_tactile = False
        if left_cache is not None:
            print()
            print("=== Left hand ===")
            if wait_for_first(left_cache, args.state_timeout):
                left_sample, _ = left_cache.latest()
                report_first_frame("rt/dex3/left/state", left_sample)
                left_tactile = report_tactile("rt/dex3/left/state", left_sample)
            else:
                print(
                    "[WARN] no message on rt/dex3/left/state — left hand "
                    "may be powered off or not present. Demo only needs "
                    "the right hand."
                )

        print()
        print("=== Summary ===")
        print(f"  right state:    OK")
        print(f"  TACTILE_AVAILABLE_RIGHT = {right_tactile}")
        if left_cache is not None:
            print(f"  TACTILE_AVAILABLE_LEFT  = {left_tactile}")
        if not right_tactile:
            print(
                "  -> demo will fall back to joint torque for contact "
                "detection (--no-tactile would force this anyway)."
            )
        else:
            print(
                "  -> demo will use tactile pads as the primary contact "
                "trigger; per-joint freeze still uses joint torque."
            )
    finally:
        try:
            right_sub.Close()
        except Exception:
            pass
        if left_sub is not None:
            try:
                left_sub.Close()
            except Exception:
                pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

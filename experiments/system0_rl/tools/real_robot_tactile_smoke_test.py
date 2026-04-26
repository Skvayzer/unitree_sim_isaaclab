#!/usr/bin/env python3
"""Real-robot Dex3 tactile smoke test.

Subscribes (READ-ONLY) to ``rt/dex3/left/state`` and ``rt/dex3/right/state``
on a live Unitree G1 and streams per-module pressure readings to stdout so a
human can press each pad in turn and verify the SDK module index → physical
layout mapping documented in ``dev_diary/SYSTEM0_FACTS.md``.

Aggregation: per-module value = ``sum(press_sensor_state[i].pressure[0:12])``.
The IDL exposes raw 12-cell FSR readings only — there is no pre-aggregated
``force_sum`` field, so summing is the chosen aggregation. ``temperature``,
``lost`` and ``reserve`` fields exist on each module but are not displayed.

Read-only: never publishes to any DDS topic.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

# SDK module → physical pad mapping, established empirically on the real Dex3
# (2026-04-26) using this script. Mapping is **hand-specific** for the palm:
# m0-m5 are the six finger sensors (no thumb-proximal sensor exists), m6-m8
# are the three palm sensors enumerated in opposite directions on left vs.
# right because the palm sensor strip is wired mirrored.
#
# Palm naming convention (same on both hands):
#   palm_0 = next to middle-finger side of the palm
#   palm_1 = between fingers (centre)
#   palm_2 = at index-finger side of the palm
MODULE_LABELS: dict[str, tuple[str, ...]] = {
    "left": (
        "thumb_0", "thumb_1",
        "middle_0", "middle_1",
        "index_0", "index_1",
        "palm_2", "palm_1", "palm_0",   # m6,m7,m8: index→middle direction
    ),
    "right": (
        "thumb_0", "thumb_1",
        "middle_0", "middle_1",
        "index_0", "index_1",
        "palm_0", "palm_1", "palm_2",   # m6,m7,m8: middle→index direction
    ),
}

# Threshold (in raw ADC-count sum) above which a row is flagged as a "spike".
# Idle noise on the press_sensor_state[i].pressure sum is ~20-30 ADC counts,
# so 100 cleanly separates a finger touch from background.
SPIKE_THRESHOLD = 100.0
# Time (seconds) a recent spike persists in the "Recent spike" display.
SPIKE_MEMORY_SECONDS = 5.0


def _import_sdk():
    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_  # type: ignore
    except ImportError as e:
        print(
            "[ERROR] unitree_sdk2py not importable: "
            f"{e}\n"
            "Activate the conda env that has the SDK installed "
            "(e.g. `conda activate unitree_py`) before running.",
            file=sys.stderr,
        )
        sys.exit(2)
    return ChannelFactoryInitialize, ChannelSubscriber, HandState_


class HandStateCache:
    """Thread-safe holder for the latest sample from one DDS subscriber."""

    def __init__(self, name: str):
        self.name = name
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


def module_sums(sample) -> list[float]:
    """Return per-module pressure sums (sum of 12 cells per module)."""
    out: list[float] = []
    for module in sample.press_sensor_state:
        out.append(float(sum(module.pressure)))
    return out


PAD_W = 8


def label_for(idx: int, side: str) -> str:
    labels = MODULE_LABELS.get(side, ())
    if 0 <= idx < len(labels):
        return labels[idx]
    return f"#{idx}"


def _format_cell(
    pad: str,
    raw: Optional[float],
    delta: Optional[float],
    bar_unit: float,
    bar_max: int,
    is_spike: bool,
) -> str:
    spike = "●" if is_spike else " "
    if raw is None or delta is None:
        return f"{spike} {pad:<{PAD_W}}  {'—':>8}  {'—':>8}  {'':<{bar_max}}"
    sign = "+" if delta >= 0 else "-"
    n_bar = min(bar_max, max(0, int(abs(delta) / bar_unit)))
    bar = ("▌" * n_bar) if delta > 0 else (" " * n_bar)
    return f"{spike} {pad:<{PAD_W}}  {raw:8.0f}  {sign}{abs(delta):7.0f}  {bar:<{bar_max}}"


def _argmax_delta(deltas: list[Optional[float]], threshold: float) -> Optional[int]:
    """Return module index with the largest |Δ|, or None if all below threshold."""
    best_idx, best_abs = None, threshold
    for i, d in enumerate(deltas):
        if d is None:
            continue
        a = abs(d)
        if a > best_abs:
            best_abs = a
            best_idx = i
    return best_idx


class SpikeTracker:
    """Remembers the most recent spike per hand for ``SPIKE_MEMORY_SECONDS``."""

    def __init__(self):
        self.left: Optional[tuple[int, float, float]] = None  # (idx, delta, t)
        self.right: Optional[tuple[int, float, float]] = None

    def update(self, side: str, idx: Optional[int], delta: Optional[float], now: float) -> None:
        if idx is None or delta is None:
            return
        attr = side  # "left" or "right"
        prev = getattr(self, attr)
        if prev is None or now - prev[2] > SPIKE_MEMORY_SECONDS or abs(delta) > abs(prev[1]):
            setattr(self, attr, (idx, delta, now))

    def get(self, side: str, now: float) -> Optional[tuple[int, float, float]]:
        prev = getattr(self, side)
        if prev is None or now - prev[2] > SPIKE_MEMORY_SECONDS:
            return None
        return prev


def render_panel(
    t: float,
    now: float,
    left_vals: list[float],
    right_vals: list[float],
    left_base: list[float],
    right_base: list[float],
    l_count: int,
    r_count: int,
    spikes: SpikeTracker,
    bar_unit: float = 200.0,
    bar_max: int = 10,
) -> str:
    """Static multi-line panel — ANSI 'home + clear-down' so it redraws in place.
    The per-hand row with the largest |Δ| above SPIKE_THRESHOLD gets a ● marker
    next to its pad name."""
    n = max(len(left_vals), len(right_vals), len(MODULE_LABELS["left"]))
    l_deltas: list[Optional[float]] = [
        (left_vals[i] - left_base[i]) if (i < len(left_vals) and i < len(left_base)) else None
        for i in range(n)
    ]
    r_deltas: list[Optional[float]] = [
        (right_vals[i] - right_base[i]) if (i < len(right_vals) and i < len(right_base)) else None
        for i in range(n)
    ]
    l_spike_idx = _argmax_delta(l_deltas, SPIKE_THRESHOLD)
    r_spike_idx = _argmax_delta(r_deltas, SPIKE_THRESHOLD)
    if l_spike_idx is not None:
        spikes.update("left", l_spike_idx, l_deltas[l_spike_idx], now)
    if r_spike_idx is not None:
        spikes.update("right", r_spike_idx, r_deltas[r_spike_idx], now)

    lines: list[str] = []
    # \033[H = cursor home, \033[J = clear from cursor to end of screen.
    lines.append(
        f"\033[H\033[J=== Dex3 Tactile Smoke Test ===  t={t:6.1f}s   samples L={l_count} R={r_count}"
    )
    lines.append("")
    cell_w = 1 + 1 + PAD_W + 2 + 8 + 2 + 8 + 2 + bar_max  # spike + space + pad + raw + Δ + bar
    lines.append(f"  {'mN':<3}  {'LEFT':<{cell_w}}    {'RIGHT':<{cell_w}}")
    header_one = f"{'':<2}{'pad':<{PAD_W}}  {'raw':>8}  {'Δ':>8}  {'bar':<{bar_max}}"
    lines.append(f"  {'':<3}  {header_one}    {header_one}")
    for i in range(n):
        l_pad = label_for(i, "left")
        r_pad = label_for(i, "right")
        l_raw = left_vals[i] if i < len(left_vals) else None
        r_raw = right_vals[i] if i < len(right_vals) else None
        l_cell = _format_cell(l_pad, l_raw, l_deltas[i], bar_unit, bar_max, i == l_spike_idx)
        r_cell = _format_cell(r_pad, r_raw, r_deltas[i], bar_unit, bar_max, i == r_spike_idx)
        lines.append(f"  m{i:<2}  {l_cell}    {r_cell}")
    lines.append("")

    def fmt_recent(side: str) -> str:
        rec = spikes.get(side, now)
        if rec is None:
            return "—"
        idx, dlt, ts = rec
        age = now - ts
        sign = "+" if dlt >= 0 else "-"
        return f"m{idx}={label_for(idx, side)} (Δ={sign}{abs(dlt):.0f}, {age:.1f}s ago)"

    lines.append(f"  Recent spike (last {SPIKE_MEMORY_SECONDS:.0f}s):  L={fmt_recent('left'):<40}  R={fmt_recent('right')}")
    lines.append("")
    lines.append(f"  bar = 1 block per {bar_unit:.0f} ADC counts; spike threshold = {SPIKE_THRESHOLD:.0f}.  Ctrl-C to exit.")
    return "\n".join(lines)


def fmt_raw(t: float, sample_left, sample_right) -> str:
    lines = [f"=== t={t:6.2f}s ==="]
    for hand_name, side, sample in (
        ("LEFT HAND", "left", sample_left),
        ("RIGHT HAND", "right", sample_right),
    ):
        lines.append(hand_name)
        for i, module in enumerate(sample.press_sensor_state):
            cells = " ".join(f"{c:5.2f}" for c in module.pressure)
            lines.append(
                f"  m{i} ({label_for(i, side):<8}): [{cells}]  lost={module.lost}"
            )
    return "\n".join(lines)


def wait_for_first(cache: HandStateCache, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample, _ = cache.latest()
        if sample is not None:
            return True
        time.sleep(0.05)
    return False


def capture_baseline(
    left: HandStateCache,
    right: HandStateCache,
    duration_s: float = 1.0,
    n_samples: int = 10,
) -> tuple[list[float], list[float]]:
    """Mean per-module sums across n_samples spaced over duration_s."""
    interval = duration_s / n_samples
    left_acc: Optional[list[float]] = None
    right_acc: Optional[list[float]] = None
    n = 0
    for _ in range(n_samples):
        l_sample, _ = left.latest()
        r_sample, _ = right.latest()
        if l_sample is None or r_sample is None:
            time.sleep(interval)
            continue
        l = module_sums(l_sample)
        r = module_sums(r_sample)
        if left_acc is None:
            left_acc = list(l)
            right_acc = list(r)
        else:
            for i in range(min(len(left_acc), len(l))):
                left_acc[i] += l[i]
            for i in range(min(len(right_acc), len(r))):
                right_acc[i] += r[i]
        n += 1
        time.sleep(interval)
    if n == 0 or left_acc is None or right_acc is None:
        return [], []
    return [v / n for v in left_acc], [v / n for v in right_acc]


PROTOCOL_BANNER = """\
────────────────────────────────────────────────────────────
TEST PROTOCOL — press each pad in turn and watch values rise:

  1. Right palm        → expect r[0] to spike
  2. Right thumb tip   → expect r[3] (or whichever module is thumb_2) to spike
  3. Right index tip   → expect r[7] (or whichever) to spike
  4. Right middle tip  → expect r[5] (or whichever) to spike
  5. Repeat for left hand
  6. Squeeze a soft object in right hand → multiple modules rise

If a press doesn't move the expected channel, the SDK module index does
NOT match the physical layout in SYSTEM0_FACTS.md — log the actual mapping.
────────────────────────────────────────────────────────────
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only smoke test for Dex3 tactile sensors on a real Unitree G1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iface",
        required=True,
        help="Network interface DDS will bind to (e.g. eth0, enp0s31f6).",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Print rate in Hz (SDK runs at ~1 kHz; this only throttles stdout).",
    )
    p.add_argument(
        "--mode",
        choices=("summary", "raw", "delta"),
        default="summary",
        help="Display mode.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        print("[ERROR] --rate must be positive.", file=sys.stderr)
        return 2

    ChannelFactoryInitialize, ChannelSubscriber, HandState_ = _import_sdk()

    ChannelFactoryInitialize(0, args.iface)
    print(f"[OK] DDS initialized on iface={args.iface}")

    left_cache = HandStateCache("left")
    right_cache = HandStateCache("right")

    left_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
    right_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
    left_sub.Init(left_cache.on_message, 0)
    right_sub.Init(right_cache.on_message, 0)

    try:
        if not wait_for_first(left_cache, 3.0):
            print("[ERROR] No message on rt/dex3/left/state within 3s.", file=sys.stderr)
            return 3
        if not wait_for_first(right_cache, 3.0):
            print("[ERROR] No message on rt/dex3/right/state within 3s.", file=sys.stderr)
            return 3
        print("[OK] Subscribed to left and right hand state. Press Ctrl-C to exit.")

        # Sanity-check module count vs. assumed labels (don't fail — just warn).
        l_sample, _ = left_cache.latest()
        r_sample, _ = right_cache.latest()
        for hand, sample in (("left", l_sample), ("right", r_sample)):
            n_modules = len(sample.press_sensor_state)
            expected = len(MODULE_LABELS[hand])
            if n_modules != expected:
                print(
                    f"[WARN] {hand} hand reports {n_modules} modules; "
                    f"label table assumes {expected}. "
                    "Extra modules will be shown as #idx; missing ones simply absent."
                )

        # Baseline.
        print("[..] Capturing 1s baseline (10 samples per hand)...")
        left_base, right_base = capture_baseline(left_cache, right_cache)
        if not left_base or not right_base:
            print("[ERROR] Failed to capture baseline (no samples).", file=sys.stderr)
            return 4
        print(
            "[OK] Baseline (per-module pressure sum):\n"
            f"     L: {[f'{v:.2f}' for v in left_base]}\n"
            f"     R: {[f'{v:.2f}' for v in right_base]}"
        )
        print(PROTOCOL_BANNER)

        period = 1.0 / args.rate
        t0 = time.monotonic()
        next_tick = t0
        spikes = SpikeTracker()
        try:
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(period, next_tick - now))
                    continue
                next_tick += period

                l_sample, l_count = left_cache.latest()
                r_sample, r_count = right_cache.latest()
                if l_sample is None or r_sample is None:
                    continue
                t = now - t0

                if args.mode == "raw":
                    print(fmt_raw(t, l_sample, r_sample))
                else:
                    l_vals = module_sums(l_sample)
                    r_vals = module_sums(r_sample)
                    panel = render_panel(
                        t, now, l_vals, r_vals, left_base, right_base,
                        l_count, r_count, spikes,
                    )
                    sys.stdout.write(panel)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print()  # newline below the static panel
    finally:
        try:
            left_sub.Close()
        except Exception as e:
            print(f"[WARN] left subscriber close: {e}", file=sys.stderr)
        try:
            right_sub.Close()
        except Exception as e:
            print(f"[WARN] right subscriber close: {e}", file=sys.stderr)
        _, l_count = left_cache.latest()
        _, r_count = right_cache.latest()
        elapsed = time.monotonic() - t0 if "t0" in locals() else 0.0
        print(
            f"[OK] Clean exit. left={l_count} samples, right={r_count} samples "
            f"over {elapsed:.1f}s."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

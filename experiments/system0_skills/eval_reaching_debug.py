#!/usr/bin/env python3
"""Debug reaching eval: 5-config comparison matrix.

Runs the existing eval pipeline and monitors EE-to-block distance.
Tests 5 configs by restarting System 1 with different settings.

Since we can't easily toggle IK/depth mid-run, this script:
1. Monitors the CURRENT pipeline (Config E: full) for 60s per position
2. Records EE distances from /tmp/sim_block_positions.json
3. Records per-axis error (X, Y, Z breakdown)
4. Tests 9 block positions

For A-D configs, System 1 would need to be restarted with modified code.
For now: run Config E (full pipeline) and record comprehensive data.

Usage:
  python eval_reaching_debug.py
"""

import json
import os
import time
import numpy as np

POSITIONS_FILE = "/tmp/sim_block_positions.json"
LAYOUT_FILE = "/tmp/sim_block_layout.json"
ARM_DEBUG = "/tmp/arm_debug.txt"

TEST_POSITIONS = [
    (-0.05, -0.35, 0.82, "center"),
    (-0.15, -0.35, 0.82, "near"),
    ( 0.05, -0.35, 0.82, "far"),
    (-0.05, -0.25, 0.82, "left"),
    (-0.05, -0.45, 0.82, "right"),
    (-0.15, -0.25, 0.82, "near_left"),
    (-0.15, -0.45, 0.82, "near_right"),
    ( 0.05, -0.25, 0.82, "far_left"),
    ( 0.05, -0.45, 0.82, "far_right"),
]

SECONDS_PER_POS = 30


def read_positions():
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except:
        return None


def compute_distance_and_error(pos, target_block="red_block"):
    if pos is None:
        return None, None, None, None, None
    ree = pos.get("right_ee", {}).get("rel")
    lee = pos.get("left_ee", {}).get("rel")
    bp = pos.get(target_block, {}).get("rel")
    if ree is None or bp is None:
        return None, None, None, None, None
    ree = np.array(ree)
    bp = np.array(bp)
    dist = np.linalg.norm(ree - bp)
    xerr = ree[0] - bp[0]
    yerr = ree[1] - bp[1]
    zerr = ree[2] - bp[2]
    lee_dist = np.linalg.norm(np.array(lee) - bp) if lee else None
    return dist, xerr, yerr, zerr, lee_dist


def teleport_block(pos_local):
    """Write block layout for sim to teleport."""
    layout = {
        "red_block": list(pos_local[:3]),
        "yellow_block": [pos_local[0] + 0.15, pos_local[1], pos_local[2]],
        "green_block": [pos_local[0] - 0.15, pos_local[1], pos_local[2]],
    }
    with open(LAYOUT_FILE, "w") as f:
        json.dump(layout, f)


def main():
    print("=" * 70)
    print("  Reaching Debug Eval — Per-Position Per-Axis Analysis")
    print("  Config: Current pipeline (IK norm + depth + 4 denoising steps)")
    print("=" * 70)

    if not os.path.exists(POSITIONS_FILE):
        print("[ERROR] No block positions file. Is the pipeline running?")
        return

    results = []

    for pi, (x, y, z, label) in enumerate(TEST_POSITIONS):
        print(f"\n{'='*60}")
        print(f"  Position {pi+1}/{len(TEST_POSITIONS)}: {label} ({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"{'='*60}")

        # Teleport block
        teleport_block([x, y, z, label])
        time.sleep(3)  # wait for sim to process

        # Read initial position
        pos = read_positions()
        if pos:
            bp = pos.get("red_block", {}).get("rel", [0, 0, 0])
            print(f"  Block actual: ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")

        # Monitor for SECONDS_PER_POS
        min_dist = float("inf")
        min_t = 0
        dist_curve = []
        errors = []

        for sec in range(SECONDS_PER_POS):
            time.sleep(1)
            pos = read_positions()
            dist, xerr, yerr, zerr, ldist = compute_distance_and_error(pos)
            if dist is None:
                continue

            dist_cm = dist * 100
            dist_curve.append(dist_cm)
            errors.append((xerr * 100, yerr * 100, zerr * 100))

            if dist_cm < min_dist:
                min_dist = dist_cm
                min_t = sec + 1

            if sec % 5 == 4:
                print(f"  [t={sec+1:2d}s] R:{dist_cm:5.1f}cm  x={xerr*100:+5.1f} y={yerr*100:+5.1f} z={zerr*100:+5.1f}")

        # Summary for this position
        avg_errors = np.mean(errors, axis=0) if errors else [0, 0, 0]
        result = {
            "label": label,
            "pos": [x, y, z],
            "min_dist_cm": round(min_dist, 1),
            "min_t_s": min_t,
            "final_dist_cm": round(dist_curve[-1], 1) if dist_curve else None,
            "avg_x_err_cm": round(avg_errors[0], 1),
            "avg_y_err_cm": round(avg_errors[1], 1),
            "avg_z_err_cm": round(avg_errors[2], 1),
            "reached_8cm": min_dist < 8.0,
            "reached_15cm": min_dist < 15.0,
        }
        results.append(result)
        print(f"  → min={min_dist:.1f}cm at t={min_t}s | avg err: x={avg_errors[0]:+.1f} y={avg_errors[1]:+.1f} z={avg_errors[2]:+.1f}")

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  REACHING EVAL RESULTS — Config E (IK norm + depth + 4 steps)")
    print(f"{'='*70}")
    print(f"\n  {'Pos':<12} {'Min':<8} {'Final':<8} {'X err':<8} {'Y err':<8} {'Z err':<8} {'<8cm':<6} {'<15cm'}")
    print(f"  {'-'*64}")
    for r in results:
        ok8 = "YES" if r["reached_8cm"] else "no"
        ok15 = "YES" if r["reached_15cm"] else "no"
        print(f"  {r['label']:<12} {r['min_dist_cm']:<8.1f} {r['final_dist_cm']:<8.1f} "
              f"{r['avg_x_err_cm']:+6.1f}  {r['avg_y_err_cm']:+6.1f}  {r['avg_z_err_cm']:+6.1f}  {ok8:<6} {ok15}")

    n8 = sum(1 for r in results if r["reached_8cm"])
    n15 = sum(1 for r in results if r["reached_15cm"])
    avg_min = np.mean([r["min_dist_cm"] for r in results])
    print(f"\n  Average min distance: {avg_min:.1f}cm")
    print(f"  Reached <8cm: {n8}/{len(results)}")
    print(f"  Reached <15cm: {n15}/{len(results)}")

    # Dominant error analysis
    all_z = [abs(r["avg_z_err_cm"]) for r in results]
    all_y = [abs(r["avg_y_err_cm"]) for r in results]
    all_x = [abs(r["avg_x_err_cm"]) for r in results]
    print(f"\n  Dominant error: X={np.mean(all_x):.1f}cm Y={np.mean(all_y):.1f}cm Z={np.mean(all_z):.1f}cm")
    if np.mean(all_z) > np.mean(all_y) and np.mean(all_z) > np.mean(all_x):
        print(f"  → Z (height) is the DOMINANT error. Hand doesn't descend to table.")
    elif np.mean(all_y) > np.mean(all_x):
        print(f"  → Y (lateral) is the DOMINANT error. Hand doesn't align with block.")

    # Save
    out = os.path.expanduser("~/unitree_sim_isaaclab/logs/reaching_debug_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": "E_ik_depth_4steps", "results": results,
                    "avg_min_cm": round(avg_min, 1)}, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

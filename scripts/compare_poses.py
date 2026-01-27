#!/usr/bin/env python3
"""Compare poses between CUDA and Autoware NDT implementations.

This script compares the alignment results from both implementations
to measure position and rotation differences.

Usage:
    python3 scripts/compare_poses.py logs/ndt_cuda_profiling.jsonl logs/ndt_autoware_profiling.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_profiling_log(path: Path) -> list:
    """Load JSONL profiling log, skipping header."""
    entries = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if "timestamp_ns" in data and "final_pose" in data:
                entries.append(data)
    return entries


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (RPY) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def quaternion_angle_diff(q1: tuple, q2: tuple) -> float:
    """Compute angle between two quaternions in radians."""
    # q1 and q2 are (w, x, y, z)
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = min(1.0, dot)  # Clamp for numerical stability
    return 2 * math.acos(dot)


def compute_pose_diff(pose1: list, pose2: list) -> dict:
    """Compute position and rotation difference between two poses."""
    # Position difference (Euclidean)
    pos1 = np.array(pose1[:3])
    pos2 = np.array(pose2[:3])
    pos_diff = np.linalg.norm(pos1 - pos2)
    pos_diff_2d = np.linalg.norm(pos1[:2] - pos2[:2])

    # Rotation difference via quaternions
    q1 = euler_to_quaternion(pose1[3], pose1[4], pose1[5])
    q2 = euler_to_quaternion(pose2[3], pose2[4], pose2[5])
    rot_diff = quaternion_angle_diff(q1, q2)

    return {
        "position_3d": pos_diff,
        "position_2d": pos_diff_2d,
        "rotation": rot_diff,
    }


def compare_implementations(cuda_entries: list, autoware_entries: list) -> dict:
    """Compare CUDA and Autoware alignment results."""
    # Index by timestamp
    cuda_by_ts = {e["timestamp_ns"]: e for e in cuda_entries}
    autoware_by_ts = {e["timestamp_ns"]: e for e in autoware_entries}

    # Find common timestamps
    common_ts = set(cuda_by_ts.keys()) & set(autoware_by_ts.keys())

    if not common_ts:
        return {"error": "No common timestamps found"}

    # Compute differences for each common timestamp
    pos_diffs_3d = []
    pos_diffs_2d = []
    rot_diffs = []
    iter_diffs = []
    details = []

    for ts in sorted(common_ts):
        cuda = cuda_by_ts[ts]
        autoware = autoware_by_ts[ts]

        diff = compute_pose_diff(cuda["final_pose"], autoware["final_pose"])
        pos_diffs_3d.append(diff["position_3d"])
        pos_diffs_2d.append(diff["position_2d"])
        rot_diffs.append(diff["rotation"])

        iter_diff = cuda["total_iterations"] - autoware["total_iterations"]
        iter_diffs.append(iter_diff)

        details.append({
            "timestamp_ns": ts,
            "position_diff_3d": diff["position_3d"],
            "position_diff_2d": diff["position_2d"],
            "rotation_diff": diff["rotation"],
            "cuda_iterations": cuda["total_iterations"],
            "autoware_iterations": autoware["total_iterations"],
            "cuda_converged": cuda["convergence_status"] == "Converged",
            "autoware_converged": autoware["convergence_status"] == "Converged",
        })

    pos_diffs_3d = np.array(pos_diffs_3d)
    pos_diffs_2d = np.array(pos_diffs_2d)
    rot_diffs = np.array(rot_diffs)
    iter_diffs = np.array(iter_diffs)

    return {
        "common_frames": len(common_ts),
        "cuda_total": len(cuda_entries),
        "autoware_total": len(autoware_entries),
        "position_3d": {
            "mean": float(np.mean(pos_diffs_3d)),
            "std": float(np.std(pos_diffs_3d)),
            "max": float(np.max(pos_diffs_3d)),
            "min": float(np.min(pos_diffs_3d)),
            "p95": float(np.percentile(pos_diffs_3d, 95)),
        },
        "position_2d": {
            "mean": float(np.mean(pos_diffs_2d)),
            "std": float(np.std(pos_diffs_2d)),
            "max": float(np.max(pos_diffs_2d)),
            "min": float(np.min(pos_diffs_2d)),
            "p95": float(np.percentile(pos_diffs_2d, 95)),
        },
        "rotation": {
            "mean": float(np.mean(rot_diffs)),
            "std": float(np.std(rot_diffs)),
            "max": float(np.max(rot_diffs)),
            "min": float(np.min(rot_diffs)),
            "p95": float(np.percentile(rot_diffs, 95)),
        },
        "iterations": {
            "mean_diff": float(np.mean(iter_diffs)),
            "std": float(np.std(iter_diffs)),
            "range": [int(np.min(iter_diffs)), int(np.max(iter_diffs))],
        },
        "details": details,
    }


def print_comparison(result: dict, verbose: bool = False):
    """Print comparison results."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 60)
    print("CUDA vs Autoware Pose Comparison")
    print("=" * 60)

    print(f"\nFrames: {result['common_frames']} common "
          f"(CUDA: {result['cuda_total']}, Autoware: {result['autoware_total']})")

    print("\n--- Position Difference (3D) ---")
    p3d = result["position_3d"]
    print(f"  Mean:  {p3d['mean']:.4f} m")
    print(f"  Std:   {p3d['std']:.4f} m")
    print(f"  Max:   {p3d['max']:.4f} m")
    print(f"  P95:   {p3d['p95']:.4f} m")

    print("\n--- Position Difference (2D) ---")
    p2d = result["position_2d"]
    print(f"  Mean:  {p2d['mean']:.4f} m")
    print(f"  Std:   {p2d['std']:.4f} m")
    print(f"  Max:   {p2d['max']:.4f} m")
    print(f"  P95:   {p2d['p95']:.4f} m")

    print("\n--- Rotation Difference ---")
    rot = result["rotation"]
    print(f"  Mean:  {rot['mean']:.4f} rad ({np.degrees(rot['mean']):.2f} deg)")
    print(f"  Std:   {rot['std']:.4f} rad ({np.degrees(rot['std']):.2f} deg)")
    print(f"  Max:   {rot['max']:.4f} rad ({np.degrees(rot['max']):.2f} deg)")
    print(f"  P95:   {rot['p95']:.4f} rad ({np.degrees(rot['p95']):.2f} deg)")

    print("\n--- Iteration Count Difference (CUDA - Autoware) ---")
    it = result["iterations"]
    print(f"  Mean:  {it['mean_diff']:+.2f}")
    print(f"  Std:   {it['std']:.2f}")
    print(f"  Range: [{it['range'][0]:+d}, {it['range'][1]:+d}]")

    if verbose and result["details"]:
        print("\n--- Largest Position Differences ---")
        sorted_details = sorted(result["details"],
                                key=lambda x: x["position_diff_3d"],
                                reverse=True)[:10]
        print(f"{'Timestamp':<20} {'Pos Diff':>10} {'Rot Diff':>10} {'CUDA It':>8} {'AW It':>8}")
        print("-" * 60)
        for d in sorted_details:
            ts_short = str(d["timestamp_ns"])[-12:]
            print(f"...{ts_short:<17} {d['position_diff_3d']:>10.4f} "
                  f"{d['rotation_diff']:>10.4f} {d['cuda_iterations']:>8} "
                  f"{d['autoware_iterations']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CUDA and Autoware NDT poses"
    )
    parser.add_argument("cuda_log", type=Path, help="CUDA profiling log (JSONL)")
    parser.add_argument("autoware_log", type=Path, help="Autoware profiling log (JSONL)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed per-frame differences")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")
    args = parser.parse_args()

    if not args.cuda_log.exists():
        print(f"Error: CUDA log not found: {args.cuda_log}", file=sys.stderr)
        sys.exit(1)
    if not args.autoware_log.exists():
        print(f"Error: Autoware log not found: {args.autoware_log}", file=sys.stderr)
        sys.exit(1)

    cuda_entries = load_profiling_log(args.cuda_log)
    autoware_entries = load_profiling_log(args.autoware_log)

    print(f"Loaded {len(cuda_entries)} CUDA entries, {len(autoware_entries)} Autoware entries")

    result = compare_implementations(cuda_entries, autoware_entries)

    if args.json:
        # Remove verbose details for JSON output unless verbose
        if not args.verbose:
            result.pop("details", None)
        print(json.dumps(result, indent=2))
    else:
        print_comparison(result, verbose=args.verbose)


if __name__ == "__main__":
    main()

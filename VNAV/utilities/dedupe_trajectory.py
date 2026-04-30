"""Collapse a per-iter trajectory log to one sample per target.

Each target writes its target_pose unchanged for every inner-loop iteration,
so consecutive identical GT lines bound one target. We keep the last line of
each run in both the GT and EST files. Originals are saved to ``*.original``.
"""
import argparse
import os
import shutil
import sys


def _pose_key(line: str):
    parts = line.strip().split()
    if len(parts) != 8:
        return None
    return tuple(parts[1:])  # (tx, ty, tz, qx, qy, qz, qw) as strings


def dedupe(run_dir: str) -> int:
    gt_path = os.path.join(run_dir, "trajectory_groundtruth.txt")
    est_path = os.path.join(run_dir, "trajectory_estimated.txt")

    for p in (gt_path, est_path):
        if not os.path.exists(p):
            print(f"[error] {p} not found", file=sys.stderr)
            return -1

    with open(gt_path) as f:
        gt_lines = [l for l in f if l.strip()]
    with open(est_path) as f:
        est_lines = [l for l in f if l.strip()]

    if len(gt_lines) != len(est_lines):
        print(
            f"[error] line count mismatch: gt={len(gt_lines)} est={len(est_lines)}",
            file=sys.stderr,
        )
        return -1

    keep = []
    prev = None
    for i, line in enumerate(gt_lines):
        key = _pose_key(line)
        if key is None:
            continue
        if prev is not None and key != prev:
            keep.append(i - 1)
        prev = key
    if gt_lines:
        keep.append(len(gt_lines) - 1)

    print(f"Read {len(gt_lines)} lines, detected {len(keep)} targets.")

    shutil.copy(gt_path, gt_path + ".original")
    shutil.copy(est_path, est_path + ".original")

    with open(gt_path, "w") as f:
        for idx in keep:
            f.write(gt_lines[idx])
    with open(est_path, "w") as f:
        for idx in keep:
            f.write(est_lines[idx])

    print(f"Wrote {len(keep)} lines to {gt_path}")
    print(f"Wrote {len(keep)} lines to {est_path}")
    print(f"Backups: {gt_path}.original, {est_path}.original")
    return len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Path to a RUNS/<timestamp> directory")
    ap.add_argument("--rerun-evo", action="store_true", help="Run evo evaluation after dedup")
    args = ap.parse_args()

    n = dedupe(args.run_dir)
    if n < 0:
        sys.exit(1)

    if args.rerun_evo:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from VNAV.control.servoing import _run_evo_evaluation
        gt = os.path.join(args.run_dir, "trajectory_groundtruth.txt")
        est = os.path.join(args.run_dir, "trajectory_estimated.txt")
        _run_evo_evaluation(gt, est, args.run_dir)


if __name__ == "__main__":
    main()

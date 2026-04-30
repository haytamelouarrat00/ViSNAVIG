#!/bin/bash
# Run a full evo evaluation report (APE x {trans_part, angle_rad, full},
# RPE x {1, 10, 100} frame deltas) on a RUNS/<stamp> directory and aggregate
# the results into a single summary CSV + plots in <run>/report/.
#
# Usage:
#   scripts/evo_report.sh <run_dir> [report_subdir]
#
# Examples:
#   scripts/evo_report.sh RUNS/20260429_235036
#   scripts/evo_report.sh RUNS/20260429_235036 report_dedup
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <run_dir> [report_subdir]" >&2
    exit 1
fi

RUN_DIR="$1"
REPORT_SUBDIR="${2:-report}"

if [ ! -d "$RUN_DIR" ]; then
    echo "[error] Run directory not found: $RUN_DIR" >&2
    exit 1
fi

GT="$RUN_DIR/trajectory_groundtruth.txt"
EST="$RUN_DIR/trajectory_estimated.txt"

for f in "$GT" "$EST"; do
    if [ ! -f "$f" ]; then
        echo "[error] Missing trajectory file: $f" >&2
        exit 1
    fi
done

REPORT_DIR="$(realpath -m "$RUN_DIR/$REPORT_SUBDIR")"
mkdir -p "$REPORT_DIR"

GT_ABS="$(realpath "$GT")"
EST_ABS="$(realpath "$EST")"

# Clean previous artifacts so evo doesn't prompt to overwrite
rm -f "$REPORT_DIR"/ape_*.pdf "$REPORT_DIR"/ape_*.zip \
      "$REPORT_DIR"/rpe_*.pdf "$REPORT_DIR"/rpe_*.zip \
      "$REPORT_DIR"/summary.csv

cd "$REPORT_DIR"

APE_RELATIONS=(trans_part angle_rad full)
RPE_DELTAS=(1 10 100)

echo "=== APE ==="
for rel in "${APE_RELATIONS[@]}"; do
    echo "--- pose_relation=$rel ---"
    evo_ape tum "$GT_ABS" "$EST_ABS" \
        -v --pose_relation="$rel" \
        --save_plot "ape_${rel}.pdf" \
        --save_results "ape_${rel}.zip"
done

echo "=== RPE (translation, per frame delta) ==="
for d in "${RPE_DELTAS[@]}"; do
    echo "--- delta=$d frames ---"
    evo_rpe tum "$GT_ABS" "$EST_ABS" \
        -v --pose_relation=trans_part \
        --delta "$d" --delta_unit f \
        --save_plot "rpe_d${d}.pdf" \
        --save_results "rpe_d${d}.zip"
done

echo "=== Aggregating with evo_res ==="
evo_res ape_*.zip rpe_*.zip --use_filenames --ignore_title --save_table summary.csv

echo ""
echo "Report written to: $REPORT_DIR"
ls -1

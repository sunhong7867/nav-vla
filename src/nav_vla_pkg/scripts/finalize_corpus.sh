#!/bin/bash
# Everything between "collection finished" and "ready to train", in order.
#
#   bash finalize_corpus.sh SESSION [SESSION ...]
#
# Fail-fast on purpose. Each stage answers a question the next one assumes, and
# running the next stage on a failed answer is how a bad corpus reaches training
# looking healthy:
#
#   verify    are the streams joinable and do the axes actually diverge?
#   resample  can the four streams be put on one 10 Hz grid within tolerance?
#   cmi       does the instruction still explain the action once you can see?
#   package   the 10 Hz subset, ~30% of the bytes, for the GPU box
#
# The resample step is the slow one (a few minutes per hundred episodes) and the
# only one that writes into the session directories.
set -euo pipefail

REPO="$HOME/ROS2_project/nav-vla"
S="$REPO/src/nav_vla_pkg/scripts"
OUT="${PACK_OUT:-$REPO/src/nav_vla_pkg/data_v2/packed_v2}"
SESSIONS=("$@")
[ ${#SESSIONS[@]} -gt 0 ] || { echo "give at least one session directory" >&2; exit 2; }

echo "############ 1/4  verify ############"
python3 "$S/verify_corpus.py" "${SESSIONS[@]}" --json "$REPO/verify_report.json"

echo
echo "############ 2/4  resample to 10 Hz ############"
for sess in "${SESSIONS[@]}"; do
  echo "--- $sess"
  # -90 deg is measured, not assumed: straight-line runs from four headings gave
  # -90.00 with sd 0.00 (docs/ver/20260728_1713 section 7). tf is the stream the
  # action labels are built from; the CLI fallback is 15 Hz and lags.
  python3 "$S/resample_episodes.py" --session "$sess" \
      --yaw-offset-deg -90 --pose-source tf
done

echo
echo "############ 3/4  conditional mutual information ############"
# Per axis, because they are different claims. The ordinal axis is the one that
# has to carry criterion (a); speed is expected to be near zero here and that is
# correct, not a failure.
for axis in ordinal lane speed; do
  echo "--- axis=$axis"
  python3 "$S/measure_cmi.py" "${SESSIONS[@]}" --axis "$axis" \
      --json "$REPO/cmi_${axis}.json" || true
done

echo
echo "############ 4/4  package ############"
rm -rf "$OUT"
python3 "$S/package_corpus.py" "$OUT" "${SESSIONS[@]}"

echo
echo "done. ship it with:"
echo "  rsync -a --info=progress2 $OUT/ autolab_sw@115.145.211.157:~/sunhong/nav-vla/data/packed_v2/"

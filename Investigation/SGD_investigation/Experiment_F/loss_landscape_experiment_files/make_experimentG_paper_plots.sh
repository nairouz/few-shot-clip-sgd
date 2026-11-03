#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <logdir> <outdir> <paper_fig_dir>"
  exit 1
fi

LOGDIR="$1"
OUTDIR="$2"
PAPERDIR="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_experimentG_connectivity_paper.py"

mkdir -p "$OUTDIR" "$PAPERDIR"

find "$LOGDIR" -maxdepth 1 -name '*_config.json' | sort | while read -r cfg; do
  base="$(basename "$cfg" _config.json)"
  run_out="${OUTDIR}/${base}"
  mkdir -p "$run_out"
  python "$PLOT_SCRIPT" --logdir "$LOGDIR" --run_name "$base" --outdir "$run_out"

  dataset="$(echo "$base" | sed -E 's/.*dataset_([^_]+(_[^_]+)?)_shots.*/\1/')"
  short="$dataset"
  if [[ "$dataset" == "oxford_pets" ]]; then short="oxford"; fi
  cp "$run_out/expG_connectivity.pdf" "$PAPERDIR/expG_${short}_connectivity.pdf"
  cp "$run_out/expG_connectivity.png" "$PAPERDIR/expG_${short}_connectivity.png"
  cp "$run_out/expG_pairwise_barriers.pdf" "$PAPERDIR/expG_${short}_barriers.pdf"
  cp "$run_out/expG_pairwise_barriers.png" "$PAPERDIR/expG_${short}_barriers.png"
done

echo "Saved paper figures to: $PAPERDIR"

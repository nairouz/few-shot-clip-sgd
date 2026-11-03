#!/usr/bin/env bash
set -euo pipefail
LOGDIR=${1:?need logdir}
OUTDIR=${2:?need outdir}
FIGDIR=${3:?need figdir}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROTOCOL=J3
python "$SCRIPT_DIR/plot_experimentJ_paper.py" --logdir "$LOGDIR" --outdir "$OUTDIR" --protocol "$PROTOCOL"
case "$PROTOCOL" in
  J1) base="expJ1_epoch_matched_summary" ;;
  J2) base="expJ2_update_matched_summary" ;;
  J3) base="expJ3_example_matched_summary" ;;
  J4) base="expJ4_update_example_matched_summary" ;;
  *) echo "Unknown protocol" >&2; exit 1 ;;
esac
mkdir -p "$FIGDIR"
cp "$OUTDIR/${base}.pdf" "$FIGDIR/${base}.pdf"
cp "$OUTDIR/${base}.png" "$FIGDIR/${base}.png"
echo "Copied $base to $FIGDIR"

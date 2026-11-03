#!/usr/bin/env bash
set -euo pipefail

LOGDIR=${1:-./results_without_aug/experimentF_logs}
OUTDIR=${2:-./results_without_aug/experimentF_plots_paper}
FIGDIR=${3:-./figures/sgd_success}

mkdir -p "$OUTDIR" "$FIGDIR"

python loss_landscape_experiment_files/plot_experimentF_block_safety_paper.py \
  --logdir "$LOGDIR" \
  --outdir "$OUTDIR"

cp "$OUTDIR/expF_block_safety_summary.pdf" "$FIGDIR/expF_block_safety_summary.pdf"
cp "$OUTDIR/expF_block_safety_summary.png" "$FIGDIR/expF_block_safety_summary.png"

echo "Saved: $FIGDIR/expF_block_safety_summary.pdf"

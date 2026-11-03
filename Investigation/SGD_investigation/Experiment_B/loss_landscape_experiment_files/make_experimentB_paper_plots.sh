#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

LOGDIR="${1:-./results_without_aug/experimentB_logs}"
OUTROOT="${2:-./results_without_aug/experimentB_plots_paper}"
FIGDIR="${3:-./figures/sgd_success}"
PLOT_SCRIPT="loss_landscape_experiment_files/plot_experimentB_localstep.py"

mkdir -p "$OUTROOT" "$FIGDIR"

runs=(
  "eurosat:experimentB_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
  "oxford:experimentB_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
  "dtd:experimentB_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
)

for item in "${runs[@]}"; do
  dataset="${item%%:*}"
  run_name="${item#*:}"
  outdir="$OUTROOT/$run_name"
  python "$PLOT_SCRIPT" --logdir "$LOGDIR" --run_name "$run_name" --outdir "$outdir"

  cp "$outdir/expB_safe_radius.pdf" "$FIGDIR/expB_${dataset}_safe_radius.pdf"
  cp "$outdir/expB_safe_radius.png" "$FIGDIR/expB_${dataset}_safe_radius.png"
  cp "$outdir/expB_minimizer_multiplier.pdf" "$FIGDIR/expB_${dataset}_minimizer.pdf"
  cp "$outdir/expB_minimizer_multiplier.png" "$FIGDIR/expB_${dataset}_minimizer.png"
  cp "$outdir/expB_localstep_losses.pdf" "$FIGDIR/expB_${dataset}_localstep.pdf"
  cp "$outdir/expB_localstep_losses.png" "$FIGDIR/expB_${dataset}_localstep.png"
done

echo "Saved paper-ready Experiment B plots under: $FIGDIR"

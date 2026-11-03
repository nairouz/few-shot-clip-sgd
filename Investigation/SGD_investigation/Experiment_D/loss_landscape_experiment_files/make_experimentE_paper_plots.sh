#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${1:-./results_without_aug/experimentE_logs}"
OUTROOT="${2:-./results_without_aug/experimentE_plots_paper}"
FIGDIR="${3:-./figures/sgd_success}"
PLOT_SCRIPT="loss_landscape_experiment_files/plot_experimentE_directional_curvature.py"

mkdir -p "$OUTROOT" "$FIGDIR"

runs=(
  "eurosat:experimentE_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
  "oxford:experimentE_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
  "dtd:experimentE_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true"
)

for item in "${runs[@]}"; do
  dataset="${item%%:*}"
  run_name="${item#*:}"
  outdir="$OUTROOT/$run_name"
  python "$PLOT_SCRIPT" --logdir "$LOGDIR" --run_name "$run_name" --outdir "$outdir" --yscale symlog --linthresh 1.0

  cp "$outdir/expE_directional_curvature_over_time.pdf" "$FIGDIR/expE_${dataset}_directional.pdf"
  cp "$outdir/expE_directional_curvature_over_time.png" "$FIGDIR/expE_${dataset}_directional.png"
  cp "$outdir/expE_block_curvature_full_t30.pdf" "$FIGDIR/expE_${dataset}_blocks_t30.pdf"
  cp "$outdir/expE_block_curvature_full_t30.png" "$FIGDIR/expE_${dataset}_blocks_t30.png"
done

echo "Saved paper-ready Experiment E plots under: $FIGDIR"

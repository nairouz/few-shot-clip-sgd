#!/usr/bin/env bash
set -euo pipefail

LOGDIR=${1:-./results_without_aug/experimentA_logs}
OUTROOT=${2:-./results_without_aug/experimentA_plots_paper_notation}

runs=(
  experimentA_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true
  experimentA_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true
  experimentA_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true
)

for run_name in "${runs[@]}"; do
  python loss_landscape_experiment_files/plot_experimentA_interpolation.py     --logdir "$LOGDIR"     --run_name "$run_name"     --outdir "$OUTROOT/$run_name"
done

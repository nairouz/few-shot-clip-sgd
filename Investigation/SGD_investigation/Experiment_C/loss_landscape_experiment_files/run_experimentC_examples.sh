#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# ==========================================================
# 1) Save SGD checkpoints for the three representative datasets
# ==========================================================

python loss_landscape_experiment_files/measure_experimentC_checkpoints.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentC_logs

python loss_landscape_experiment_files/measure_experimentC_checkpoints.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentC_logs

python loss_landscape_experiment_files/measure_experimentC_checkpoints.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentC_logs

# ==========================================================
# 2) Example surface measurements
#    (one checkpoint + one probe per call; loop as needed)
# ==========================================================

for probe in hessian random fullgrad adam; do
  python loss_landscape_experiment_files/measure_experimentC_surface.py \
    --dataset eurosat \
    --shots 1 \
    --backbone ViT-B/16 \
    --position all \
    --encoder both \
    --lr 1e-4 \
    --full_support_train \
    --checkpoint_epoch 10 \
    --probe ${probe} \
    --grid_size 25 \
    --grid_radius 2.0 \
    --step_scale_mode relative_to_sgd_step \
    --device cuda:0 \
    --checkpoints_logdir ./results_without_aug/experimentC_logs \
    --out_dir ./results_without_aug/experimentC_surfaces

done

# ==========================================================
# 3) Example plotting
# ==========================================================

python loss_landscape_experiment_files/plot_experimentC_surface.py \
  --logdir ./results_without_aug/experimentC_surfaces \
  --run_name surface_dataset_eurosat_step_10_probe_hessian_backbone_ViT-B-16_pos_all_enc_both \
  --outdir ./results_without_aug/experimentC_plots/surface_dataset_eurosat_step_10_probe_hessian_backbone_ViT-B-16_pos_all_enc_both

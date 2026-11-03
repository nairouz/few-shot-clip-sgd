#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Example commands for Experiment F on the three representative datasets.
# Run from the repository root.

python loss_landscape_experiment_files/measure_experimentF_block_sensitivity.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --alpha_min -2.0 \
  --alpha_max 2.0 \
  --alpha_points 21 \
  --eval_batch_size 256 \
  --device cuda:1 \
  --out_dir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentF_block_sensitivity.py \
  --logdir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentF_plots/experimentF_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/measure_experimentF_block_sensitivity.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --alpha_min -2.0 \
  --alpha_max 2.0 \
  --alpha_points 21 \
  --eval_batch_size 256 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentF_block_sensitivity.py \
  --logdir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentF_plots/experimentF_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/measure_experimentF_block_sensitivity.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --alpha_min -2.0 \
  --alpha_max 2.0 \
  --alpha_points 21 \
  --eval_batch_size 256 \
  --device cuda:2 \
  --out_dir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentF_block_sensitivity.py \
  --logdir ./results_without_aug/experimentF_logs \
  --run_name experimentF_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentF_plots/experimentF_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

COMMON_ARGS=(
  --shots 1
  --backbone ViT-B/16
  --position all
  --encoder both
  --lr 1e-4
  --max_epochs 30
  --full_support_train
  --checkpoint_updates 0,1,2,5,10,20,30
  --budget_protocol update_matched
  --out_dir ./results_without_aug/experimentJ_update_control_logs
)

python loss_landscape_experiment_files/measure_experimentJ_noise_scale.py \
  --dataset eurosat \
  --device cuda:0 \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  "${COMMON_ARGS[@]}"

python loss_landscape_experiment_files/plot_experimentJ_noise_scale.py \
  --logdir ./results_without_aug/experimentJ_update_control_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  --outdir ./results_without_aug/experimentJ_update_control_plots/experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update

python loss_landscape_experiment_files/measure_experimentJ_noise_scale.py \
  --dataset oxford_pets \
  --device cuda:1 \
  --run_name experimentJ_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  "${COMMON_ARGS[@]}"

python loss_landscape_experiment_files/plot_experimentJ_noise_scale.py \
  --logdir ./results_without_aug/experimentJ_update_control_logs \
  --run_name experimentJ_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  --outdir ./results_without_aug/experimentJ_update_control_plots/experimentJ_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update

python loss_landscape_experiment_files/measure_experimentJ_noise_scale.py \
  --dataset dtd \
  --device cuda:2 \
  --run_name experimentJ_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  "${COMMON_ARGS[@]}"

python loss_landscape_experiment_files/plot_experimentJ_noise_scale.py \
  --logdir ./results_without_aug/experimentJ_update_control_logs \
  --run_name experimentJ_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  --outdir ./results_without_aug/experimentJ_update_control_plots/experimentJ_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update

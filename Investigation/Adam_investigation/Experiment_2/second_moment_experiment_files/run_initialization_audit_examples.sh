#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

python second_moment_experiment_files/measure_second_moment.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:1 \
  --init_only_audit \
  --out_dir ./results_without_aug/initialization_audit_logs \
  --run_name initialization_audit_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_second_moment.py \
  --logdir ./results_without_aug/initialization_audit_logs \
  --run_name initialization_audit_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ./results_without_aug/initialization_audit_plots/initialization_audit_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_seed_1_pos_all_enc_both_vest_batch-mean-square

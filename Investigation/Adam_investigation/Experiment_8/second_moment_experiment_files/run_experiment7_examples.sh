#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Example Experiment 7 commands for the three main datasets.
# Run from the repository root.

# =========================
# EuroSAT
# =========================
python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
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
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:2 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --ref_mode full_train \
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:0 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/plot_experiment7_temporal_staleness.py \
  --logdir ./results_without_aug/gradient_staleness_logs \
  --run_name_sgd gradient_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5 \
  --run_name_adam gradient_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5 \
  --outdir ./results_without_aug/experiment7_plots/gradient_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam

# =========================
# Oxford-Pets
# =========================
python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --ref_mode full_train \
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:2 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --ref_mode full_train \
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:3 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/plot_experiment7_temporal_staleness.py \
  --logdir ./results_without_aug/gradient_staleness_logs \
  --run_name_sgd gradient_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5 \
  --run_name_adam gradient_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5 \
  --outdir ./results_without_aug/experiment7_plots/gradient_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam

# =========================
# DTD
# =========================
python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --ref_mode full_train \
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:1 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/measure_experiment7_temporal_staleness.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --ref_mode full_train \
  --diag_every 1 \
  --diag_batch_size 256 \
  --lags 1,2,5 \
  --history_dtype float32 \
  --device cuda:3 \
  --out_dir ./results_without_aug/gradient_staleness_logs \
  --run_name gradient_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5

python second_moment_experiment_files/plot_experiment7_temporal_staleness.py \
  --logdir ./results_without_aug/gradient_staleness_logs \
  --run_name_sgd gradient_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_lags_1-2-5 \
  --run_name_adam gradient_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_lags_1-2-5 \
  --outdir ./results_without_aug/experiment7_plots/gradient_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam

#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Example commands for Experiment 5.
# Adjust --device to your machine. The batch sizes match the user's full-support setup.

LOGDIR=./results_without_aug/second_moment_logs
PLOTDIR=./results_without_aug/experiment5_plots

# -----------------------------------------------------------------------------
# EuroSAT
# -----------------------------------------------------------------------------
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
  --device cuda:2 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
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
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:0 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --match_first_step_to_optimizer SGD \
  --match_first_step_ref_lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:1 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_experiment5_update_budget.py \
  --logdir ${LOGDIR} \
  --run_name_sgd second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam_matched second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ${PLOTDIR}/second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam_adammatched

# -----------------------------------------------------------------------------
# Oxford-Pets
# -----------------------------------------------------------------------------
python second_moment_experiment_files/measure_second_moment.py \
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
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:2 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
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
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:3 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --match_first_step_to_optimizer SGD \
  --match_first_step_ref_lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:0 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_experiment5_update_budget.py \
  --logdir ${LOGDIR} \
  --run_name_sgd second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam_matched second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ${PLOTDIR}/second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam_adammatched

# -----------------------------------------------------------------------------
# DTD
# -----------------------------------------------------------------------------
python second_moment_experiment_files/measure_second_moment.py \
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
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:1 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
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
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:3 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/measure_second_moment.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --match_first_step_to_optimizer SGD \
  --match_first_step_ref_lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:0 \
  --out_dir ${LOGDIR} \
  --run_name second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_experiment5_update_budget.py \
  --logdir ${LOGDIR} \
  --run_name_sgd second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam_matched second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_matchstep0_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ${PLOTDIR}/second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_compare_sgd_adam_adammatched

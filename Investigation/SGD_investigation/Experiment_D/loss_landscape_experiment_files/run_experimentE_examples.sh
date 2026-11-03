#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# EuroSAT
python loss_landscape_experiment_files/measure_experimentE_directional_curvature.py \
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
  --eval_batch_size 128 \
  --analysis_microbatch_size 2 \
  --empty_cache_between_phases \
  --power_iters 6 \
  --num_random_dirs 3 \
  --device cuda:1 \
  --out_dir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentE_directional_curvature.py \
  --logdir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentE_plots/experimentE_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --block_epochs 0,30

# Oxford-Pets
python loss_landscape_experiment_files/measure_experimentE_directional_curvature.py \
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
  --eval_batch_size 128 \
  --analysis_microbatch_size 2 \
  --empty_cache_between_phases \
  --power_iters 6 \
  --num_random_dirs 3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentE_directional_curvature.py \
  --logdir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentE_plots/experimentE_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --block_epochs 0,30

# DTD
python loss_landscape_experiment_files/measure_experimentE_directional_curvature.py \
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
  --eval_batch_size 128 \
  --analysis_microbatch_size 2 \
  --empty_cache_between_phases \
  --power_iters 6 \
  --num_random_dirs 3 \
  --device cuda:2 \
  --out_dir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentE_directional_curvature.py \
  --logdir ./results_without_aug/experimentE_logs \
  --run_name experimentE_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentE_plots/experimentE_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --block_epochs 0,30

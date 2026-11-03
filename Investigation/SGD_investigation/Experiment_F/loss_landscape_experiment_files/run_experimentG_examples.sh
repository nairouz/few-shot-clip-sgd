#!/usr/bin/env bash
set -euo pipefail

# Experiment G — Basin connectivity across SGD seeds
# Geometry protocol: 1-shot, full-support training, no augmentation,
# both encoders, all selected blocks trainable.

# -------------------------- EuroSAT --------------------------------------
python loss_landscape_experiment_files/measure_experimentG_connectivity.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --seeds 1,2,3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentG_connectivity.py \
  --logdir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentG_plots/experimentG_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

# ----------------------- Oxford-Pets -------------------------------------
python loss_landscape_experiment_files/measure_experimentG_connectivity.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --seeds 1,2,3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentG_connectivity.py \
  --logdir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentG_plots/experimentG_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

# ----------------------------- DTD ---------------------------------------
python loss_landscape_experiment_files/measure_experimentG_connectivity.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --seeds 1,2,3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentG_connectivity.py \
  --logdir ./results_without_aug/experimentG_logs \
  --run_name experimentG_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentG_plots/experimentG_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seeds_1-2-3_pos_all_enc_both_fullsupport_true

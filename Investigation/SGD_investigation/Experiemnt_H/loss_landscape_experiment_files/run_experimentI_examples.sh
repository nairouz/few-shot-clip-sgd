#!/usr/bin/env bash
set -euo pipefail

# EuroSAT
python loss_landscape_experiment_files/measure_experimentI_feature_drift.py \
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
  --device cuda:1 \
  --out_dir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentI_feature_drift.py \
  --logdir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentI_plots/experimentI_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

# Oxford-Pets
python loss_landscape_experiment_files/measure_experimentI_feature_drift.py \
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
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentI_feature_drift.py \
  --logdir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentI_plots/experimentI_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

# DTD
python loss_landscape_experiment_files/measure_experimentI_feature_drift.py \
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
  --device cuda:2 \
  --out_dir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentI_feature_drift.py \
  --logdir ./results_without_aug/experimentI_logs \
  --run_name experimentI_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentI_plots/experimentI_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

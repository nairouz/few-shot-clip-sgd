#!/usr/bin/env bash
set -euo pipefail

# Experiment H: Support-vs-full landscape mismatch
# Full protocol on the three representative datasets.

python loss_landscape_experiment_files/measure_experimentH_support_full_mismatch.py \
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
  --surface_epochs 0,1,2,5,10,20,30 \
  --eval_batch_size 256 \
  --analysis_microbatch_size 64 \
  --hessian_microbatch_size 4 \
  --hessian_power_iters 6 \
  --surface_points 11 \
  --surface_radius_steps 2.0 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py \
  --logdir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentH_plots/experimentH_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/measure_experimentH_support_full_mismatch.py \
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
  --surface_epochs 0,1,2,5,10,20,30 \
  --eval_batch_size 256 \
  --analysis_microbatch_size 64 \
  --hessian_microbatch_size 4 \
  --hessian_power_iters 6 \
  --surface_points 11 \
  --surface_radius_steps 2.0 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py \
  --logdir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentH_plots/experimentH_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/measure_experimentH_support_full_mismatch.py \
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
  --surface_epochs 0,1,2,5,10,20,30 \
  --eval_batch_size 256 \
  --analysis_microbatch_size 64 \
  --hessian_microbatch_size 4 \
  --hessian_power_iters 6 \
  --surface_points 11 \
  --surface_radius_steps 2.0 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py \
  --logdir ./results_without_aug/experimentH_logs \
  --run_name experimentH_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentH_plots/experimentH_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

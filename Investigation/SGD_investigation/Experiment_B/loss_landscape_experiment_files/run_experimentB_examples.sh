#!/usr/bin/env bash
set -euo pipefail

# EuroSAT
python loss_landscape_experiment_files/measure_experimentB_localstep.py \
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
  --probe_directions sgd_step,random,hessian_top \
  --step_mult_min -1.0 \
  --step_mult_max 3.0 \
  --line_points 41 \
  --hessian_power_iters 6 \
  --hessian_backend auto \
  --hessian_fd_eps 1e-3 \
  --device cuda:1 \
  --out_dir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentB_localstep.py \
  --logdir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentB_plots/experimentB_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

# Oxford-Pets
python loss_landscape_experiment_files/measure_experimentB_localstep.py \
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
  --probe_directions sgd_step,random,hessian_top \
  --step_mult_min -1.0 \
  --step_mult_max 3.0 \
  --line_points 41 \
  --hessian_power_iters 6 \
  --hessian_backend auto \
  --hessian_fd_eps 1e-3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentB_localstep.py \
  --logdir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentB_plots/experimentB_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

# DTD
python loss_landscape_experiment_files/measure_experimentB_localstep.py \
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
  --probe_directions sgd_step,random,hessian_top \
  --step_mult_min -1.0 \
  --step_mult_max 3.0 \
  --line_points 41 \
  --hessian_power_iters 6 \
  --hessian_backend auto \
  --hessian_fd_eps 1e-3 \
  --device cuda:2 \
  --out_dir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

python loss_landscape_experiment_files/plot_experimentB_localstep.py \
  --logdir ./results_without_aug/experimentB_logs \
  --run_name experimentB_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true \
  --outdir ./results_without_aug/experimentB_plots/experimentB_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true

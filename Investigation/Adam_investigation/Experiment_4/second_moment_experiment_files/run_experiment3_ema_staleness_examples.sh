#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Experiment 3 is an Adam-only audit.
# Repeat for beta2 in {0.9, 0.99, 0.999}.

for B2 in 0.9 0.99 0.999; do
  TAG=$(python - <<PY
b2 = float("$B2")
print(str(b2).replace('.', 'p'))
PY
)

  # EuroSAT
  python second_moment_experiment_files/measure_experiment3_ema_staleness.py \
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
    --few_diag_batch_size 5 \
    --ref_diag_batch_size 256 \
    --adam_beta2 $B2 \
    --device cuda:2 \
    --out_dir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

  python second_moment_experiment_files/plot_experiment3_ema_staleness.py \
    --logdir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both \
    --outdir ./results_without_aug/experiment3_ema_staleness_plots/ema_staleness_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

  # Oxford-Pets
  python second_moment_experiment_files/measure_experiment3_ema_staleness.py \
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
    --few_diag_batch_size 16 \
    --ref_diag_batch_size 256 \
    --adam_beta2 $B2 \
    --device cuda:3 \
    --out_dir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

  python second_moment_experiment_files/plot_experiment3_ema_staleness.py \
    --logdir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both \
    --outdir ./results_without_aug/experiment3_ema_staleness_plots/ema_staleness_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

  # DTD
  python second_moment_experiment_files/measure_experiment3_ema_staleness.py \
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
    --few_diag_batch_size 16 \
    --ref_diag_batch_size 256 \
    --adam_beta2 $B2 \
    --device cuda:0 \
    --out_dir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

  python second_moment_experiment_files/plot_experiment3_ema_staleness.py \
    --logdir ./results_without_aug/ema_staleness_logs \
    --run_name ema_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both \
    --outdir ./results_without_aug/experiment3_ema_staleness_plots/ema_staleness_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_b2_${TAG}_seed_1_pos_all_enc_both

done

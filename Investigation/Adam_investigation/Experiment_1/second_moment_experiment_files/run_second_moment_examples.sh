#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Root of the datasets folder. Override with:
#   DATASETS_ROOT=/path/to/datasets bash run_second_moment_examples.sh
ROOT_PATH="${DATASETS_ROOT:-$HOME/datasets}"
LOGDIR="./results_without_aug/second_moment_logs"
PLOTDIR="./results_without_aug/second_moment_plots"

COMMON_ARGS=(
  --root_path "$ROOT_PATH"
  --backbone ViT-B/16
  --position all
  --encoder both
  --traj_optimizer Adam
  --lr 1e-4
  --max_epochs 30
  --full_support_train
  --ref_mode full_train
  --v_estimator batch_mean_square
  --diag_every 1
  --diag_batch_size 256
  --out_dir "$LOGDIR"
)

# ---------------------------------------------------------------------------
# EuroSAT (10 classes -> 10 support images in the 1-shot setting)
# ---------------------------------------------------------------------------
RUN_EUROSAT="second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square"
python second_moment_experiment_files/measure_second_moment.py \
  --dataset eurosat \
  --shots 1 \
  --batch_size 10 \
  --device cuda:1 \
  --run_name "$RUN_EUROSAT" \
  "${COMMON_ARGS[@]}"

python second_moment_experiment_files/plot_second_moment.py \
  --logdir "$LOGDIR" \
  --run_name "$RUN_EUROSAT" \
  --outdir "$PLOTDIR/$RUN_EUROSAT"

# ---------------------------------------------------------------------------
# Oxford-Pets (37 classes -> 37 support images in the 1-shot setting)
# ---------------------------------------------------------------------------
RUN_PETS="second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square"
python second_moment_experiment_files/measure_second_moment.py \
  --dataset oxford_pets \
  --shots 1 \
  --batch_size 37 \
  --device cuda:1 \
  --run_name "$RUN_PETS" \
  "${COMMON_ARGS[@]}"

python second_moment_experiment_files/plot_second_moment.py \
  --logdir "$LOGDIR" \
  --run_name "$RUN_PETS" \
  --outdir "$PLOTDIR/$RUN_PETS"

# ---------------------------------------------------------------------------
# DTD (47 classes -> 47 support images in the 1-shot setting)
# ---------------------------------------------------------------------------
RUN_DTD="second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square"
python second_moment_experiment_files/measure_second_moment.py \
  --dataset dtd \
  --shots 1 \
  --batch_size 47 \
  --device cuda:0 \
  --run_name "$RUN_DTD" \
  "${COMMON_ARGS[@]}"

python second_moment_experiment_files/plot_second_moment.py \
  --logdir "$LOGDIR" \
  --run_name "$RUN_DTD" \
  --outdir "$PLOTDIR/$RUN_DTD"

# ---------------------------------------------------------------------------
# Optional control: repeat the exact same runs on an SGD trajectory instead of
# an Adam trajectory by replacing:
#   --traj_optimizer Adam
# with
#   --traj_optimizer SGD
# in COMMON_ARGS above.
# ---------------------------------------------------------------------------

#!/usr/bin/env bash
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

set -euo pipefail

# Memory-safe full Experiment C runner
# ------------------------------------
# Runs/resumes the complete matrix over:
#   datasets    = {eurosat, oxford_pets, dtd}
#   checkpoints = {0,1,2,5,10,20,30}
#   probes      = {hessian, random, fullgrad, adam}
# and plots every measured surface.
#
# Important A6000 defaults:
#   EVAL_BATCH_SIZE=128       for loss-only surface evaluation
#   GRAD_EVAL_BATCH_SIZE=32   for full-gradient probe computations
#
# If you still OOM on fullgrad, rerun with GRAD_EVAL_BATCH_SIZE=16.
# Completed surfaces are skipped by default, so reruns resume automatically.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"
SURFACE_DEVICE="${SURFACE_DEVICE:-${TRAIN_DEVICE}}"
ROOT_PATH="${ROOT_PATH:-$HOME/datasets}"
OUT_ROOT="${OUT_ROOT:-./results_without_aug}"
GRID_SIZE="${GRID_SIZE:-25}"
GRID_RADIUS="${GRID_RADIUS:-2.0}"
HVP_MODE="${HVP_MODE:-finite_difference}"
HVP_FD_EPS="${HVP_FD_EPS:-1e-3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
GRAD_EVAL_BATCH_SIZE="${GRAD_EVAL_BATCH_SIZE:-32}"
SURFACE_FLUSH_CACHE_EVERY="${SURFACE_FLUSH_CACHE_EVERY:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_CHECKPOINTS="${RUN_CHECKPOINTS:-1}"
RUN_SURFACES="${RUN_SURFACES:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BACKBONE="ViT-B/16"
SHOTS=1
POSITION="all"
ENCODER="both"
LR="1e-4"
MAX_EPOCHS=30
CHECKPOINT_EPOCHS="0,1,2,5,10,20,30"

# Override with e.g. CHECKPOINTS_STR="10 20 30" PROBES_STR="fullgrad adam"
DATASETS_STR="${DATASETS_STR:-eurosat oxford_pets dtd}"
CHECKPOINTS_STR="${CHECKPOINTS_STR:-0 1 2 5 10 20 30}"
PROBES_STR="${PROBES_STR:-hessian random fullgrad adam}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a CHECKPOINTS <<< "${CHECKPOINTS_STR}"
read -r -a PROBES <<< "${PROBES_STR}"

declare -A BATCH_SIZE=(
  [eurosat]=10
  [oxford_pets]=37
  [dtd]=47
)

LOG_DIR="${OUT_ROOT}/experimentC_logs"
SURFACE_DIR="${OUT_ROOT}/experimentC_surfaces"
PLOT_DIR="${OUT_ROOT}/experimentC_plots"
mkdir -p "${LOG_DIR}" "${SURFACE_DIR}" "${PLOT_DIR}"

make_ckpt_run_name() {
  local dataset="$1"
  local bb="${BACKBONE//\//-}"
  printf 'experimentC_dataset_%s_shots_%s_backbone_%s_opt_sgd_seed_1_pos_%s_enc_%s_fullsupport_true' \
    "${dataset}" "${SHOTS}" "${bb}" "${POSITION}" "${ENCODER}"
}

make_surface_run_name() {
  local dataset="$1"
  local epoch="$2"
  local probe="$3"
  local bb="${BACKBONE//\//-}"
  printf 'surface_dataset_%s_step_%s_probe_%s_backbone_%s_pos_%s_enc_%s' \
    "${dataset}" "${epoch}" "${probe}" "${bb}" "${POSITION}" "${ENCODER}"
}

run_python_surface() {
  local dataset="$1"
  local epoch="$2"
  local probe="$3"
  shift 3
  python loss_landscape_experiment_files/measure_experimentC_surface.py \
    --root_path "${ROOT_PATH}" \
    --dataset "${dataset}" \
    --shots "${SHOTS}" \
    --backbone "${BACKBONE}" \
    --position "${POSITION}" \
    --encoder "${ENCODER}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE[${dataset}]}" \
    --full_support_train \
    --checkpoint_epoch "${epoch}" \
    --probe "${probe}" \
    --grid_size "${GRID_SIZE}" \
    --grid_radius "${GRID_RADIUS}" \
    --step_scale_mode relative_to_sgd_step \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --grad_eval_batch_size "${GRAD_EVAL_BATCH_SIZE}" \
    --surface_flush_cache_every "${SURFACE_FLUSH_CACHE_EVERY}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${SURFACE_DEVICE}" \
    --checkpoints_logdir "${LOG_DIR}" \
    --out_dir "${SURFACE_DIR}" \
    "$@"
}

if [[ "${RUN_CHECKPOINTS}" == "1" ]]; then
  echo "[ExperimentC] Phase 1/3: saving SGD checkpoints"
  for dataset in "${DATASETS[@]}"; do
    run_name="$(make_ckpt_run_name "${dataset}")"
    final_ckpt="${LOG_DIR}/checkpoints/${run_name}/epoch_30_trainable.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${final_ckpt}" ]]; then
      echo "[ExperimentC] Skip checkpoints for ${dataset} (found ${final_ckpt})"
      continue
    fi
    python loss_landscape_experiment_files/measure_experimentC_checkpoints.py \
      --root_path "${ROOT_PATH}" \
      --dataset "${dataset}" \
      --shots "${SHOTS}" \
      --backbone "${BACKBONE}" \
      --position "${POSITION}" \
      --encoder "${ENCODER}" \
      --lr "${LR}" \
      --max_epochs "${MAX_EPOCHS}" \
      --batch_size "${BATCH_SIZE[${dataset}]}" \
      --full_support_train \
      --checkpoint_epochs "${CHECKPOINT_EPOCHS}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --device "${TRAIN_DEVICE}" \
      --out_dir "${LOG_DIR}"
  done
fi

if [[ "${RUN_SURFACES}" == "1" ]]; then
  echo "[ExperimentC] Phase 2/3: measuring 2D local surfaces"
  for dataset in "${DATASETS[@]}"; do
    for epoch in "${CHECKPOINTS[@]}"; do
      for probe in "${PROBES[@]}"; do
        surface_run_name="$(make_surface_run_name "${dataset}" "${epoch}" "${probe}")"
        npz_path="${SURFACE_DIR}/${surface_run_name}_surface.npz"
        meta_path="${SURFACE_DIR}/${surface_run_name}_meta.json"
        if [[ "${SKIP_EXISTING}" == "1" && -f "${npz_path}" && -f "${meta_path}" ]]; then
          echo "[ExperimentC] Skip surface ${surface_run_name}"
          continue
        fi

        echo "[ExperimentC] Measure ${surface_run_name} | eval_bs=${EVAL_BATCH_SIZE} grad_bs=${GRAD_EVAL_BATCH_SIZE} device=${SURFACE_DEVICE}"
        extra_probe_args=()
        if [[ "${probe}" == "hessian" ]]; then
          extra_probe_args+=(--hvp_mode "${HVP_MODE}" --hvp_fd_eps "${HVP_FD_EPS}")
        fi
        run_python_surface "${dataset}" "${epoch}" "${probe}" "${extra_probe_args[@]}"
      done
    done
  done
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
  echo "[ExperimentC] Phase 3/3: plotting surfaces"
  for dataset in "${DATASETS[@]}"; do
    for epoch in "${CHECKPOINTS[@]}"; do
      for probe in "${PROBES[@]}"; do
        surface_run_name="$(make_surface_run_name "${dataset}" "${epoch}" "${probe}")"
        npz_path="${SURFACE_DIR}/${surface_run_name}_surface.npz"
        meta_path="${SURFACE_DIR}/${surface_run_name}_meta.json"
        outdir="${PLOT_DIR}/${surface_run_name}"
        if [[ ! -f "${npz_path}" || ! -f "${meta_path}" ]]; then
          echo "[ExperimentC] Skip plot ${surface_run_name} (surface not found)"
          continue
        fi
        if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/expC_surface_contours.png" && -f "${outdir}/expC_surface_cross_sections.png" ]]; then
          echo "[ExperimentC] Skip plot ${surface_run_name}"
          continue
        fi
        python loss_landscape_experiment_files/plot_experimentC_surface.py \
          --logdir "${SURFACE_DIR}" \
          --run_name "${surface_run_name}" \
          --outdir "${outdir}"
      done
    done
  done
fi

echo "[ExperimentC] Done."
echo "  Checkpoints: ${LOG_DIR}"
echo "  Surfaces:    ${SURFACE_DIR}"
echo "  Plots:       ${PLOT_DIR}"

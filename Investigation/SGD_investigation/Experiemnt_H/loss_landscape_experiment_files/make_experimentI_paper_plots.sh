#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${1:-./results_without_aug/experimentI_logs}"
OUTDIR="${2:-./results_without_aug/experimentI_paper_plots}"
FIGDIR="${3:-./figures/sgd_success}"

mkdir -p "$OUTDIR" "$FIGDIR"

for cfg in "$LOGDIR"/experimentI_*_config.json; do
  [ -e "$cfg" ] || continue
  run_name="$(basename "$cfg" _config.json)"
  dataset="$(python - <<'PY' "$cfg"
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg['dataset'])
PY
)"
  per_run_out="$OUTDIR/$run_name"
  mkdir -p "$per_run_out"
  python loss_landscape_experiment_files/plot_experimentI_feature_drift_paper.py \
    --logdir "$LOGDIR" \
    --run_name "$run_name" \
    --outdir "$per_run_out"

  cp "$per_run_out/expI_${dataset}_losses_and_gaps.pdf" "$FIGDIR/"
  cp "$per_run_out/expI_${dataset}_feature_drift.pdf" "$FIGDIR/"
  cp "$per_run_out/expI_${dataset}_block_parameter_drift.pdf" "$FIGDIR/"
  cp "$per_run_out/expI_${dataset}_losses_and_gaps.png" "$FIGDIR/" || true
  cp "$per_run_out/expI_${dataset}_feature_drift.png" "$FIGDIR/" || true
  cp "$per_run_out/expI_${dataset}_block_parameter_drift.png" "$FIGDIR/" || true
  echo "[Experiment I] Generated paper plots for $dataset"
done

echo "Saved Experiment I paper plots to: $OUTDIR"
echo "Copied selected PDFs to: $FIGDIR"

#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${1:-./results_without_aug/experimentH_logs}"
OUTDIR="${2:-./results_without_aug/experimentH_paper_plots}"
FIGDIR="${3:-./figures/sgd_success}"

mkdir -p "$OUTDIR" "$FIGDIR"

for cfg in "$LOGDIR"/experimentH_*_config.json; do
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
  python loss_landscape_experiment_files/plot_experimentH_support_full_mismatch_paper.py \
    --logdir "$LOGDIR" \
    --run_name "$run_name" \
    --outdir "$per_run_out"

  cp "$per_run_out/expH_${dataset}_alignment.pdf" "$FIGDIR/"
  cp "$per_run_out/expH_${dataset}_surface_text.pdf" "$FIGDIR/"
  cp "$per_run_out/expH_${dataset}_surface_vision.pdf" "$FIGDIR/"
  cp "$per_run_out/expH_${dataset}_alignment.png" "$FIGDIR/" || true
  cp "$per_run_out/expH_${dataset}_surface_text.png" "$FIGDIR/" || true
  cp "$per_run_out/expH_${dataset}_surface_vision.png" "$FIGDIR/" || true
  echo "[Experiment H] Generated paper plots for $dataset"
done


python loss_landscape_experiment_files/plot_experimentH_directional_curvature_ratio.py \
  --logdir "$LOGDIR" \
  --outdir "$OUTDIR"

cp "$OUTDIR/expH_directional_curvature_ratio.pdf" "$FIGDIR/"
cp "$OUTDIR/expH_directional_curvature_ratio.png" "$FIGDIR/" || true
echo "[Experiment H] Generated directional-curvature ratio plot"

echo "Saved Experiment H paper plots to: $OUTDIR"
echo "Copied selected PDFs to: $FIGDIR"

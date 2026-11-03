#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <logdir> <plot_outdir> <paper_figdir>"
  exit 1
fi

LOGDIR="$1"
PLOTDIR="$2"
FIGDIR="$3"
mkdir -p "$PLOTDIR" "$FIGDIR"

plot_one() {
  local RUN_NAME="$1"
  local SHORT="$2"
  local OUT="$PLOTDIR/$RUN_NAME"
  mkdir -p "$OUT"

  python loss_landscape_experiment_files/plot_experimentA_interpolation.py \
    --logdir "$LOGDIR" \
    --run_name "$RUN_NAME" \
    --outdir "$OUT" \
    --paper_prefix "expA_${SHORT}"

  cp "$OUT/expA_${SHORT}_interpolation.pdf" "$FIGDIR/expA_${SHORT}_interpolation.pdf"
  cp "$OUT/expA_${SHORT}_interpolation.png" "$FIGDIR/expA_${SHORT}_interpolation.png"
  cp "$OUT/expA_${SHORT}_projection.pdf" "$FIGDIR/expA_${SHORT}_projection.pdf"
  cp "$OUT/expA_${SHORT}_projection.png" "$FIGDIR/expA_${SHORT}_projection.png"
}

plot_one experimentA_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true eurosat
plot_one experimentA_dataset_oxford_pets_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true oxford
plot_one experimentA_dataset_dtd_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_fullsupport_true dtd

echo "Saved paper figures to: $FIGDIR"

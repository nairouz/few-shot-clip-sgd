# Experiment H — Support-vs-full landscape mismatch

This experiment studies whether SGD works in few-shot CLIP adaptation because it
stays in regions where the **support** and **full-train** losses are still
geometrically aligned.

## Protocol

- Backbone: `ViT-B/16`
- Shots: `1`
- Training trajectory: `SGD`
- Full-support training: one batch per epoch (`--full_support_train`)
- No augmentation (geometry experiment)
- Train both encoders with all selected blocks trainable
- Datasets: `eurosat`, `oxford_pets`, `dtd`
- Saved / analysed checkpoints:
  `t ∈ {0,1,2,5,10,20,30}`

## What is measured at each saved checkpoint

1. **Support-vs-full gradient alignment**
   - cosine between the support gradient and the full-train gradient,
   - sign agreement,
   - gradient norm ratio.

2. **Support-vs-full Hessian mismatch (per encoder)**
   - dominant Hessian eigenvalue for the support loss and for the full-train loss,
   - cosine between the dominant support and full eigendirections,
   - directional curvature along the support gradient and along the full gradient.

3. **Local 2D support/full loss surfaces**
   - evaluated in the plane spanned by
     - the support gradient, and
     - the orthogonal component of the full gradient.

## Files written by the measurement script

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_encoder_alignment.csv`
- `<run_name>_block_alignment.csv`
- `<run_name>_surface.csv`
- `<run_name>_summary.json`
- `checkpoints/<run_name>/epoch_XX_trainable.pt`

## Companion plotting script

`loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py`

## Practical note

Experiment H is substantially more expensive than the interpolation experiment
because it performs second-order diagnostics at every saved checkpoint. If you
want a quick smoke test, reduce:

- `--hessian_power_iters` (e.g. `3` instead of `6`)
- `--surface_points` (e.g. `7` instead of `11`)
- `--surface_epochs` (e.g. `0,10,30` instead of all checkpoints)

## Directional curvature-scale ratio plot

The patched code adds a paper-ready plotting script:

```bash
python loss_landscape_experiment_files/plot_experimentH_directional_curvature_ratio.py \
  --logdir ./results_without_aug/experimentH_logs \
  --outdir ./results_without_aug/experimentH_paper_plots
```

This script generates:

- `expH_directional_curvature_ratio.pdf`
- `expH_directional_curvature_ratio.png`

The plotted metric is

```math
\log_{10} Q_{\kappa}^{(e)}(t),
```

where

```math
Q_{\kappa}^{(e)}(t)
=
\frac{
    |\kappa_{\mathrm{full}}^{(e)}(\widehat{u}_{\mathrm{full},t}^{(e)};t)|+10^{-12}
}{
    |\kappa_{\mathrm{few}}^{(e)}(\widehat{u}_{\mathrm{full},t}^{(e)};t)|+10^{-12}
}.
```

Values above zero mean that the support loss underestimates the full-train
curvature along the normalized full-train gradient direction.

The helper script `make_experimentH_paper_plots.sh` now calls this plotting
script automatically and copies the resulting PDF/PNG into the paper figure
folder.

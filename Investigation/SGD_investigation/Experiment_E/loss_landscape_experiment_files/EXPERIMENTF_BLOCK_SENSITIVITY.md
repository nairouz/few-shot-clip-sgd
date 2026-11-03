# Experiment F — Per-block loss landscape and sensitivity map

Experiment F asks:

> Which blocks / encoders are safe to move, and which are fragile, along the
> actual SGD geometry of few-shot CLIP adaptation?

The measurement script `measure_experimentF_block_sensitivity.py` implements the
exact protocol previously recommended for Experiment F:

- CLIP ViT-B/16 (default, but configurable)
- 1-shot adaptation
- SGD trajectory
- full-support training (one batch per epoch with `--full_support_train`)
- no augmentation
- both encoders trainable
- all selected blocks trainable
- checkpoints saved at `t in {0,1,2,5,10,20,30}` by default

At each checkpoint `t`, the script computes the full-support few-shot gradient
and forms the actual SGD block step

`Delta_t^(ell) = - lr * g_t^(ell)`

where only block `ell` is kept. It then probes the local loss landscape

`L(Theta_t + alpha * Delta_t^(ell))`

for a symmetric alpha grid.

## Main outputs

For each run, the script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_block_landscape.csv`
- `<run_name>_block_summary.csv`
- `<run_name>_summary.json`
- `checkpoints/<run_name>/epoch_XX_trainable.pt`

The plotting script `plot_experimentF_block_sensitivity.py` produces paper-style
figures using the same visual style as the reference geometry experiment code.

## Key summary metrics per block

- `full_delta_at1`: `L_full(alpha=1) - L_full(0)`
- `test_delta_at1`: `L_test(alpha=1) - L_test(0)`
- `full_curvature0`, `test_curvature0`: finite-difference local curvature at alpha=0
- `full_safe_radius`, `test_safe_radius`: largest positive alpha that keeps the
  corresponding loss within a small tolerance of the checkpoint loss
- `full_alpha_min`, `test_alpha_min`: alpha minimizing the probed loss along that
  block direction on the chosen grid
- `gap_full_minus_few_at1`: support/full mismatch at alpha=1

These metrics jointly identify:
- blocks that are safe to move,
- blocks that are fragile,
- blocks whose SGD step is conservative (`alpha*_full > 1`), and
- blocks where support and full-train objectives disagree strongly.


## Paper summary figure

For the NeurIPS paper, use
`plot_experimentF_block_sensitivity_paper.py` together with
`make_experimentF_paper_plots.sh`. The paper figure reports the checkpoint-averaged
block diagnostics
`\overline{\Delta}_{\mathrm{full}}^{(\ell)}(1)` and
`\overline{\Gamma}^{(\ell)}(1)` with the notation used in the LaTeX source.

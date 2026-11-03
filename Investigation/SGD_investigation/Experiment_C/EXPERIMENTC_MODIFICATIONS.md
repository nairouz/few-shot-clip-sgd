# Modifications added for Experiment C

This codebase adds a standalone implementation of **Experiment C — 2D local surfaces around SGD checkpoints**.

## Added files

- `loss_landscape_experiment_files/experimentC_utils.py`
  - shared helpers for datasets, trainable-state save/load, gradients, second moments, Hessian power iteration, and perturbation-state construction.

- `loss_landscape_experiment_files/measure_experimentC_checkpoints.py`
  - runs the exact SGD trajectory used for the geometry study and saves trainable checkpoints at `t in {0,1,2,5,10,20,30}`.

- `loss_landscape_experiment_files/measure_experimentC_surface.py`
  - loads one checkpoint and evaluates the 2D surface for a selected probe direction.

- `loss_landscape_experiment_files/plot_experimentC_surface.py`
  - plots contour maps and central cuts in the same presentation style as Experiment A.

- `loss_landscape_experiment_files/EXPERIMENTC_SURFACES.md`
  - usage notes.

- `loss_landscape_experiment_files/run_experimentC_examples.sh`
  - ready-to-run examples.

## Why the implementation is split in two stages

Experiment C is much heavier than Experiment A. Running the full trajectory and all 2D surfaces in one pass would be extremely wasteful, because:

- training checkpoints need to be produced only once,
- each `(checkpoint, probe)` surface can then be evaluated independently.

So the workflow is:

1. save SGD checkpoints once,
2. measure local surfaces checkpoint-by-checkpoint,
3. plot them.

## Definition of the two surface directions

At checkpoint `Theta_t`:

- `d_t` is the **actual local SGD descent direction**. Since the trajectory uses plain SGD without momentum, this is the normalized negative few-shot gradient at the checkpoint.
- `e_t` is an orthogonal probe direction:
  - `hessian`: top few-shot Hessian eigenvector, orthogonalized to `d_t`
  - `random`: random direction, orthogonalized to `d_t`
  - `fullgrad`: full-train gradient direction, orthogonalized to `d_t`
  - `adam`: virtual Adam-like direction `m_few / sqrt(v_few + eps)`, orthogonalized to `d_t`

## Surface parameterization

The perturbation is parameterized in units of the actual SGD step norm:

`Theta_t(alpha, beta) = Theta_t + ||Delta_Theta_t^SGD|| * (alpha d_t + beta e_t)`

when `--step_scale_mode relative_to_sgd_step` is used.

This makes the surface comparable across checkpoints.

## Figure style

The plotting script intentionally reuses the same typography and canvas style as Experiment A:

- transparent background,
- identical font sizes / weights,
- shared global x/y labels,
- same general layout rhythm.

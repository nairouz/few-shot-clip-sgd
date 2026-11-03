# Experiment J modifications

This codebase extends the standalone loss-landscape experiment under
`loss_landscape_experiment_files/` for **Experiment J — Noise-scale / batch-size
landscape study**.

## New functionality
The original Experiment J compared SGD variants under an **epoch-matched**
budget, which means the mini-batch variants made more optimizer updates than the
full-batch baseline.

This revision integrates the follow-up **update-matched fairness control**
directly into the same pipeline.

## Main code changes
- `measure_experimentJ_noise_scale.py`
  - added `--budget_protocol {epoch_matched,update_matched}`
  - added `--checkpoint_updates`
  - added `--target_total_updates`
  - tracks `global_update`, `epochs_completed`, `examples_seen`,
    `example_budget_ratio`, and `update_budget_ratio`
  - saves update-based checkpoints for update-matched runs
  - writes the resolved budgeting protocol into the config and CSV logs

- `plot_experimentJ_noise_scale.py`
  - automatically chooses the x-axis based on the budgeting protocol
  - remains compatible with the original epoch-matched logs
  - adds `expJ_budget_accounting.(png|pdf)` for the new fairness-control run

- `EXPERIMENTJ_NOISE_SCALE.md`
  - documents both budgeting modes

- `run_experimentJ_examples.sh`
  - now launches the new update-matched control by default

The baseline training code in `main.py`, `train.py`, `run_utils.py`, and the
CLIP / dataset modules remains unchanged.


## J3 follow-up added

This codebase now also supports `budget_protocol=example_matched`, which equalizes the total number of support examples processed across variants. This is the complementary control to J2 (`update_matched`). Together, the three protocols separate the effects of epoch count, update count, and support exposure.

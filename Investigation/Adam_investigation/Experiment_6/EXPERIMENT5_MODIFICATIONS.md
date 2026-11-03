# Experiment 5 modifications

This codebase extends the Experiment 2 bundle with a minimal implementation of
**Experiment 5 — Update-budget matched Adam vs SGD**.

## Files added

- `second_moment_experiment_files/plot_experiment5_update_budget.py`
- `second_moment_experiment_files/EXPERIMENT5_UPDATE_BUDGET.md`
- `second_moment_experiment_files/run_experiment5_examples.sh`

## Files modified

- `second_moment_experiment_files/measure_second_moment.py`

## Main idea

The measurement script now logs the exact **planned optimizer update** before
`optimizer.step()` and can optionally rescale the learning rate at step 0 so
that Adam matches SGD's first-step global relative update norm.

This directly supports the question:

> Is Adam bad mainly because of wrong scale, or because of wrong geometry?

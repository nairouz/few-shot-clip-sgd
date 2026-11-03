# Experiment 6 additions

This package extends the Experiment 2 codebase with a feature-drift audit.

## New files
- `second_moment_experiment_files/measure_feature_drift.py`
- `second_moment_experiment_files/plot_experiment6_feature_drift.py`
- `second_moment_experiment_files/EXPERIMENT6_FEATURE_DRIFT.md`
- `second_moment_experiment_files/run_experiment6_examples.sh`

## Design philosophy
The implementation intentionally avoids touching the existing Experiment 2
measurement logic. Instead, it reuses its dataset/model-loading helpers and adds
new scripts dedicated to the Experiment 6 protocol.

This keeps the code changes minimal and makes it easy to run Experiment 2 and
Experiment 6 side by side.

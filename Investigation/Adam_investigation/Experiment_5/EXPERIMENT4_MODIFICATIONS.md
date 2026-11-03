# Experiment 4 modifications

This patch adds Experiment 4 with **new files only**. No existing Experiment 2
measurement or plotting files are modified.

## Added files

- `second_moment_experiment_files/measure_experiment4_counterfactuals.py`
- `second_moment_experiment_files/plot_experiment4_counterfactuals.py`
- `second_moment_experiment_files/EXPERIMENT4_COUNTERFACTUALS.md`
- `second_moment_experiment_files/run_experiment4_examples.sh`

## Why a new measurement file?

Experiment 4 needs extra state that Experiment 2 does not store:

- the few-shot denominator frozen at initialization,
- lagged few-shot denominators,
- temporary norm-matched probe updates and their resulting losses.

Adding a separate measurement script keeps the original Experiment 2 pipeline
unchanged and avoids unrequired modifications.

## Measurement outputs

Experiment 4 writes:

- `<run_name>_config.json`
- `<run_name>_counterfactual_per_iteration.csv`
- `<run_name>_counterfactual_per_block.csv`
- `<run_name>_probe_per_iteration.csv`

The plotting script consumes those files directly.

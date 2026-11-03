# Experiment I modifications

This codebase adds a new folder:

- `loss_landscape_experiment_files/`

with four files:

1. `measure_experimentI_feature_drift.py`
   - trains an SGD few-shot CLIP trajectory using the geometry protocol,
   - saves trainable-only checkpoints at user-specified epochs,
   - measures support/full/test losses and accuracies, parameter drift,
     image/text feature drift, and class-margin drift.

2. `plot_experimentI_feature_drift.py`
   - reads the CSV/JSON bundle from the measurement script,
   - produces publication-style figures using the same visual style as the
     uploaded Experiment-2-style plotting code.

3. `EXPERIMENTI_FEATURE_DRIFT.md`
   - short usage/documentation file.

4. `run_experimentI_examples.sh`
   - example commands for EuroSAT, Oxford-Pets, and DTD.

The original few-shot training code is left intact; Experiment I is added as a
standalone geometry experiment so the baseline training path remains unchanged.

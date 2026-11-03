# Experiment A modifications

This codebase adds a new folder:

- `loss_landscape_experiment_files/`

with four files:

1. `measure_experimentA_interpolation.py`
   - trains an SGD few-shot CLIP trajectory using the geometry protocol,
   - saves trainable-only checkpoints at user-specified epochs,
   - measures the 1D interpolation losses from the pretrained initialization to the final SGD solution.

2. `plot_experimentA_interpolation.py`
   - reads the CSV/JSON bundle from the measurement script,
   - produces publication-style figures using the same visual style as Experiment 2.

3. `EXPERIMENTA_INTERPOLATION.md`
   - short usage/documentation file.

4. `run_experimentA_examples.sh`
   - example commands for EuroSAT, Oxford-Pets, and DTD.

The original few-shot training code is left intact; Experiment A is added as a
standalone geometry experiment so the baseline training path remains unchanged.

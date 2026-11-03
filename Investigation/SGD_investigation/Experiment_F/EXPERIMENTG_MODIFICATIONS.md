# Experiment G modifications

This codebase adds a new folder:

- `loss_landscape_experiment_files/`

with four files:

1. `measure_experimentG_connectivity.py`
   - trains one full-support SGD trajectory per requested seed,
   - saves trainable-only checkpoints at the requested epochs for every seed,
   - measures pairwise linear interpolation losses between final seed solutions.

2. `plot_experimentG_connectivity.py`
   - reads the CSV/JSON bundle from the measurement script,
   - produces publication-style figures using the same visual style as Experiment 2.

3. `EXPERIMENTG_CONNECTIVITY.md`
   - short usage/documentation file.

4. `run_experimentG_examples.sh`
   - example commands for EuroSAT, Oxford-Pets, and DTD.

The original few-shot training code is left intact; Experiment G is added as a
standalone geometry experiment so the baseline training path remains unchanged.

# Experiment F modifications

This codebase adds a new folder:

- `loss_landscape_experiment_files/`

with four files:

1. `measure_experimentF_block_sensitivity.py`
   - trains the SGD few-shot CLIP trajectory using the geometry protocol,
   - saves trainable-only checkpoints at user-specified epochs,
   - measures the local blockwise loss landscape around each checkpoint along the
     actual SGD block direction.

2. `plot_experimentF_block_sensitivity.py`
   - reads the CSV/JSON bundle from the measurement script,
   - produces publication-style figures using the same visual style as the
     reference geometry experiment archive.

3. `EXPERIMENTF_BLOCK_SENSITIVITY.md`
   - short usage/documentation file.

4. `run_experimentF_examples.sh`
   - example commands for EuroSAT, Oxford-Pets, and DTD.

The original few-shot training code is left intact; Experiment F is added as a
standalone geometry experiment so the baseline training path remains unchanged.

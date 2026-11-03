# Experiment E modifications

This codebase adds a new folder:

- `loss_landscape_experiment_files/`

with four new files:

1. `measure_experimentE_directional_curvature.py`
   - trains an SGD few-shot CLIP trajectory with the geometry protocol,
   - saves trainable-only checkpoints at user-specified epochs,
   - measures directional curvature `kappa(d)=d^T H d` for the requested probe directions.

2. `plot_experimentE_directional_curvature.py`
   - reads the CSV/JSON bundle from the measurement script,
   - produces publication-style figures using the same visual style as the previous experiment code.

3. `EXPERIMENTE_DIRECTIONAL_CURVATURE.md`
   - short usage/documentation file.

4. `run_experimentE_examples.sh`
   - example commands for EuroSAT, Oxford-Pets, and DTD.

The original few-shot training code is left intact; Experiment E is added as a
standalone geometry experiment so the baseline training path remains unchanged.


## A6000-safe patch
This version adds internal micro-batching to the loss / gradient / curvature routines and streams probe directions from CPU to GPU on the fly during HVP computation. These changes reduce peak GPU memory without changing the underlying directional-curvature quantities.

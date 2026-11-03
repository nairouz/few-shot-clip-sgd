# Experiment H modifications

This package extends the original `code.zip` with a new standalone experiment:

- `loss_landscape_experiment_files/measure_experimentH_support_full_mismatch.py`
- `loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py`
- `loss_landscape_experiment_files/clip_block_utils.py`
- `loss_landscape_experiment_files/EXPERIMENTH_SUPPORT_FULL_MISMATCH.md`
- `loss_landscape_experiment_files/run_experimentH_examples.sh`

No existing training code was changed. The base `main.py` / `train.py` workflow
still behaves exactly as before. Experiment H is implemented as a **standalone
analysis module** that reuses the same CLIP model-loading, dataset, and
trainable-block selection logic.

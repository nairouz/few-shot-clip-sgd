# Experiment J modifications

This codebase adds a standalone loss-landscape experiment under
`loss_landscape_experiment_files/` for **Experiment J — Noise-scale / batch-size
landscape study**.

Added files:
- `loss_landscape_experiment_files/measure_experimentJ_noise_scale.py`
- `loss_landscape_experiment_files/plot_experimentJ_noise_scale.py`
- `loss_landscape_experiment_files/EXPERIMENTJ_NOISE_SCALE.md`
- `loss_landscape_experiment_files/run_experimentJ_examples.sh`

The baseline few-shot training code in `main.py`, `train.py`, `run_utils.py`,
and the dataset/CLIP modules is left unchanged. Experiment J is implemented as a
standalone measurement / plotting pipeline so the original SGD training script
remains intact.

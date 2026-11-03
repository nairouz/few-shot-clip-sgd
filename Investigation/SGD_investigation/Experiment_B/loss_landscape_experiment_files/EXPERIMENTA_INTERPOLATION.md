# Experiment A — 1D interpolation from initialization to final SGD solution

Experiment A asks:

> Does SGD move into a broad basin, or into a narrow support-fitting basin?

The script `measure_experimentA_interpolation.py` implements the exact training
protocol requested for the geometry experiment:

- CLIP ViT-B/16 (default, but configurable)
- 1-shot adaptation
- SGD trajectory
- full-support training (one batch per epoch with `--full_support_train`)
- no augmentation
- both encoders trainable
- all selected blocks trainable
- checkpoints saved at `t in {0,1,2,5,10,20,30}` by default

The script trains the few-shot model, saves trainable-only checkpoints, and then
measures the losses along the interpolation line

`Theta(alpha) = (1-alpha) Theta_0 + alpha Theta_final`

where only the trainable parameters are interpolated.

## Output files

For each run, the script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_checkpoint_projection.csv`
- `<run_name>_interpolation.csv`
- `<run_name>_summary.json`
- `checkpoints/<run_name>/epoch_XX_trainable.pt`

The plotting script `plot_experimentA_interpolation.py` produces:

- `expA_interpolation_losses.(png|pdf)`
- `expA_checkpoint_projection.(png|pdf)`

The figure style (font sizes, transparent background, axis treatment, global
labels, line styling) is matched to the Experiment 2 plotting script.

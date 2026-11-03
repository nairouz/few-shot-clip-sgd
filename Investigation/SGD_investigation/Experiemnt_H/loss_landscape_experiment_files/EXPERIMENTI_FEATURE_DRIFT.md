# Experiment I — Feature-drift landscape

Experiment I asks:

> Does SGD remain in a low-drift region of CLIP feature space, or does it move
> into a narrow support-fitting basin during few-shot adaptation?

The script `measure_experimentI_feature_drift.py` implements the requested
training protocol:

- CLIP ViT-B/16 (default, but configurable)
- 1-shot adaptation
- SGD trajectory
- full-support training (one batch per epoch with `--full_support_train`)
- no augmentation
- both encoders trainable
- all selected blocks trainable
- checkpoints saved at `t in {0,1,2,5,10,20,30}` by default

At each checkpoint, the script measures:

- support / full-train / test loss and accuracy,
- relative parameter drift from initialization,
- per-encoder and per-block parameter drift,
- image embedding drift on support / full-train / test,
- text embedding drift,
- class-margin drift on support / full-train / test.

## Output files

For each run, the script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_block_drift.csv`
- `<run_name>_summary.json`
- `checkpoints/<run_name>/epoch_XX_trainable.pt`

The plotting script `plot_experimentI_feature_drift.py` produces:

- `expI_losses_and_gaps.(png|pdf)`
- `expI_feature_drift.(png|pdf)`
- `expI_block_parameter_drift.(png|pdf)`

The figure style (font sizes, transparent background, axis treatment, global
labels, line styling) is matched to the uploaded Experiment-2-style plotting
code.

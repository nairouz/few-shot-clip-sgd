# Experiment G — Basin connectivity across SGD seeds

This module adds a standalone geometry experiment for **basin connectivity across SGD seeds** in few-shot CLIP adaptation.

## Protocol implemented

- CLIP ViT-B/16 by default.
- 1-shot adaptation by default.
- Full-support training (`--full_support_train`): one batch per epoch.
- No augmentation by default.
- Train both encoders, all selected blocks trainable.
- Save checkpoints at `t in {0,1,2,5,10,20,30}` for every seed.
- Train one SGD trajectory per requested seed.
- Interpolate linearly between **final seed solutions** and evaluate:
  - pair-support loss: `0.5 * (L_few^i + L_few^j)`
  - full-train loss
  - test loss

Because this geometry setup is otherwise deterministic, changing the seed mainly changes the sampled **few-shot support task**. Therefore the pair-support loss is defined as the mean support loss of the two endpoint tasks.

## Main scripts

- `measure_experimentG_connectivity.py`
- `plot_experimentG_connectivity.py`
- `run_experimentG_examples.sh`

## Output files

For a run name `<run_name>` the measurement script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_pairwise_interpolation.csv`
- `<run_name>_pairwise_summary.csv`
- `checkpoints/<run_name>/seed_XX/epoch_YY_trainable.pt`

The plot script writes:

- `expG_seed_trajectories.(png|pdf)`
- `expG_pairwise_interpolation_losses.(png|pdf)`
- `expG_pairwise_barriers.(png|pdf)`

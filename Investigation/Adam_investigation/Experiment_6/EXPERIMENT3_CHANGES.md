# Changes for Experiment 3 (few-shot vs full-data second moment)

This modified codebase adds a paper-style implementation of the second-moment
experiment used to analyse why Adam collapses in few-shot CLIP adaptation.

## New / updated files

- `second_moment_experiment_files/measure_second_moment.py`
  - standalone diagnostic runner
  - supports full-support training trajectory
  - supports `full_train` or `kshot` reference sets
  - supports two second-moment estimators:
    - `batch_mean_square`
    - `full_support_square`
  - writes a complete log bundle (`config.json`, `per_iteration.csv`, `per_block.csv`, `hists_and_curves.npz`)

- `second_moment_experiment_files/plot_second_moment.py`
  - same visual style as the ECCV sign-correctness plots
  - 1x2 vision/text figures with global x/y labels

- `second_moment_experiment_files/clip_block_utils.py`
  - block mapping helper used by the second-moment diagnostic

- `second_moment_experiment_files/run_second_moment_examples.sh`
  - ready-to-run commands for EuroSAT, Oxford-Pets, and DTD

- `second_moment_experiment_files/SECOND_MOMENT_EXPERIMENT.md`
  - detailed explanation of the protocol and exact commands

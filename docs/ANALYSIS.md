# Analysis utilities

Experimental result files and generated tables are not distributed with this repository. Quantitative results are reported in the paper and supplementary material. The analysis code processes local experiment outputs.

## Training-run analysis

The [shot-scaling analysis scripts](../Investigation/ShotScaling/analysis/) aggregate local CLIP runs and generate diagnostic plots. The [SigLIP analysis scripts](../FT/SigLIP/analysis/) aggregate local accuracy, preconditioning, and update-budget measurements. Their command-line options specify the input and output paths. The [experiment index](EXPERIMENTS.md) lists the remaining measurement and plotting scripts.

## Table-generation scripts

The utilities in [`scripts/analysis/`](../scripts/analysis/) operate on locally prepared CSV exports. They require pandas and NumPy. Their input filenames and column names follow the readers in each script; raw training directories are not direct inputs to these utilities.

Local exports belong in `results/inputs/`. The following commands run from the repository root after those exports have been prepared:

```bash
python scripts/analysis/recompute_tables.py
python scripts/analysis/write_manuscript_rows.py
python scripts/analysis/recompute_p1_complete.py
```

The scripts write local outputs to `results/generated/`. `write_manuscript_rows.py` uses files produced by `recompute_tables.py`. The entire `results/` directory is ignored by Git.

| Script | Local input exports |
| --- | --- |
| `recompute_tables.py` | `p2_accuracy_per_seed.csv`, `p5_accuracy_per_seed_verified.csv`, `p2_per_seed_fixed_horizon_and_validation_grid.csv`, `p2_mechanism_per_seed_scope.csv`, `p5_mechanism_diagnostic_per_seed_scope.csv`, `p2_mechanism_by_shot.csv`, `p5_first_step_budgets_per_seed_scope.csv`, `p2_accuracy_primary_paired_bootstrap.csv`, `p5_accuracy_primary_paired_bootstrap.csv` |
| `write_manuscript_rows.py` | `p1_closure_by_shot.csv`, `p1_closure_overall.csv`, `p1_lr_warmup_vs_sgd.csv`, `p1_lr_warmup_vs_sgd_bootstrap.csv`, `group2_trajectory_overall.csv`, `group2_exact_trajectory_metric_matrix.csv`, `group2_trajectory_per_run.csv` |
| `recompute_p1_complete.py` | `p1_complete_per_seed.csv`, `p1_complete_pairing.csv` |

## Aggregation conventions

Accuracies are percentages. Sample standard deviations use `ddof=1`. Signed gap recovery is computed from macro-mean differences as `100 × (Adam-M0 − Adam)/(SGD − Adam)` and is not clipped at 100%. Dataset-cluster confidence intervals retain paired shots and seeds. Checkpoints are not treated as independent samples.

An epoch-300 average includes only runs with an observed epoch-300 value, and its sample count is reported. Peak and last-observed summaries describe the recorded trajectories. Peak accuracy is not a checkpoint-selection rule.

The complete every-step matching grid has no separate training entry point in this release. Its table utility requires local per-run exports and their pairing checks. First-step matching is implemented in `Investigation/Adam_investigation/Experiment_6` and in the shot-scaling and SigLIP experiments.

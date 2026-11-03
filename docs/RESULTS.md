# Results and table reproduction

The numerical exports in [`results/`](../results/) cover update-budget matching, learning-rate and warmup controls, shot scaling, SigLIP adaptation, and long-horizon comparisons. The calculation scripts in [`scripts/analysis/`](../scripts/analysis/) reproduce the aggregation tables from these exports.

## Table generation

The following commands run from the repository root:

```bash
python scripts/analysis/recompute_tables.py
python scripts/analysis/write_manuscript_rows.py
python scripts/analysis/recompute_p1_complete.py
```

The scripts read `results/inputs/` and write `results/generated/`. They require pandas and NumPy. Table generation does not require model training or dataset downloads. [`input_sha256.json`](../results/input_sha256.json) identifies the input files.

| Analysis | Input files under `results/inputs/` | Output |
| --- | --- | --- |
| First-step versus every-step update matching | `p1_complete_per_seed.csv`, `p1_complete_pairing.csv` | 150 paired Adam-M0/Adam-Mt runs across ten datasets, five shot counts, and three seeds |
| Matching with SGD and Adam references | `p1_closure_by_shot.csv`, `p1_closure_overall.csv` | Separate 27-cell SGD/Adam/Adam-M0/Adam-Mt comparison and signed gap recovery |
| Learning rate and warmup | `p1_lr_warmup_vs_sgd.csv`, `p1_lr_warmup_vs_sgd_bootstrap.csv` | Final accuracy, paired difference to SGD, and paired intervals |
| CLIP shot scaling | `p2_accuracy_per_seed.csv` | Dataset–shot means and sample standard deviations across seeds 1–3 |
| SigLIP adaptation | `p5_accuracy_per_seed_verified.csv` | Dataset–shot means and sample standard deviations across seeds 1–3 |
| Preconditioning and update budgets | Per-seed scope and first-step budget exports | Global-scope summaries and initial applied-update ratios |
| Long-horizon trajectories | `group2_trajectory_overall.csv`, `group2_exact_trajectory_metric_matrix.csv`, `group2_trajectory_per_run.csv` | Epoch-30, observed epoch-300, peak, and last-observed summaries with coverage counts |

## Populations and aggregation

The ten-dataset update-matching comparison contains 300 completed runs, forming 150 Adam-M0/Adam-Mt pairs. It yields mean accuracies of 78.92% for Adam-M0 and 77.69% for global Adam-Mt. The 27-cell comparison with SGD and raw Adam is a separate matched population. [`p1_full_integrity.json`](../results/p1_full_integrity.json) records the source fingerprint and consistency checks for the ten-dataset comparison.

The CLIP shot-scaling table covers six datasets, five shot levels, and three seeds. The SigLIP accuracy table covers four datasets at 1, 4, and 16 shots with three seeds. SigLIP mechanism trajectories cover EuroSAT and DTD at 1 and 16 shots. Each table retains its stated population.

All accuracies are percentages. Sample standard deviations use `ddof=1`. Signed gap recovery is computed from macro-mean differences as `100 × (Adam-M0 − Adam)/(SGD − Adam)` and is not clipped at 100%. Confidence intervals use whole-dataset clusters while retaining paired shots and seeds. Checkpoints are not treated as independent samples.

An epoch-300 average includes only runs with an observed epoch-300 value, and its sample count is reported. Peak and last-observed results summarize the recorded trajectories. Peak accuracy is descriptive and is not a checkpoint-selection rule.

## Training and diagnostics

| Experiment | Implementation |
| --- | --- |
| CLIP shot scaling with SGD, Adam, and Adam-M0 | [Shot-scaling guide](../Investigation/ShotScaling/README.md) |
| SigLIP optimizers, zero-shot evaluation, and diagnostics | [SigLIP guide](../FT/SigLIP/README.md) |
| First-step update matching and other optimizer diagnostics | [Experiment index](EXPERIMENTS.md) |

The ten-dataset matching tables are reproducible from their per-run exports. A separate training entry point for that complete every-step matching grid is not included. First-step matching is implemented in `Investigation/Adam_investigation/Experiment_6` and in the shot-scaling and SigLIP experiments. Dataset images and the complete raw training archives are external inputs.

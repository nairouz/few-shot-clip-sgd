# Supplementary experiments and result tables

The camera-ready supplementary material adds matched-update interventions, learning-rate and warmup controls, shot scaling, SigLIP transfer, and long-horizon comparisons. This directory provides the selected numerical exports and calculation scripts, together with the exact P2 and P5 training packages bundled in their result archives.

## Table reproduction

From the repository root:

```bash
python Rebuttal/scripts/recompute_tables.py
python Rebuttal/scripts/write_manuscript_rows.py
python Rebuttal/scripts/recompute_p1_complete.py
```

These commands read `results/inputs/` and write `results/generated/`. They use pandas and NumPy. They do not retrain models, select new checkpoints, or download datasets. `input_sha256.json` identifies the exact included input tables.

| Tables | Source exports | Calculation |
| --- | --- | --- |
| Complete P1 matched updates | `p1_complete_per_seed.csv`, `p1_complete_pairing.csv` | 150 paired Adam-M0/Adam-Mt runs; ten datasets, five shot counts, three seeds |
| Restricted P1 reference comparison | `p1_closure_by_shot.csv`, `p1_closure_overall.csv` | Separate 27-cell SGD/Adam/Adam-M0/Adam-Mt comparison and signed gap recovery |
| P1 learning rate and warmup | `p1_lr_warmup_vs_sgd.csv`, `p1_lr_warmup_vs_sgd_bootstrap.csv` | Final accuracy, paired difference to SGD, and supplied paired intervals |
| P2 shot scaling | `p2_accuracy_per_seed.csv` | Dataset–shot means and sample SD across seeds 1–3 |
| P5 SigLIP | `p5_accuracy_per_seed_verified.csv` | Dataset–shot means and sample SD across seeds 1–3 |
| P2/P5 mechanisms | Per-seed scope and first-step budget exports | Separate global-scope summaries and initial applied-update ratios |
| Long horizon | `group2_trajectory_overall.csv`, `group2_exact_trajectory_metric_matrix.csv`, `group2_trajectory_per_run.csv` | Epoch-30, observed epoch-300, peak, and last-observed summaries with coverage counts |

The complete P1 export comes from the current 300-run archive. It reproduces mean accuracies of 78.92% for Adam-M0 and 77.69% for global Adam-Mt. The separate 27-cell reference comparison retains its restricted scope and does not replace the complete paired table. `results/p1_full_integrity.json` records the source archive fingerprint and its independent checks.

The P2 table covers six datasets, five shot levels, and three seeds. The P5 accuracy table covers four datasets at 1, 4, and 16 shots with three seeds. P5 mechanism trajectories cover the narrower EuroSAT/DTD subset at 1 and 16 shots. The scripts preserve these distinct populations.

All accuracies are percentages. Sample standard deviations use `ddof=1`. Signed gap recovery is the ratio of macro-mean differences, `100 × (Adam-M0 − Adam)/(SGD − Adam)`, and is not clipped at 100%. Confidence intervals in the supplied exports use whole-dataset clusters with paired shots and seeds. The calculation scripts retain these intervals rather than treating checkpoints as independent samples.

For the long-horizon table, an epoch-300 average contains only runs with an observed epoch-300 value. The corresponding count is reported. Peak and last-observed results use the supplied trajectory summaries. Peak accuracy is descriptive and is not used as a checkpoint-selection rule.

The numerical exports are sufficient to reproduce these aggregation tables. The full raw training archives and datasets are not included in this repository.

## Included training packages

- [P2 shot scaling](P2_ShotScaling/README.md): CLIP SGD, raw Adam, Adam-M0, and diagnostics.
- [P5 SigLIP](P5_SigLIP/README.md): pinned SigLIP checkpoint manifest, training variants, and diagnostics.

P1 training source was not present in its supplied result archive. Its numerical exports are included above. The separate first-step matching diagnostic is available in `Investigation/Adam_investigation/Experiment_6`. The result-table scripts do not imply a complete P1 training package.

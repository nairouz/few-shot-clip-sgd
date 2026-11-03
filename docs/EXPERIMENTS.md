# Diagnostic experiments

The diagnostic source is grouped under `Investigation/`. Each experiment directory contains its own CLIP and dataset modules. The historical folder names are retained because they identify the submitted experiment packages. The scientific purpose is determined by the measurement script, rather than by a folder letter alone.

## Experiment index

The folder names and historical script experiment letters are not always the same. The following table preserves the submitted folder names and reports actual measurement entry points after removal of any one enclosing wrapper directory.

### Adam diagnostics

All measurement files below are inside `second_moment_experiment_files/`.

| Submitted folder | Measurement entry point | Scientific purpose | Distinguishing options |
| --- | --- | --- | --- |
| `Adam_investigation/Experiment_1` | `measure_second_moment.py` | Few-shot versus reference second-moment calibration along a training trajectory | `--ref_mode full_train`, `--v_estimator`, `--traj_optimizer` |
| `Adam_investigation/Experiment_2` | `measure_second_moment.py` | Initialization-only audit before an optimizer update | `--init_only_audit` |
| `Adam_investigation/Experiment_3` | `measure_second_moment.py` | Time-resolved numerator and denominator comparison | paired SGD and Adam trajectories; companion `plot_experiment2_time_series.py` |
| `Adam_investigation/Experiment_4` | `measure_experiment3_ema_staleness.py` | Instantaneous, diagnostic-batch, and running-EMA second moments against the full-data reference | `--adam_beta1`, `--adam_beta2`, `--few_diag_batch_size`, `--ref_diag_batch_size` |
| `Adam_investigation/Experiment_5` | `measure_experiment4_counterfactuals.py` | Few-shot, oracle, frozen, and lagged denominator counterfactuals | `--lag_steps`, `--probe_alpha`, `--probe_match_norm_to` |
| `Adam_investigation/Experiment_6` | `measure_second_moment.py` | Adam with first-step update-budget matching to SGD | `--match_first_step_to_optimizer SGD`, `--match_first_step_ref_lr` |
| `Adam_investigation/Experiment_7` | `measure_feature_drift.py` | Parameter, image-feature, text-feature, and related drift along SGD/Adam trajectories | `--traj_optimizer`, `--diag_every` |
| `Adam_investigation/Experiment_8` | `measure_experiment7_temporal_staleness.py` | Temporal autocorrelation and future full-data alignment of few-shot gradients | `--lags`, `--history_dtype` |

The archived wrappers have been removed so each experiment is directly browsable.

The Adam-diagnostic entry points support `--root_path`, `--dataset`, `--shots`, `--backbone`, `--seed`, `--lr`, `--max_epochs`, `--encoder`, `--position`, `--full_support_train`, `--out_dir`, and `--run_name`. Full-support training is an explicit option. It must not be described as the ordinary minibatch benchmark protocol. Reference-set caps, estimator definitions, and diagnostic-batch sizes must follow the intended experiment configuration.

### SGD geometry diagnostics

All measurement files below are inside `loss_landscape_experiment_files/`.

| Submitted folder | Measurement entry point | Scientific purpose |
| --- | --- | --- |
| `SGD_investigation/Experiment_A` | `measure_experimentA_interpolation.py` | One-dimensional interpolation from pretrained initialization to the final SGD checkpoint |
| `SGD_investigation/Experiment_B` | `measure_experimentB_localstep.py` | Local loss curves along the actual SGD update, with optional random and Hessian-direction probes |
| `SGD_investigation/Experiment_C` | `measure_experimentC_checkpoints.py`, then `measure_experimentC_surface.py` | Checkpoint generation followed by two-dimensional local loss surfaces |
| `SGD_investigation/Experiment_D` | `measure_experimentE_directional_curvature.py` | Directional-curvature decomposition along SGD trajectories |
| `SGD_investigation/Experiment_E` | `measure_experimentF_block_sensitivity.py` | Per-block loss sensitivity and safe-update radii |
| `SGD_investigation/Experiment_F` | `measure_experimentG_connectivity.py` | Linear connectivity between final solutions obtained with different seeds |
| `SGD_investigation/Experiemnt_G` | `measure_experimentH_support_full_mismatch.py` | Support/full-data gradient alignment, Hessian eigendirections, and local surfaces |
| `SGD_investigation/Experiemnt_H` | `measure_experimentI_feature_drift.py` | Parameter, feature, and class-margin drift during SGD adaptation |
| `SGD_investigation/Experiemnt_I1` | `measure_experimentJ_noise_scale.py` | Epoch-matched comparison of full-support, minibatch, noisy, and momentum SGD |
| `SGD_investigation/Experiment_I2` | `measure_experimentJ_noise_scale.py` | Adds update-matched stochasticity comparisons |
| `SGD_investigation/Experiment_I3` | `measure_experimentJ_noise_scale.py` | Adds example-matched stochasticity comparisons |
| `SGD_investigation/Experiment_I4` | `measure_experimentJ_noise_scale.py` | Adds joint update-and-example matching through resampling and gradient accumulation |

The submitted misspelling `Experiemnt` is preserved in the table because it is part of the path. Common archive wrappers have been removed.

The SGD geometry scripts generally use ViT-B/16, one shot, both encoders, and checkpoint epochs `0,1,2,5,10,20,30` by default. Most require `--full_support_train` for the documented full-support protocol. Their input flags use `--root_path`, `--dataset`, `--shots`, `--backbone`, `--lr`, `--max_epochs`, and `--out_dir`. Connectivity uses `--seeds` instead of `--seed`. Surface generation takes `--checkpoints_logdir`, `--checkpoint_epoch`, and `--probe`; the probe choices are `hessian`, `random`, `fullgrad`, and `adam`.

`Experiment_I4` accepts four `--budget_protocol` values: `epoch_matched`, `update_matched`, `example_matched`, and `update_example_matched`. Its default remains `epoch_matched`. The I4 folder alone does not imply the joint-matching configuration was selected.


## Example: first-step matching

The following command exposes the argument interface for the supplied first-step matching diagnostic after dependencies have been installed:

```bash
cd Investigation/Adam_investigation/Experiment_6
python second_moment_experiment_files/measure_second_moment.py --help
```

Its relevant options include `--match_first_step_to_optimizer SGD` and `--match_first_step_ref_lr`. The remaining settings, including the reference-set definition and diagnostic-batch size, follow the corresponding experiment in the supplementary material. Existing README files in each experiment directory describe further measurement and plotting options.

## Output and interpretation

Measurement scripts produce CSV, JSON, and, for some geometry experiments, NPZ files. Their companion `plot_*.py` scripts generate figures. Dataset paths, output paths, reference budgets, and stochasticity controls are explicit command-line options.

The original experiment source is retained. Bytecode and dependency stubs have been omitted. Dataset defaults and username-dependent model-cache routing have been made portable. No diagnostic result was recomputed during preparation of this release.

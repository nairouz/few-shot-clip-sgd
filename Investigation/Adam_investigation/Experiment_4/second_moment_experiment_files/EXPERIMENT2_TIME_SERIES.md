# Experiment 2 — Time-series numerator vs denominator collapse

Experiment 2 asks a more specific question than the initial Adam investigation:

> During few-shot adaptation, does the few-shot gradient itself become too small,
> or does the few-shot second moment collapse faster than the gradient?

The key point is that **the measurement code is already in the existing
`measure_second_moment.py` script**. No new measurement pipeline is required.

The existing logs already contain, at every diagnostic step:

- `few_grad_l2`, `ref_grad_l2`
- `few_v_l2`, `ref_v_l2`
- `grad_cosine`, `grad_sign_agreement`
- `adam_cos_fewfew_vs_fewref`, `adam_cos_fewref_vs_refref`
- `adam_l2_fewfew`, `adam_l2_fewref`, `adam_l2_refref`
- `few_loss`, `ref_loss`

So Experiment 2 only adds a new **trajectory-comparison plotting script**:

- `plot_experiment2_time_series.py`

It takes one SGD run and one Adam run for the *same dataset/task setup* and
plots the most important time-series needed to answer the experiment:

1. gradient alignment over time,
2. gradient norm ratio vs second-moment norm ratio,
3. preconditioned direction alignment,
4. preconditioned direction norm ratios,
5. support/reference loss trajectories.

## Important note on run names

When running SGD and Adam, **use different run names**. Otherwise the Adam run
will overwrite the SGD run.

Good naming convention:

- `..._trajopt_sgd_...`
- `..._trajopt_adam_...`

## Example plotting command

```bash
python second_moment_experiment_files/plot_experiment2_time_series.py \
  --logdir ./results_without_aug/second_moment_logs \
  --run_name_sgd second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_sgd_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --run_name_adam second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ./results_without_aug/experiment2_plots/eurosat_compare
```

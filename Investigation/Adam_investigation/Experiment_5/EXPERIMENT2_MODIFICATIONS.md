# Experiment 2 modifications

## Summary

Experiment 2 does **not** require a new measurement pipeline.
The existing `second_moment_experiment_files/measure_second_moment.py` already
logs all the quantities needed for the time-series numerator-vs-denominator
collapse study:

- `few_grad_l2`, `ref_grad_l2`
- `few_v_l2`, `ref_v_l2`
- `grad_cosine`, `grad_sign_agreement`
- `adam_cos_fewfew_vs_fewref`, `adam_cos_fewref_vs_refref`
- `adam_l2_fewfew`, `adam_l2_fewref`, `adam_l2_refref`
- `few_loss`, `ref_loss`

So the only code added for Experiment 2 is a **new plotting script**:

- `second_moment_experiment_files/plot_experiment2_time_series.py`

plus:

- `second_moment_experiment_files/EXPERIMENT2_TIME_SERIES.md`
- `second_moment_experiment_files/run_experiment2_examples.sh`

## Why no measurement changes were made

Experiment 2 asks:

> During few-shot adaptation, does the few-shot gradient collapse, or does the
> few-shot second moment collapse faster than the gradient?

The current CSV logs already contain exactly the required time-series.
Adding a second measurement script would only duplicate code and increase the
chance of inconsistencies.

## New figures produced

The new plotting script produces trajectory-comparison figures for:

1. gradient alignment over time,
2. gradient norm ratio vs second-moment norm ratio,
3. preconditioned direction alignment,
4. preconditioned direction norm ratios,
5. support/reference loss trajectories.

## Important run-name fix

When comparing SGD and Adam trajectories, use **different run names**.
If both trajectories use the same `--run_name`, the second run overwrites the
first one and the comparison becomes impossible.

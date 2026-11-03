# Experiment 3 — EMA staleness audit: modifications

This patch adds Experiment 3 **without changing the Experiment 2 measurement code**.

## New files

- `second_moment_experiment_files/measure_experiment3_ema_staleness.py`
- `second_moment_experiment_files/plot_experiment3_ema_staleness.py`
- `second_moment_experiment_files/EXPERIMENT3_EMA_STALENESS.md`
- `second_moment_experiment_files/run_experiment3_ema_staleness_examples.sh`

## Why a new measurement script was added

Experiment 3 does not simply compare the few-shot and reference second moments.
It compares **three** few-shot denominator candidates against the same reference:

1. `instant`  = the instantaneous full-support squared gradient on the few-shot set
2. `batch`    = the few-shot batch second moment `E_b[g_b^2]`
3. `ema`      = the actual bias-corrected Adam running second moment

That requires logging rows indexed by `moment_source`, which is conceptually
cleaner as a new script than as a large modification of `measure_second_moment.py`.

## Why separate diagnostic batch sizes were added

If the few-shot diagnostic loader uses the entire support set in one batch, then
`instant` and `batch` become identical in one-shot adaptation.

So Experiment 3 adds:

- `--few_diag_batch_size`
- `--ref_diag_batch_size`

and you should choose `few_diag_batch_size < few_size` for this audit.

## Measurement output format

The new measurement script writes long-format CSV rows with:

- `moment_source = instant`
- `moment_source = batch`
- `moment_source = ema`

and logs:

- gradient alignment (`grad_cosine`, `grad_sign_agreement`)
- source/reference second-moment mismatch
- `cos(u_source, u_oracle)`
- `cos(u_source, u_rr)`
- norm ratios `||u_source|| / ||u_rr||`

## Plotting

The new plotting script overlays the three sources in paper-style 1x2
vision/text plots.

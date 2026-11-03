# Experiment 3 — EMA staleness audit

This experiment answers the question:

> Is the problem specifically Adam's *running* second moment, or would even the
> instantaneous few-shot second moment already be bad?

At each logged checkpoint \(\Theta_t\) along an **Adam trajectory**, the code
compares three few-shot denominator candidates:

- `instant`: \(v_{inst} = m_{few}^2\)
- `batch`:   \(v_{batch} = E_b[g_b^2]\) on few-shot diagnostic mini-batches
- `ema`:     the *actual* bias-corrected Adam state \(\hat v_t\)

against the same full-data reference second moment \(v_{ref}\).

## Why a separate few-shot diagnostic batch size is necessary

If the few-shot diagnostic loader uses the *entire* support set in a single
batch, then `instant` and `batch` become identical in one-shot problems.

So this experiment adds:

- `--few_diag_batch_size`
- `--ref_diag_batch_size`

and uses a **smaller few-shot diagnostic batch size** to make the batch-based
few-shot second moment meaningfully different from the instantaneous squared
support gradient.

## Main outputs

The measurement script writes:

- `<run_name>_config.json`
- `<run_name>_per_iteration.csv`
- `<run_name>_per_block.csv`

with long-format rows identified by:

- `moment_source = instant`
- `moment_source = batch`
- `moment_source = ema`

The plotting script overlays those three sources in paper-style 1x2
vision/text figures.

## Beta2 sweep

Run the measurement script multiple times with:

- `--adam_beta2 0.9`
- `--adam_beta2 0.99`
- `--adam_beta2 0.999`

This is the intended staleness sweep.

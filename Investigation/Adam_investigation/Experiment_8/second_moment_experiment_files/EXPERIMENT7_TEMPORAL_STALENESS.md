# Experiment 7 — Temporal staleness of the few-shot gradient

Experiment 7 asks:

> Does the few-shot gradient itself become stale over time?

The exact protocol is:

For temporal lags \(\Delta \in \{1,2,5\}\), compare at checkpoints
\(\Theta_t\) and \(\Theta_{t+\Delta}\):

- **few/few temporal autocorrelation**

  \[
  \cos(m_t^{few}, m_{t+\Delta}^{few})
  \]

- **few/full future-task alignment**

  \[
  \cos(m_t^{few}, m_{t+\Delta}^{full})
  \]

where

- \(m_t^{few} = \nabla L_{few}(\Theta_t)\)
- \(m_t^{full} = \nabla L_{full}(\Theta_t)\)

and `full` refers to the reference data source selected by `--ref_mode`
(typically `full_train`).

## Interpretation

- If `cos(m_t^few, m_{t+Δ}^few)` drops quickly, the few-shot gradient itself is
  changing rapidly in time.
- If `cos(m_t^few, m_{t+Δ}^full)` drops even faster, the few-shot gradient at
  time `t` is becoming stale with respect to the *future full-data objective*.
- If Adam drops faster than SGD, Adam is amplifying gradient staleness.

## Files

Measurement script:

- `measure_experiment7_temporal_staleness.py`

Plotting script:

- `plot_experiment7_temporal_staleness.py`

## Output bundle

For each run, the measurement script writes:

- `<run_name>_config.json`
- `<run_name>_temporal_staleness.csv`

The CSV contains both encoder aggregates (`scope=encoder`) and per-block rows
(`scope=block`).

## Important note on history storage

The rolling history buffer stores only the past few-shot gradients needed for
future lag comparisons.

- `--history_dtype float32` gives exact history storage and is the recommended
  setting when host RAM is sufficient.
- `--history_dtype float16` reduces RAM substantially and is often sufficient
  in practice because tensors are cast back to `float32` before the cosine
  computations.

## Important note on lags

With `--full_support_train` and `--diag_every 1`, one optimizer step equals one
full-support epoch. In that common setting, lags `1,2,5` mean **1, 2, and 5
training epochs apart**.

## Important note on run names

Use different run names for SGD and Adam, exactly as in Experiment 2, e.g.

- `..._trajopt_sgd_...`
- `..._trajopt_adam_...`

Otherwise one run will overwrite the other.

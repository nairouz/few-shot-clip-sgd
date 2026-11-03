# Experiment 4 — Frozen-denominator and oracle-denominator counterfactuals

This experiment answers the causal question:

> If you only fix the denominator, how much of Adam can be rescued?

It reuses the same dataset/model setup as Experiments 2 and 3, but adds four
counterfactual adaptive directions at each checkpoint \Theta_t:

- `u_ff`   = `m_few / sqrt(v_few + eps)`
- `u_fr`   = `m_few / sqrt(v_ref + eps)`
- `u_rr`   = `m_ref / sqrt(v_ref + eps)`
- `u_f0`   = `m_few / sqrt(v_frozen_init + eps)`
- `u_flag` = `m_few / sqrt(v_few(t-lag) + eps)`

The script logs:

1. per-encoder and per-block counterfactual direction diagnostics,
2. one-step norm-matched probe losses on the few-shot, reference, and test splits.

## Important implementation choice

All probe directions are norm-matched to `u_ff` by default. This means the
counterfactuals answer the intended question directly under the same adaptive
update budget as the live few-shot Adam denominator.

## Outputs

- `<run_name>_counterfactual_per_iteration.csv`
- `<run_name>_counterfactual_per_block.csv`
- `<run_name>_probe_per_iteration.csv`

The companion plotting script is `plot_experiment4_counterfactuals.py`.

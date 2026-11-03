# Experiment 1 — Initialization Audit

This experiment measures the few-shot gradient and second moment **at the pretrained CLIP checkpoint** `Theta_0`, before any optimizer update is applied.

## Purpose

The goal is to determine whether Adam's denominator pathology is already present at initialization, or whether it only appears later during adaptation.

At `Theta_0`, the code compares:

- the few-shot gradient `m_few`
- the full/reference gradient `m_ref`
- the few-shot second moment `v_few`
- the full/reference second moment `v_ref`

and forms the three Adam-like directions:

- `u_ff = m_few / sqrt(v_few + eps)`
- `u_fr = m_few / sqrt(v_ref + eps)`
- `u_rr = m_ref / sqrt(v_ref + eps)`

## How it is enabled

Use the same `measure_second_moment.py` script as Experiment 3, but add:

```bash
--init_only_audit
```

In this mode, the script:

1. builds the same few-shot and reference datasets,
2. loads CLIP and marks the same trainable parameters,
3. logs the diagnostic **once** at `global_step=0`, `epoch=0`, `step_in_epoch=0`,
4. exits **before** any optimizer update.

## Important note about trajectory

Because no training step is taken, the initialization audit is **trajectory-independent**.

So in practice you only need **one run** per dataset/shot/backbone configuration. Running it once with `--traj_optimizer SGD` and once with `--traj_optimizer Adam` gives the same tensors at `Theta_0`; only the metadata / run name differ.

# Experiment 5 — Update-budget matched Adam vs SGD

Experiment 5 asks a narrower question than Experiment 2:

> Is Adam bad mainly because of **wrong scale**, or because of **wrong geometry**?

The protocol is:

1. run an **SGD** trajectory at the baseline learning rate,
2. run a **raw Adam** trajectory at the same nominal learning rate,
3. run an **Adam-matched** trajectory where the Adam learning rate is rescaled **once at step 0** so that
   the first-step global relative update norm matches the SGD baseline.

The matching statistic is

\[
  r_t = \frac{\|\Delta \Theta_t\|_2}{\|\Theta_t\|_2}
\]

computed over the **trainable parameters only**.

## Minimal code changes relative to Experiment 2

No new measurement pipeline is introduced. The existing
`measure_second_moment.py` script is extended with:

- first-step learning-rate matching flags,
- exact logging of the **planned update** before `optimizer.step()`,
- new CSV fields for global / encoder / block relative update norms.

The second-moment diagnostics remain unchanged.

## New CLI flags

### `--match_first_step_to_optimizer {none,SGD,Adam}`
If not `none`, rescale the trajectory learning rate **once at step 0** so that
its first-step global relative update norm matches the chosen reference
optimizer on the same first training batch.

### `--match_first_step_ref_lr FLOAT`
Learning rate of the reference optimizer used for matching.
Defaults to `--lr`.

## New CSV fields

These are added to both `*_per_iteration.csv` and `*_per_block.csv`:

- `optimizer_lr`
- `requested_lr`
- `lr_match_scale`
- `init_target_rel_update_norm`
- `init_pre_match_rel_update_norm`
- `rel_update_norm_global`
- `update_l2_global`
- `param_l2_global`
- `rel_update_norm_scope`
- `update_l2_scope`
- `param_l2_scope`

Interpretation:

- `rel_update_norm_global` is the planned update budget for the whole trainable set,
- `rel_update_norm_scope` is the same quantity at the encoder/block scope,
- `lr_match_scale` is the one-shot multiplier applied at step 0,
- `init_pre_match_rel_update_norm` and `init_target_rel_update_norm` tell you
  what was matched.

## Plotting script

Use:

```bash
python second_moment_experiment_files/plot_experiment5_update_budget.py \
  --logdir ./results_without_aug/second_moment_logs \
  --run_name_sgd <sgd_run_name> \
  --run_name_adam <adam_run_name> \
  --run_name_adam_matched <adam_matched_run_name> \
  --outdir ./results_without_aug/experiment5_plots/<dataset_name>
```

Generated figures:

1. `exp5_global_update_budget_compare`
2. `exp5_encoder_update_budget_compare`
3. `exp5_block_step0_update_budget_compare`
4. `exp5_geometry_compare`
5. `exp5_loss_compare`

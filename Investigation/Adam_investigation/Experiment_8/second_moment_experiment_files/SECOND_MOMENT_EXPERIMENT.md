# Experiment 3: few-shot vs full-data second moment

This folder contains a **paper-style implementation** of the experiment

> *few-shot vs full-data second moment in CLIP adaptation*

for the second paper.

It is deliberately aligned with the ECCV few-shot gradient-noise study:

- the training trajectory can use **one full-support batch per epoch**,
- the plots are emitted in the **same 1x2 vision/text presentation style**,
- the notation follows the paper (
  \(K\) = shots per class,
  \(\rho = k/d\) = density,
  vision/text shown separately).

## What the code measures

At a checkpoint \(\Theta_t\), the code compares two data sources:

- **few-shot** support set: the actual \(K\)-shot support set used for training,
- **reference** support set:
  - either a larger \(K_{\text{ref}}\)-shot set from the same task,
  - or the **full training split** of the target dataset.

For each data source \(D\), it computes the full-support mean gradient

\[
m_t(D) = \nabla L_D(\Theta_t).
\]

Then it builds a diagonal second-moment estimator \(v_t(D)\) using one of two modes.

### 1) `batch_mean_square`  **(recommended main experiment)**

\[
v_t(D) = \mathbb{E}_b[g_b \odot g_b],
\]

where \(g_b\) is the gradient of the **batch-mean** loss on a diagnostic batch \(b\).
This is the more Adam-faithful option.

### 2) `full_support_square`  **(exact Fig. 1-style analogue)**

\[
v_t(D) = m_t(D) \odot m_t(D).
\]

This matches the full-support spirit of the ECCV gradient-noise study exactly.
It is useful as a clean appendix/control analysis.

## Adam-like directions that are compared

The script forms three adaptive directions:

\[
u_{ff} = \frac{m_{few}}{\sqrt{v_{few}+\varepsilon}}, \qquad
u_{fr} = \frac{m_{few}}{\sqrt{v_{ref}+\varepsilon}}, \qquad
u_{rr} = \frac{m_{ref}}{\sqrt{v_{ref}+\varepsilon}}.
\]

These directions answer two questions:

1. **How wrong is the few-shot denominator?**
   This is captured by the mismatch between \(v_{few}\) and \(v_{ref}\).

2. **How much does replacing only the denominator move the adaptive direction?**
   This is captured by
   - \(\cos(u_{ff}, u_{fr})\), and
   - \(\cos(u_{fr}, u_{rr})\).

If replacing only the denominator changes the direction a lot and moves it much
closer to the reference one, the failure is specifically in the few-shot
second-moment estimate.

## Files

### `measure_second_moment.py`
Runs the diagnostic during CLIP few-shot adaptation.

Outputs:

- `<run_name>_config.json`
- `<run_name>_per_iteration.csv`
- `<run_name>_per_block.csv`
- `<run_name>_hists_and_curves.npz`

### `plot_second_moment.py`
Consumes the log bundle and produces paper-ready figures in the same visual
style as the ECCV sign-correctness plots.

Generated figures:

- `hist_log10_v_ratio.(png|pdf)`
- `v_ratio_per_iteration.(png|pdf)`
- `v_ratio_per_block.(png|pdf)`
- `precond_cos_per_iteration.(png|pdf)`
- `precond_cos_per_block.(png|pdf)`
- `topk_v_overlap_curve.(png|pdf)`

### `clip_block_utils.py`
Maps CLIP parameters to `(encoder, transformer block)` for separate vision/text
and per-block analysis.

### `run_second_moment_examples.sh`
Ready-to-run examples for EuroSAT, Oxford-Pets, and DTD.

## Recommended protocol for the paper

### Main version

Use:

- `--full_support_train`
- `--ref_mode full_train`
- `--v_estimator batch_mean_square`

This gives:

- the **same one-batch-per-epoch trajectory style** as your ECCV gradient study,
- but a **more Adam-faithful** second-moment estimator.

### Appendix / control version

Use:

- `--full_support_train`
- `--ref_mode full_train`
- `--v_estimator full_support_square`

This gives the **exact full-support analogue** of the ECCV Fig. 1 construction.

## Example commands

### EuroSAT

```bash
python second_moment_experiment_files/measure_second_moment.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:1 \
  --out_dir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_second_moment.py \
  --logdir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ./results_without_aug/second_moment_plots/second_moment_dataset_eurosat_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square
```

### Oxford-Pets

```bash
python second_moment_experiment_files/measure_second_moment.py \
  --dataset oxford_pets \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 37 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:1 \
  --out_dir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_second_moment.py \
  --logdir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ./results_without_aug/second_moment_plots/second_moment_dataset_oxford_pets_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square
```

### DTD

```bash
python second_moment_experiment_files/measure_second_moment.py \
  --dataset dtd \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --traj_optimizer Adam \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 47 \
  --full_support_train \
  --ref_mode full_train \
  --v_estimator batch_mean_square \
  --diag_every 1 \
  --diag_batch_size 256 \
  --device cuda:0 \
  --out_dir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square

python second_moment_experiment_files/plot_second_moment.py \
  --logdir ./results_without_aug/second_moment_logs \
  --run_name second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square \
  --outdir ./results_without_aug/second_moment_plots/second_moment_dataset_dtd_noisyshots_1_ref_full_backbone_ViT-B-16_trajopt_adam_seed_1_pos_all_enc_both_vest_batch-mean-square
```

## Optional control trajectory

To check whether the phenomenon is tied to Adam’s *path* or to the
few-shot/full-data second moment mismatch itself, rerun the exact same commands
with

```bash
--traj_optimizer SGD
```

and keep all diagnostic options identical.

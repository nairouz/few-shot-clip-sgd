# Experiment E — Directional curvature decomposition

This experiment measures the directional curvature

\[
\kappa(d) = d^\top H d
\]

at checkpoints along an SGD few-shot CLIP trajectory.

## Goal
Understand which directions are actually dangerous during few-shot adaptation.

## Training protocol
- CLIP ViT-B/16 by default
- 1-shot adaptation by default
- full-support training when `--full_support_train` is set
- no augmentation (geometry experiment)
- train both encoders
- all selected blocks trainable
- checkpoints at `t in {0,1,2,5,10,20,30}` by default

## Directions analysed
- `sgd_step`     : actual SGD step direction
- `few_grad`     : normalized support-gradient direction
- `full_grad`    : normalized full-train gradient direction
- `top_eig_full` : approximate top Hessian eigen-direction of the full-train loss
- `random_i`     : random normalized probe directions

Under the exact default protocol here (vanilla SGD, no momentum, one batch per
epoch), `sgd_step` and `few_grad` are identical by construction. They are both
logged so the experimental design stays explicit.

## Losses whose Hessian is probed
- `few`  : support loss
- `full` : full-train loss
- `test` : test loss

## Outputs
For a run name `<run_name>`, the measurement script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_directional_curvature.csv`
- `<run_name>_block_curvature.csv`
- `<run_name>_summary.json`
- checkpoints in `checkpoints/<run_name>/epoch_XX_trainable.pt`

The plotting script emits:

- `expE_directional_curvature_over_time.(png|pdf)`
- `expE_block_curvature_full_t00.(png|pdf)`
- `expE_block_curvature_full_tXX.(png|pdf)` for the final requested checkpoint

## Typical usage
```bash
python loss_landscape_experiment_files/measure_experimentE_directional_curvature.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --optimizer SGD \
  --lr 1e-4 \
  --max_epochs 30 \
  --batch_size 10 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --eval_batch_size 128 \
  --analysis_microbatch_size 4 \
  --empty_cache_between_phases \
  --power_iters 6 \
  --num_random_dirs 3 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentE_logs

python loss_landscape_experiment_files/plot_experimentE_directional_curvature.py \
  --logdir ./results_without_aug/experimentE_logs \
  --run_name <run_name> \
  --outdir ./results_without_aug/experimentE_plots/<run_name> \
  --block_epochs 0,30
```


## A6000 memory-safe mode
For local NVIDIA RTX A6000 runs, the main memory-safety knob is `--analysis_microbatch_size`.
The default in this version is `4`, which keeps the HVP / curvature path small enough for local 48–49 GB cards.
Recommended local flags:

```bash
--eval_batch_size 128 \
--analysis_microbatch_size 4 \
--empty_cache_between_phases
```

This does **not** change the experiment mathematically; it only splits each evaluation batch into smaller GPU micro-batches inside the gradient/HVP routines.


## Local A6000 note

Experiment E computes Hessian-vector products (double backward) through CLIP attention. On recent PyTorch builds, `nn.MultiheadAttention` may route to the efficient scaled-dot-product attention kernel, which does **not** support second derivatives. The measurement code therefore forces the **math SDPA kernel** only inside the HVP path. This makes curvature computation compatible with local A6000 runs. Because the math kernel uses more memory than the efficient kernel, the safer default is `--analysis_microbatch_size 2` (and `1` if you still hit OOM).

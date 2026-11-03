# Experiment C A6000 memory-safety patch

This patch fixes CUDA out-of-memory failures observed during the `fullgrad` probe, e.g. Oxford-Pets checkpoint 10 on an NVIDIA RTX A6000.

## Root cause

The previous script used `--eval_batch_size 256` for both:

1. loss-only surface evaluation, and
2. gradient-bearing full-train gradient computations.

The second case is much heavier. Computing the full-gradient probe with all CLIP ViT-B/16 blocks trainable must store backward activations. Batch size 256 can exceed 48 GB VRAM.

## Main changes

### 1. Separate gradient batch size

`measure_experimentC_surface.py` now has:

```bash
--eval_batch_size 128
--grad_eval_batch_size 32
```

`eval_batch_size` is used for no-grad loss evaluation.
`grad_eval_batch_size` is used only for full-gradient computations, especially the `fullgrad` probe.

If an OOM still occurs, rerun with:

```bash
GRAD_EVAL_BATCH_SIZE=16
```

### 2. Dedicated full-gradient dataloader

`prepare_common_experiment()` now returns:

```python
full_grad_loader
```

so `fullgrad` uses a small batch, while surface loss evaluation can keep a larger batch.

### 3. CUDA cleanup between heavy operations

The code now clears PyTorch's CUDA cache after gradient-bearing mini-batches and periodically during surface evaluation.

### 4. Resume-safe full-matrix runner

`loss_landscape_experiment_files/run_experimentC_full_matrix.sh` now defaults to:

```bash
EVAL_BATCH_SIZE=128
GRAD_EVAL_BATCH_SIZE=32
SURFACE_FLUSH_CACHE_EVERY=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Completed surfaces are skipped when `SKIP_EXISTING=1`, so rerunning the script resumes from the first missing surface.

## Recommended resume command

```bash
TRAIN_DEVICE=cuda:1 SURFACE_DEVICE=cuda:1 \
GRAD_EVAL_BATCH_SIZE=32 EVAL_BATCH_SIZE=128 \
HVP_MODE=finite_difference \
bash loss_landscape_experiment_files/run_experimentC_full_matrix.sh
```

If Oxford-Pets or DTD still OOMs:

```bash
TRAIN_DEVICE=cuda:1 SURFACE_DEVICE=cuda:1 \
GRAD_EVAL_BATCH_SIZE=16 EVAL_BATCH_SIZE=64 \
HVP_MODE=finite_difference \
bash loss_landscape_experiment_files/run_experimentC_full_matrix.sh
```

## Resume from a narrow subset

To resume only Oxford-Pets from checkpoint 10 onward:

```bash
RUN_CHECKPOINTS=0 RUN_SURFACES=1 RUN_PLOTS=1 \
DATASETS_STR="oxford_pets dtd" \
CHECKPOINTS_STR="10 20 30" \
GRAD_EVAL_BATCH_SIZE=32 EVAL_BATCH_SIZE=128 \
bash loss_landscape_experiment_files/run_experimentC_full_matrix.sh
```

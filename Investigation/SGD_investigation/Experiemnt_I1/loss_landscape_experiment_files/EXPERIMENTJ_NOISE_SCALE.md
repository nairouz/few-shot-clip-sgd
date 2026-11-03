# Experiment J — Noise-scale / batch-size landscape study

## Goal
Understand whether SGD's stochasticity helps it find flatter regions during
few-shot CLIP adaptation.

## Variants compared
The measurement script runs five SGD variants on the same few-shot support set:

1. **fullbatch_nomom**: full-support SGD, no momentum
2. **div2_nomom**: half-support mini-batch SGD, no momentum
3. **div4_nomom**: quarter-support mini-batch SGD, no momentum
4. **fullbatch_noise**: full-support SGD with additive Gaussian gradient noise
5. **fullbatch_momentum**: full-support SGD with momentum

The defaults match the recommended Experiment J protocol.

## Metrics at checkpoints t ∈ {0,1,2,5,10,20,30}
- support / full-train / test loss and accuracy
- full-train gradient norm
- support gradient norm
- empirical batch-noise ratio on the support set
- injected-noise ratio and total stochasticity ratio
- finite-difference curvature proxy on the full-train loss along:
  - the normalized full-train gradient direction
  - a random normalized direction
- relative drift from the pretrained initialization

## Connectivity
After training, the script linearly interpolates between the final baseline
solution (`fullbatch_nomom`) and every other variant and evaluates support /
full-train / test losses along the interpolation line.

## Typical usage
```bash
python loss_landscape_experiment_files/measure_experimentJ_noise_scale.py \
  --dataset eurosat \
  --shots 1 \
  --backbone ViT-B/16 \
  --position all \
  --encoder both \
  --lr 1e-4 \
  --max_epochs 30 \
  --full_support_train \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentJ_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both

python loss_landscape_experiment_files/plot_experimentJ_noise_scale.py \
  --logdir ./results_without_aug/experimentJ_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both \
  --outdir ./results_without_aug/experimentJ_plots/experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both
```

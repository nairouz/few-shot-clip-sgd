# Experiment J — Noise-scale / batch-size landscape study

## Goal
Understand whether SGD's stochasticity helps it find flatter regions during
few-shot CLIP adaptation.

## What is now integrated
The code now supports **two budgeting protocols**:

1. **`epoch_matched`**
   - original Experiment J
   - every variant is trained for the same number of epochs
   - mini-batch variants therefore make *more* optimizer updates than the full-batch baseline

2. **`update_matched`**
   - fairness-control follow-up experiment
   - every variant is trained for the same total number of optimizer updates as the full-batch baseline
   - this isolates whether the gain of mini-batch SGD comes from:
     - structured stochasticity, or
     - simply taking more, smaller updates per epoch

The defaults preserve the original behavior (`epoch_matched`).

## Variants compared
The measurement script runs five SGD variants on the same few-shot support set:

1. **fullbatch_nomom**: full-support SGD, no momentum
2. **div2_nomom**: half-support mini-batch SGD, no momentum
3. **div4_nomom**: quarter-support mini-batch SGD, no momentum
4. **fullbatch_noise**: full-support SGD with additive Gaussian gradient noise
5. **fullbatch_momentum**: full-support SGD with momentum

## Metrics at checkpoints
For every saved checkpoint the script writes:
- support / full-train / test loss and accuracy
- full-train gradient norm
- support gradient norm
- empirical batch-noise ratio on the support set
- injected-noise ratio and total stochasticity ratio
- finite-difference curvature proxy on the full-train loss along:
  - the normalized full-train gradient direction
  - a random normalized direction
- relative drift from the pretrained initialization
- **budget accounting**:
  - `global_update`
  - `epochs_completed`
  - `examples_seen`
  - `example_budget_ratio`
  - `update_budget_ratio`

## Connectivity
After training, the script linearly interpolates between the final baseline
solution (`fullbatch_nomom`) and every other variant and evaluates support /
full-train / test losses along the interpolation line.

## Typical usage
### Original Experiment J (epoch-matched)
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
  --budget_protocol epoch_matched \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentJ_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both
```

### Follow-up fairness control (update-matched)
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
  --budget_protocol update_matched \
  --checkpoint_updates 0,1,2,5,10,20,30 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentJ_update_control_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update
```

Plot either protocol with the same plotting script:
```bash
python loss_landscape_experiment_files/plot_experimentJ_noise_scale.py \
  --logdir ./results_without_aug/experimentJ_update_control_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update \
  --outdir ./results_without_aug/experimentJ_update_control_plots/experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_update
```


### J3 complementary control (example-matched)
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
  --budget_protocol example_matched \
  --checkpoint_epochs 0,1,2,5,10,20,30 \
  --device cuda:0 \
  --out_dir ./results_without_aug/experimentJ3_logs \
  --run_name experimentJ_dataset_eurosat_shots_1_backbone_ViT-B-16_opt_sgd_seed_1_pos_all_enc_both_budget_examples
```

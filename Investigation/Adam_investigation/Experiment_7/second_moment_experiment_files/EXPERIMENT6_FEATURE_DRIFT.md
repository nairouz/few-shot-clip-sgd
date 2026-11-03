# Experiment 6 — Feature-drift audit

This experiment answers the question:

> Does Adam fail because it pushes CLIP too far from the pretrained solution too fast?

The implementation follows the exact protocol recommended for the feature-drift
study:

1. **Parameter drift** relative to the pretrained CLIP solution
   \[
   \frac{\|\Theta_t - \Theta_0\|_2}{\|\Theta_0\|_2}
   \]
   logged for the full trainable scope, for each encoder, and per transformer
   block.

2. **Image-feature drift** on three splits:
   - support (`few`)
   - full-train reference (`ref`)
   - test (`test`)

   using the mean cosine to the pretrained image features:
   \[
   \text{drift} = 1 - \frac{1}{N} \sum_i \cos(f_t(x_i), f_0(x_i)).
   \]

3. **Text-feature drift** using the cosine between the current text features and
   the pretrained text features for the class prompts.

4. **Logit drift** and **ground-truth margin drift** on support / full-train /
   test.

5. **Support / full-train / test loss and accuracy** at every diagnostic step.

The script writes:
- `<run_name>_config.json`
- `<run_name>_per_iteration.csv`
- `<run_name>_per_block.csv`

The compare-plot script overlays the SGD and Adam trajectories and produces:
- `exp6_param_drift_compare`
- `exp6_block_drift_compare`
- `exp6_feature_drift_compare`
- `exp6_logit_margin_compare`
- `exp6_accuracy_loss_compare`

# Experiment B — Local 1D landscape along the actual SGD step

Experiment B asks:

> What is the local safe step radius of SGD during few-shot CLIP adaptation?

The script `measure_experimentB_localstep.py` implements the exact training
protocol requested for the geometry experiment:

- CLIP ViT-B/16 (default, but configurable)
- 1-shot adaptation
- SGD trajectory
- full-support training (one batch per epoch with `--full_support_train`)
- no augmentation
- both encoders trainable
- all selected blocks trainable
- checkpoints saved at `t in {0,1,2,5,10,20,30}` by default

At each checkpoint `Theta_t`, the script computes the actual SGD step

`Delta_t = -lr * grad_few(Theta_t)`

and measures the local 1D losses along

`Theta_t + c * Delta_t`

where `c` is a step multiplier and `c=1` is the actual SGD step.

To follow the comparison protocol from the experiment plan, the script also
supports two auxiliary directions with the same norm as `Delta_t`:

- `random`
- `hessian_top` (top Hessian-eigenvector direction of the few-shot support loss)

In this exact setup (vanilla full-support SGD with no momentum), the actual SGD
step and `-grad_few` are the same direction, so a separate negative-gradient
curve would be redundant and is therefore not duplicated.

## Output files

For each run, the script writes:

- `<run_name>_config.json`
- `<run_name>_trajectory.csv`
- `<run_name>_direction_stats.csv`
- `<run_name>_localstep_curves.csv`
- `<run_name>_safe_radius.csv`
- `<run_name>_summary.json`
- `checkpoints/<run_name>/epoch_XX_trainable.pt`

The plotting script `plot_experimentB_localstep.py` produces:

- `expB_localstep_losses.(png|pdf)`
- `expB_safe_radius.(png|pdf)`
- `expB_minimizer_multiplier.(png|pdf)`

The figure style (font sizes, transparent background, axis treatment, global
labels, line styling) is matched to Experiment A.


## Hessian probe robustness

The top-Hessian probe now tries exact second-order autograd with the math-only scaled-dot-product attention backend. If the local PyTorch build still does not support backward-of-backward through attention, the script automatically falls back to a finite-difference Hessian-vector product controlled by `--hessian_backend` and `--hessian_fd_eps`.


## Paper-ready notation

The updated plotting script uses the same notation as the paper subsection:

- legend entries: $u_t^{\mathrm{sgd}}$, $u_t^{\mathrm{rand}}$, $u_t^{\mathrm{top}}$
- safe-radius axis: $\rho_{\ell,u}(t)$
- minimizer axis: $c_{\ell,u}^{\star}(t)$
- loss-line axis: $\phi_{\ell,u_t^{\mathrm{sgd}}}(c;t)$
- panel titles: $L_{\mathrm{few}}$, $L_{\mathrm{full}}$, $L_{\mathrm{test}}$

Use `make_experimentB_paper_plots.sh` to regenerate all three datasets and copy the PDFs into `figures/sgd_success/` with the exact filenames expected by the LaTeX source.

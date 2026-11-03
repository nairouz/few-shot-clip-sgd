# Experiment B modifications

This codebase keeps Experiment A intact and adds a new standalone geometry experiment in:

- `loss_landscape_experiment_files/measure_experimentB_localstep.py`
- `loss_landscape_experiment_files/plot_experimentB_localstep.py`
- `loss_landscape_experiment_files/EXPERIMENTB_LOCALSTEP.md`
- `loss_landscape_experiment_files/run_experimentB_examples.sh`

## What Experiment B does

Experiment B measures the **local 1D loss landscape along the actual SGD step**
at checkpoints `t in {0,1,2,5,10,20,30}`.

At each checkpoint `Theta_t`, it builds the actual full-support SGD step

`Delta_t = -lr * grad_few(Theta_t)`

and evaluates the losses along

`Theta_t + c * Delta_t`

where `c=1` is the actual SGD step.

To follow the recommended protocol, the script also probes two auxiliary
comparison directions with the same norm as `Delta_t`:

- `random`
- `hessian_top`

The plotting style is intentionally matched to Experiment A.


## Hessian probe robustness

The top-Hessian probe now tries exact second-order autograd with the math-only scaled-dot-product attention backend. If the local PyTorch build still does not support backward-of-backward through attention, the script automatically falls back to a finite-difference Hessian-vector product controlled by `--hessian_backend` and `--hessian_fd_eps`.

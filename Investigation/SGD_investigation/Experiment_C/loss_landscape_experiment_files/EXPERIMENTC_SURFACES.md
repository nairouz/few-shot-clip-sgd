# Experiment C — 2D local surfaces around SGD checkpoints

This experiment measures local 2D loss surfaces around a saved SGD checkpoint `Theta_t`.

For a checkpoint `t`, the code constructs:

- `d_t`: the actual local SGD descent direction at `Theta_t` (plain SGD => `- few-shot gradient`)
- `e_t`: an orthogonal probe direction, chosen from
  - `hessian`
  - `random`
  - `fullgrad`
  - `adam`

Then it evaluates

`L(alpha, beta) = L(Theta_t + scale * (alpha d_t + beta e_t))`

for three losses:

- support / few-shot loss
- full-train loss
- test loss

## Scripts

### 1. Save checkpoints

`measure_experimentC_checkpoints.py`

Runs the SGD trajectory with the geometry setup and saves checkpoints at
`t in {0,1,2,5,10,20,30}` by default.

### 2. Measure one local surface

`measure_experimentC_surface.py`

Loads one saved checkpoint and one probe type, then evaluates the 2D surface.

### 3. Plot the surface

`plot_experimentC_surface.py`

Produces:

- `expC_surface_contours.(png|pdf)`
- `expC_surface_cross_sections.(png|pdf)`

The plotting style intentionally matches the Experiment A plotting style:
transparent background, same typography, same global-axis layout, same visual rhythm.

## Notes

- The geometry study is expensive. The intended workflow is:
  1. train once and save checkpoints,
  2. measure surfaces checkpoint-by-checkpoint and probe-by-probe.
- `hessian` is the most expensive probe because it requires repeated Hessian-vector products.
- The default `grid_size=25` is a practical compromise.

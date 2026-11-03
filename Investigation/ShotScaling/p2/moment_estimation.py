# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import math
import torch

EPS_METRIC = 1e-30

def estimate_moments(batch_gradients):
    if not batch_gradients:
        raise ValueError("at least one gradient is required")
    stack = torch.stack([x.detach().float() for x in batch_gradients])
    mean = stack.mean(0)
    return mean, stack.square().mean(0), mean.square()

def denominator_direction(mean, moment, eps: float, definition: str):
    if definition == "paper_proxy":
        return mean / torch.sqrt(moment + eps)
    if definition == "adam_faithful":
        return mean / (torch.sqrt(moment) + eps)
    raise ValueError(definition)

def mismatch(source, reference, eps: float = EPS_METRIC):
    z = torch.log10((source.float() + eps) / (reference.float() + eps)).flatten()
    absolute = z.abs()
    # Percentiles are read from a fixed 65,536-bin coordinate histogram. This
    # is a bounded-memory, single-pass estimator (maximum bin-width error is
    # recorded by the caller/config), unlike torch.quantile/kthvalue which is
    # unavailable or impractical at CLIP global-vector scale.
    maximum = float(absolute.max())
    bins = 65536
    counts = torch.histc(absolute, bins=bins, min=0, max=max(maximum, 1e-30))
    cumulative = counts.cumsum(0)
    def percentile(q):
        rank = max(1, min(absolute.numel(), math.ceil(q * absolute.numel())))
        index = int(torch.searchsorted(cumulative, torch.tensor(rank, device=cumulative.device)))
        return maximum * min(bins, index + 1) / bins
    return {
        "signed_mean_z": float(z.mean()), "mean_absolute_z": float(absolute.mean()),
        "median_absolute_z": percentile(.5),
        "p90_absolute_z": percentile(.9),
        "p99_absolute_z": percentile(.99),
        "fraction_z_lt_0": float((z < 0).float().mean()),
        "fraction_z_lt_minus1": float((z < -1).float().mean()),
        "fraction_z_lt_minus2": float((z < -2).float().mean()),
        "fraction_underestimated": float((source < reference).float().mean()),
    }

def exact_topk_overlap(a, b, density: float):
    if not 0 < density <= 1 or a.numel() != b.numel():
        raise ValueError("invalid density or unequal vectors")
    k = max(1, min(a.numel(), math.ceil(density * a.numel())))
    ai, bi = torch.topk(a.flatten(), k, sorted=False).indices, torch.topk(b.flatten(), k, sorted=False).indices
    # Exact membership with one byte per coordinate. This avoids torch.isin's
    # very large pairwise/sort workspace at encoder-global rho=0.1.
    selected = torch.zeros(a.numel(), dtype=torch.bool, device=a.device)
    selected[ai] = True
    return k, float(selected[bi].sum() / k)

def collapse_ratio(rm_start, rm_end, rv_start, rv_end):
    return (rv_start / rv_end) / (rm_start / rm_end)

def extract_adam_ema(optimizer, parameters):
    result = {}
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if "exp_avg_sq" not in state:
            result[parameter] = None
            continue
        step = int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
        beta2 = optimizer.param_groups[0]["betas"][1]
        result[parameter] = state["exp_avg_sq"].detach().float() / (1 - beta2 ** step)
    return result

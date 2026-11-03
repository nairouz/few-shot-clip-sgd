# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""P5 evaluation, update-budget, drift, and moment diagnostics."""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from instrumentation import (
    ScopeMap,
    StepMeasurement,
    compute_parameter_drift,
    compute_scope_budgets,
)


ADAM_EPS = 1e-8
EPS_METRIC = 1e-30
HISTOGRAM_EDGES = np.linspace(-12.0, 12.0, 241, dtype=np.float64)


def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(
    adapter,
    loader,
    text_inputs: Mapping[str, torch.Tensor],
    device: str,
) -> Dict[str, float]:
    adapter.eval()
    with _autocast(device):
        text_embeddings = adapter.encode_text(text_inputs)
    total_loss = 0.0
    correct = 0
    count = 0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        with _autocast(device):
            image_embeddings = adapter.encode_image(images)
            logits = adapter.classification_logits(image_embeddings, text_embeddings)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        total_loss += float(loss)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.numel()
    if count == 0:
        raise RuntimeError("Cannot evaluate an empty split")
    return {
        "loss": total_loss / count,
        "accuracy": 100.0 * correct / count,
        "examples": count,
    }


def _scope_label(scope: str) -> Tuple[str, Optional[str], Optional[int]]:
    if scope == "global":
        return "global", None, None
    if scope.startswith("encoder/"):
        return "tower", scope.split("/", 1)[1], None
    _, tower, block = scope.split("/")
    return "block", tower, int(block)


def update_metric_rows(
    measurement: StepMeasurement,
    scope_map: ScopeMap,
    *,
    step: int,
    epoch: int,
    batch_index: int,
    batch_size: int,
    learning_rate: float,
) -> List[Dict[str, Any]]:
    budgets = compute_scope_budgets(
        measurement.before,
        measurement.deltas,
        scope_map,
        levels=("global", "encoder"),
    )
    rows = []
    for scope, value in budgets.items():
        level, tower, block = _scope_label(scope)
        rows.append(
            {
                "step": step,
                "epoch": epoch,
                "batch": batch_index,
                "batch_size": batch_size,
                "scope": level,
                "tower": tower,
                "block": block,
                "parameter_l2": value.parameter_l2,
                "update_l2": value.update_l2,
                "relative_budget": value.relative_budget,
                "learning_rate": learning_rate,
            }
        )
    return rows


def drift_metric_rows(
    initial: Mapping[str, torch.Tensor], scope_map: ScopeMap, epoch: int
) -> List[Dict[str, Any]]:
    drifts = compute_parameter_drift(
        initial, scope_map, levels=("global", "encoder")
    )
    rows = []
    for scope, value in drifts.items():
        level, tower, block = _scope_label(scope)
        rows.append(
            {
                "epoch": epoch,
                "scope": level,
                "tower": tower,
                "block": block,
                "initial_l2": value.initial_l2,
                "delta_l2": value.delta_l2,
                "relative_drift": value.relative_drift,
            }
        )
    return rows


def bias_corrected_adam_v(
    optimizer: torch.optim.Optimizer, scope_map: ScopeMap
) -> Optional[Dict[str, torch.Tensor]]:
    result: Dict[str, torch.Tensor] = {}
    for name, parameter in scope_map.named_parameters:
        state = optimizer.state.get(parameter, {})
        if "exp_avg_sq" not in state or "step" not in state:
            return None
        step_value = state["step"]
        step = float(step_value.item()) if torch.is_tensor(step_value) else float(step_value)
        if step <= 0:
            return None
        beta2 = float(optimizer.param_groups[0]["betas"][1])
        result[name] = (
            state["exp_avg_sq"].detach().float().cpu() / (1.0 - beta2**step)
        )
    return result


def _cat(values: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.cat([values[name].detach().float().cpu().reshape(-1) for name in names])


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.double()))


def _ratio(numerator: float, denominator: float, eps: float = EPS_METRIC) -> float:
    return numerator / (denominator + eps)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= EPS_METRIC:
        return float("nan")
    return float(torch.dot(left.double(), right.double()) / denominator)


def _moment_scope_row(
    names: Sequence[str],
    few_m: Mapping[str, torch.Tensor],
    few_v: Mapping[str, torch.Tensor],
    full_m: Mapping[str, torch.Tensor],
    full_v: Mapping[str, torch.Tensor],
    ema_v: Optional[Mapping[str, torch.Tensor]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    mf = _cat(few_m, names)
    vf = _cat(few_v, names)
    mr = _cat(full_m, names)
    vr = _cat(full_v, names)
    z = torch.log10((vf.double() + EPS_METRIC) / (vr.double() + EPS_METRIC))
    absolute_z = z.abs()
    absolute_mean = float(absolute_z.mean())
    # torch.quantile currently rejects tensors above 2**24 elements, while
    # the global SigLIP block vector has 170,108,928 coordinates. NumPy's
    # partition-based implementation supports the full vector. It operates on
    # the tensor-backed array in place to avoid another multi-gigabyte copy;
    # ordering is immaterial after these distribution summaries.
    mismatch_median, mismatch_p90, mismatch_p99 = np.quantile(
        absolute_z.numpy(),
        [0.5, 0.9, 0.99],
        overwrite_input=True,
    )
    u_ff = mf.double() / (vf.double().sqrt() + ADAM_EPS)
    u_rr = mr.double() / (vr.double().sqrt() + ADAM_EPS)
    u_fr = mf.double() / (vr.double().sqrt() + ADAM_EPS)
    u_paper_ff = mf.double() / torch.sqrt(vf.double() + ADAM_EPS)
    u_paper_rr = mr.double() / torch.sqrt(vr.double() + ADAM_EPS)
    hist, _ = np.histogram(z.numpy(), bins=HISTOGRAM_EDGES)
    ema_available = ema_v is not None
    ema_norm = float("nan")
    ema_to_few = float("nan")
    if ema_available:
        ema = _cat(ema_v, names)
        ema_norm = _norm(ema)
        ema_to_few = _ratio(ema_norm, _norm(vf))
    row = {
        "coordinates": int(z.numel()),
        "m_few_l2": _norm(mf),
        "m_full_l2": _norm(mr),
        "v_few_l2": _norm(vf),
        "v_full_l2": _norm(vr),
        "v_inst_few_l2": _norm(mf * mf),
        "v_inst_full_l2": _norm(mr * mr),
        "r_m": _ratio(_norm(mf), _norm(mr)),
        "r_v": _ratio(_norm(vf), _norm(vr)),
        "z_mean_signed": float(z.mean()),
        "mismatch_mean_abs": absolute_mean,
        "mismatch_median_abs": float(mismatch_median),
        "mismatch_p90_abs": float(mismatch_p90),
        "mismatch_p99_abs": float(mismatch_p99),
        "fraction_z_lt_0": float((z < 0).double().mean()),
        "fraction_z_lt_minus_1": float((z < -1).double().mean()),
        "fraction_z_lt_minus_2": float((z < -2).double().mean()),
        "u_adam_ff_l2": _norm(u_ff),
        "u_adam_rr_l2": _norm(u_rr),
        "r_u_adam": _ratio(_norm(u_ff), _norm(u_rr)),
        "cosine_u_ff_u_rr": _cosine(u_ff, u_rr),
        "u_adam_fr_l2": _norm(u_fr),
        "oracle_denominator_norm_ratio": _ratio(_norm(u_fr), _norm(u_rr)),
        "cosine_u_fr_u_rr": _cosine(u_fr, u_rr),
        "u_paper_ff_l2": _norm(u_paper_ff),
        "u_paper_rr_l2": _norm(u_paper_rr),
        "r_u_paper": _ratio(_norm(u_paper_ff), _norm(u_paper_rr)),
        "ema_state_available": ema_available,
        "v_ema_l2": ema_norm,
        "v_ema_to_v_few": ema_to_few,
        "adam_eps": ADAM_EPS,
        "eps_metric": EPS_METRIC,
    }
    return row, hist


def summarize_moments(
    few_m: Mapping[str, torch.Tensor],
    few_v: Mapping[str, torch.Tensor],
    full_m: Mapping[str, torch.Tensor],
    full_v: Mapping[str, torch.Tensor],
    scope_map: ScopeMap,
    *,
    epoch: int,
    ema_v: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    outputs = {"global": [], "tower": [], "block": []}
    histograms: Dict[str, np.ndarray] = {}
    groups = scope_map.all_groups()
    for scope, names in groups.items():
        row, histogram = _moment_scope_row(
            names, few_m, few_v, full_m, full_v, ema_v
        )
        level, tower, block = _scope_label(scope)
        row.update({"epoch": epoch, "scope": level, "tower": tower, "block": block})
        outputs[level].append(row)
        histogram_key = f"epoch_{epoch}/{scope}"
        histograms[histogram_key] = histogram
    return outputs["global"], outputs["tower"], outputs["block"], histograms


def collapse_metrics(r_m_0: float, r_m_30: float, r_v_0: float, r_v_30: float):
    c_m = _ratio(r_m_0, r_m_30)
    c_v = _ratio(r_v_0, r_v_30)
    return {"c_m": c_m, "c_v": c_v, "collapse_ratio": _ratio(c_v, c_m)}


def gap_closure(adam_m0: float, adam: float, sgd: float, tolerance: float = 1e-12):
    denominator = sgd - adam
    if abs(denominator) <= tolerance:
        return float("nan")
    return 100.0 * (adam_m0 - adam) / denominator


__all__ = [
    "ADAM_EPS",
    "EPS_METRIC",
    "HISTOGRAM_EDGES",
    "evaluate",
    "update_metric_rows",
    "drift_metric_rows",
    "bias_corrected_adam_v",
    "summarize_moments",
    "collapse_metrics",
    "gap_closure",
]

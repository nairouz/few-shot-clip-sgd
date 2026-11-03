#!/usr/bin/env python3
"""Experiment B — Local 1D landscape along the actual SGD step.

Goal
----
Measure the local safe step radius of SGD during few-shot CLIP adaptation.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (but backbone is configurable).
- 1-shot adaptation by default.
- Full-support training: one batch per epoch when --full_support_train is set.
- No augmentation by default (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default.
- At each checkpoint Theta_t, build the actual SGD step
      Delta_t = -lr * grad_few(Theta_t)
  and evaluate local 1D loss curves along
      Theta_t + c * Delta_t,
  where c is a step multiplier and c = 1 corresponds to the actual SGD step.
- Auxiliary comparison directions are also supported:
      * random direction (same norm as Delta_t, sign aligned with Delta_t)
      * top Hessian-eigenvector direction on the few-shot support loss
        (same norm as Delta_t, sign aligned with Delta_t)

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_direction_stats.csv
<out_dir>/<run_name>_localstep_curves.csv
<out_dir>/<run_name>_safe_radius.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

The companion plotting script is:
  loss_landscape_experiment_files/plot_experimentB_localstep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import torch
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader


def _import_clip_safely():
    """Import clip while handling torchvision builds without an exported NMS op.

    Some environments fail while importing torchvision with
        RuntimeError: operator torchvision::nms does not exist
    because torchvision's fake registration expects the operator schema to
    exist even though the geometry experiments never use NMS. We first try a
    normal import. Only if that specific error appears do we clear the partial
    torchvision import state, define the missing schema, and retry.
    """
    try:
        import clip as _clip
        return _clip
    except RuntimeError as exc:
        msg = str(exc)
        if 'operator torchvision::nms does not exist' not in msg:
            raise
        for name in list(sys.modules):
            if name == 'torchvision' or name.startswith('torchvision.'):
                sys.modules.pop(name, None)
        lib = torch.library.Library('torchvision', 'DEF')
        lib.define('nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor')
        import clip as _clip
        return _clip


clip = _import_clip_safely()
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


StateDictCPU = "OrderedDict[str, torch.Tensor]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    parser.add_argument("--root_path", type=str, default="~/datasets")
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="both",
        choices=["vision", "text", "both"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD"],
        help="Experiment B is defined for the SGD trajectory.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Ignored when --full_support_train is set.",
    )
    parser.add_argument(
        "--full_support_train",
        action="store_true",
        help="Use one full-support batch per epoch.",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=str,
        default="0,1,2,5,10,20,30",
        help="Comma-separated epochs t whose trainable checkpoints are saved and probed.",
    )
    parser.add_argument(
        "--probe_directions",
        type=str,
        default="sgd_step,random,hessian_top",
        help=(
            "Comma-separated probe directions. Supported values: sgd_step, random, hessian_top. "
            "In this full-support vanilla-SGD setup, sgd_step and -grad_few are identical directions, "
            "so a separate negative-gradient probe is redundant and not duplicated."
        ),
    )
    parser.add_argument(
        "--step_mult_min",
        type=float,
        default=-1.0,
        help="Minimum step multiplier c in Theta_t + c * Delta_t.",
    )
    parser.add_argument(
        "--step_mult_max",
        type=float,
        default=3.0,
        help="Maximum step multiplier c in Theta_t + c * Delta_t.",
    )
    parser.add_argument(
        "--line_points",
        type=int,
        default=41,
        help="Number of multiplier points used to sample each local 1D loss curve.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size used for full-train / test loss evaluation.",
    )
    parser.add_argument(
        "--hessian_power_iters",
        type=int,
        default=6,
        help="Power-iteration steps used to estimate the top Hessian eigenvector on the support loss.",
    )
    parser.add_argument(
        "--hessian_backend",
        type=str,
        default="auto",
        choices=["auto", "exact", "finite_diff"],
        help=(
            "How to compute Hessian-vector products for the top-Hessian probe. "
            "'exact' uses second-order autograd, 'finite_diff' uses gradient finite differences, "
            "and 'auto' tries exact first then falls back to finite differences if the attention backend "
            "does not support second derivatives."
        ),
    )
    parser.add_argument(
        "--hessian_fd_eps",
        type=float,
        default=1e-3,
        help="Finite-difference epsilon used when Hessian-vector products fall back to gradient differencing.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentB_logs",
        help="Directory where logs and checkpoints are written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, one is generated automatically.",
    )
    args = parser.parse_args()
    args.root_path = resolve_root(args.root_path)
    args.checkpoint_epochs = sorted({int(x) for x in args.checkpoint_epochs.split(",") if x.strip()})
    if 0 not in args.checkpoint_epochs:
        args.checkpoint_epochs = [0] + args.checkpoint_epochs
    if args.max_epochs not in args.checkpoint_epochs:
        args.checkpoint_epochs.append(args.max_epochs)
    args.checkpoint_epochs = sorted(set(args.checkpoint_epochs))
    if args.line_points < 5:
        raise ValueError("--line_points must be >= 5")
    if args.step_mult_max <= args.step_mult_min:
        raise ValueError("--step_mult_max must be > --step_mult_min")
    args.probe_directions = [x.strip() for x in args.probe_directions.split(",") if x.strip()]
    supported = {"sgd_step", "random", "hessian_top"}
    bad = set(args.probe_directions) - supported
    if bad:
        raise ValueError(f"Unsupported probe directions: {sorted(bad)}")
    return args


def load_clip_model(backbone: str, device: str):
    import getpass

    cache = os.getenv("CLIP_DOWNLOAD_ROOT")
    model, preprocess = clip.load(backbone, device=device, download_root=cache)
    return model, preprocess


def build_loader_from_source(data_source, *, batch_size: int, preprocess, is_train: bool, shuffle: bool, num_workers: int):
    if hasattr(data_source, "samples") and hasattr(data_source, "__getitem__"):
        return DataLoader(
            data_source,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return build_data_loader(
        data_source=data_source,
        batch_size=batch_size,
        tfm=preprocess,
        is_train=is_train,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def tokenize_class_texts(classnames: Sequence[str], template: Sequence[str], device: str) -> torch.Tensor:
    prompt_template = template[0]
    texts = [prompt_template.format(name.replace("_", " ")) for name in classnames]
    return clip.tokenize(texts).to(device)


def compute_loader_loss_acc(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target, reduction="mean")
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * int(target.size(0))
            total_correct += int((pred == target).sum().item())
            total_items += int(target.size(0))
    model.train(was_training)
    if total_items == 0:
        return float("nan"), float("nan")
    return total_loss / total_items, 100.0 * total_correct / total_items


def compute_bundle_metrics(
    model,
    few_loader,
    full_loader,
    test_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> Dict[str, float]:
    few_loss, few_acc = compute_loader_loss_acc(model, few_loader, class_tokens, logit_scale, device)
    full_loss, full_acc = compute_loader_loss_acc(model, full_loader, class_tokens, logit_scale, device)
    test_loss, test_acc = compute_loader_loss_acc(model, test_loader, class_tokens, logit_scale, device)
    return {
        "few_loss": float(few_loss),
        "few_acc": float(few_acc),
        "full_loss": float(full_loss),
        "full_acc": float(full_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def get_trainable_state(model: torch.nn.Module) -> StateDictCPU:
    state: StateDictCPU = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().clone()
    return state


def get_trainable_param_names(model: torch.nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def set_trainable_state(model: torch.nn.Module, state: MutableMapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def save_trainable_checkpoint(path: Path, state: MutableMapping[str, torch.Tensor], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": dict(state), "meta": meta}, path)


def load_trainable_checkpoint(path: Path) -> Tuple[StateDictCPU, Dict[str, object]]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = OrderedDict((k, v) for k, v in obj["trainable_state"].items())
    meta = dict(obj.get("meta", {}))
    return state, meta


def _state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    s = 0.0
    for k in a.keys():
        s += float((a[k].double() * b[k].double()).sum().item())
    return s


def _state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return _state_dot(a, a)


def _state_l2(a: MutableMapping[str, torch.Tensor]) -> float:
    return math.sqrt(max(_state_sqnorm(a), 0.0))


def _state_scale(a: MutableMapping[str, torch.Tensor], scale: float) -> StateDictCPU:
    out: StateDictCPU = OrderedDict()
    for k, v in a.items():
        out[k] = v * float(scale)
    return out


def _state_add(base: MutableMapping[str, torch.Tensor], direction: MutableMapping[str, torch.Tensor], scale: float) -> StateDictCPU:
    out: StateDictCPU = OrderedDict()
    s = float(scale)
    for k in base.keys():
        out[k] = base[k] + s * direction[k]
    return out


def _state_zero_like(state: MutableMapping[str, torch.Tensor]) -> StateDictCPU:
    return OrderedDict((k, torch.zeros_like(v)) for k, v in state.items())


def _maybe_align_sign(direction: StateDictCPU, reference: StateDictCPU) -> StateDictCPU:
    if _state_dot(direction, reference) < 0.0:
        return _state_scale(direction, -1.0)
    return OrderedDict((k, v.clone()) for k, v in direction.items())


def _normalized_random_direction(reference_state: StateDictCPU, target_norm: float, seed: int) -> StateDictCPU:
    if target_norm <= 0.0:
        return _state_zero_like(reference_state)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    out: StateDictCPU = OrderedDict()
    sq = 0.0
    for k, v in reference_state.items():
        r = torch.randn(v.shape, generator=g, dtype=v.dtype)
        out[k] = r
        sq += float((r.double() * r.double()).sum().item())
    norm = math.sqrt(max(sq, 1e-32))
    out = _state_scale(out, target_norm / norm)
    return _maybe_align_sign(out, reference_state)


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


def compute_support_loss(model, images, target, class_tokens: torch.Tensor, logit_scale: float, device: str) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = model.encode_text(class_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = logit_scale * (image_features @ text_features.t())
    return F.cross_entropy(logits, target)


def get_full_support_batch(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    for images, target in loader:
        return images.cpu(), target.cpu()
    raise RuntimeError("Support loader is empty.")


def _sdp_math_only_context(device: str):
    """Prefer the math SDPA backend when second-order derivatives are needed.

    PyTorch's efficient / flash attention kernels often do not implement the
    backward-of-backward required by Hessian-vector products. The math backend is
    slower but supports higher-order autograd much more reliably.
    """
    if not str(device).startswith("cuda"):
        return nullcontext()
    sdp = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp is None:
        return nullcontext()
    try:
        return sdp(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except TypeError:
        return nullcontext()


def compute_support_grads(
    model,
    params: Sequence[torch.nn.Parameter],
    class_tokens: torch.Tensor,
    logit_scale: float,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    *,
    create_graph: bool = False,
    retain_graph: bool = False,
):
    model.zero_grad(set_to_none=True)
    loss = compute_support_loss(model, support_batch[0], support_batch[1], class_tokens, logit_scale, device)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=retain_graph)
    return loss, grads


def hvp_support_loss_finite_difference(
    model,
    params: Sequence[torch.nn.Parameter],
    class_tokens: torch.Tensor,
    logit_scale: float,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    vec_list: Sequence[torch.Tensor],
    device: str,
    fd_eps: float,
) -> List[torch.Tensor]:
    eps = float(fd_eps)
    if eps <= 0.0:
        raise ValueError("fd_eps must be > 0")

    with torch.no_grad():
        for p, v in zip(params, vec_list):
            p.add_(v, alpha=eps)
    _, grads_pos = compute_support_grads(
        model, params, class_tokens, logit_scale, support_batch, device, create_graph=False, retain_graph=False
    )

    with torch.no_grad():
        for p, v in zip(params, vec_list):
            p.add_(v, alpha=-2.0 * eps)
    _, grads_neg = compute_support_grads(
        model, params, class_tokens, logit_scale, support_batch, device, create_graph=False, retain_graph=False
    )

    with torch.no_grad():
        for p, v in zip(params, vec_list):
            p.add_(v, alpha=eps)

    model.zero_grad(set_to_none=True)
    return [((gp.detach() - gn.detach()) / (2.0 * eps)) for gp, gn in zip(grads_pos, grads_neg)]


def compute_sgd_step_state(
    model,
    trainable_names: Sequence[str],
    class_tokens: torch.Tensor,
    logit_scale: float,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    lr: float,
) -> Tuple[StateDictCPU, float, float]:
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    images, target = support_batch
    model.zero_grad(set_to_none=True)
    loss = compute_support_loss(model, images, target, class_tokens, logit_scale, device)
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
    step_state: StateDictCPU = OrderedDict()
    sq = 0.0
    grad_sq = 0.0
    for name, g in zip(trainable_names, grads):
        gcpu = g.detach().cpu().clone()
        grad_sq += float((gcpu.double() * gcpu.double()).sum().item())
        step = -float(lr) * gcpu
        step_state[name] = step
        sq += float((step.double() * step.double()).sum().item())
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return step_state, math.sqrt(max(sq, 0.0)), math.sqrt(max(grad_sq, 0.0))


def hvp_support_loss(
    model,
    params: Sequence[torch.nn.Parameter],
    class_tokens: torch.Tensor,
    logit_scale: float,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    vec_list: Sequence[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    with _sdp_math_only_context(device):
        _, grads = compute_support_grads(
            model, params, class_tokens, logit_scale, support_batch, device, create_graph=True, retain_graph=True
        )
        dot = None
        for g, v in zip(grads, vec_list):
            term = (g * v).sum()
            dot = term if dot is None else dot + term
        hv = torch.autograd.grad(dot, params, retain_graph=False, create_graph=False)
    model.zero_grad(set_to_none=True)
    return [h.detach() for h in hv]


def _list_sqnorm(vecs: Sequence[torch.Tensor]) -> float:
    s = 0.0
    for v in vecs:
        s += float((v.double() * v.double()).sum().item())
    return s


def _list_dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += float((x.double() * y.double()).sum().item())
    return s


def _normalize_list(vecs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    n = math.sqrt(max(_list_sqnorm(vecs), 1e-32))
    return [v / n for v in vecs]


def estimate_top_hessian_direction(
    model,
    trainable_names: Sequence[str],
    class_tokens: torch.Tensor,
    logit_scale: float,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    target_norm: float,
    align_to: StateDictCPU,
    power_iters: int,
    seed: int,
    *,
    hessian_backend: str = "auto",
    hessian_fd_eps: float = 1e-3,
) -> Tuple[StateDictCPU, float]:
    if target_norm <= 0.0:
        return _state_zero_like(align_to), 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    vecs = [torch.randn(p.shape, device=device, dtype=p.dtype, generator=g) for p in params]
    vecs = _normalize_list(vecs)
    rayleigh = 0.0

    exact_ok = False
    if hessian_backend in {"auto", "exact"}:
        try:
            for _ in range(max(int(power_iters), 1)):
                hv = hvp_support_loss(model, params, class_tokens, logit_scale, support_batch, vecs, device)
                rayleigh = _list_dot(vecs, hv)
                hv_norm_sq = _list_sqnorm(hv)
                if hv_norm_sq <= 1e-32:
                    break
                vecs = [h / math.sqrt(hv_norm_sq) for h in hv]
            exact_ok = True
        except RuntimeError as exc:
            msg = str(exc)
            if hessian_backend == "exact":
                raise
            warnings.warn(
                "Exact Hessian-vector product failed; falling back to finite differences. "
                f"Original error: {msg}",
                RuntimeWarning,
            )
            # Reinitialize so the finite-difference path is deterministic and independent of partial progress.
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            vecs = [torch.randn(p.shape, device=device, dtype=p.dtype, generator=g) for p in params]
            vecs = _normalize_list(vecs)

    if (not exact_ok) and hessian_backend in {"auto", "finite_diff"}:
        for _ in range(max(int(power_iters), 1)):
            hv = hvp_support_loss_finite_difference(
                model, params, class_tokens, logit_scale, support_batch, vecs, device, hessian_fd_eps
            )
            rayleigh = _list_dot(vecs, hv)
            hv_norm_sq = _list_sqnorm(hv)
            if hv_norm_sq <= 1e-32:
                break
            vecs = [h / math.sqrt(hv_norm_sq) for h in hv]

    direction: StateDictCPU = OrderedDict()
    for name, v in zip(trainable_names, vecs):
        direction[name] = v.detach().cpu().clone()
    direction = _state_scale(direction, target_norm)
    direction = _maybe_align_sign(direction, align_to)
    return direction, float(rayleigh)


def local_curve_summary(step_mult: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    x = torch.tensor(list(step_mult), dtype=torch.float64)
    y = torch.tensor(list(values), dtype=torch.float64)
    idx0 = int(torch.argmin(torch.abs(x)).item())
    x0 = x[idx0:]
    y0 = y[idx0:]
    baseline = float(y[idx0].item())
    idx_min = int(torch.argmin(y0).item())
    mult_min = float(x0[idx_min].item())
    loss_min = float(y0[idx_min].item())

    def _safe_radius(tol_frac: float) -> float:
        thresh = baseline + tol_frac * max(abs(baseline), 1e-12)
        ok = torch.nonzero(y0 <= thresh, as_tuple=False).flatten()
        if ok.numel() == 0:
            return 0.0
        return float(x0[int(ok[-1].item())].item())

    return {
        "baseline_loss": baseline,
        "multiplier_at_min": mult_min,
        "loss_min": loss_min,
        "safe_noninc": _safe_radius(0.0),
        "safe_1pct": _safe_radius(0.01),
        "safe_5pct": _safe_radius(0.05),
        "loss_at_step1": float(y[int(torch.argmin(torch.abs(x - 1.0)).item())].item()),
    }


def default_run_name(args: argparse.Namespace) -> str:
    bb = args.backbone.replace("/", "-")
    return (
        f"experimentB_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
        f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
    )


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    run_name = args.run_name if args.run_name is not None else default_run_name(args)
    ckpt_dir = out_dir / "checkpoints" / run_name
    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)

    few_train_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=maybe_full_support_bs(len(few_dataset.train_x), args),
        preprocess=preprocess,
        is_train=True,
        shuffle=not args.full_support_train and args.batch_size > 0,
        num_workers=args.num_workers,
    )
    few_eval_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=max(1, len(few_dataset.train_x)),
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    full_eval_loader = build_loader_from_source(
        full_dataset.train_x,
        batch_size=args.eval_batch_size,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader_from_source(
        full_dataset.test,
        batch_size=args.eval_batch_size,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    support_batch = get_full_support_batch(few_eval_loader)

    clip_model = clip_model.to(args.device)
    clip_model.eval()
    mark_as_trainable(clip_model, args)
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    trainable_names = get_trainable_param_names(clip_model)
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    optimizer = SGD(trainable_params, lr=args.lr)
    scaler = GradScaler("cuda") if str(args.device).startswith("cuda") else GradScaler("cpu")

    init_state = get_trainable_state(clip_model)
    cfg = {
        "seed": args.seed,
        "device": args.device,
        "root_path": args.root_path,
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "position": args.position,
        "encoder": args.encoder,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "full_support_train": args.full_support_train,
        "checkpoint_epochs": args.checkpoint_epochs,
        "probe_directions": args.probe_directions,
        "step_mult_min": args.step_mult_min,
        "step_mult_max": args.step_mult_max,
        "line_points": args.line_points,
        "eval_batch_size": args.eval_batch_size,
        "hessian_power_iters": args.hessian_power_iters,
        "num_workers": args.num_workers,
        "out_dir": str(out_dir),
        "run_name": run_name,
        "few_size": len(few_dataset.train_x),
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "logit_scale": logit_scale,
        "no_augmentation": True,
        "actual_sgd_equals_negative_gradient": True,
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    trajectory_rows: List[Dict[str, object]] = []

    def eval_and_save(epoch: int, state: MutableMapping[str, torch.Tensor]) -> None:
        set_trainable_state(clip_model, state)
        metrics = compute_bundle_metrics(
            clip_model,
            few_eval_loader,
            full_eval_loader,
            test_loader,
            class_tokens,
            logit_scale,
            args.device,
        )
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_trainable.pt"
        save_trainable_checkpoint(
            ckpt_path,
            state,
            {
                "epoch": int(epoch),
                "dataset": args.dataset,
                "shots": args.shots,
                "backbone": args.backbone,
                "optimizer": args.optimizer,
                "position": args.position,
                "encoder": args.encoder,
            },
        )
        trajectory_rows.append(
            {
                "epoch": int(epoch),
                **metrics,
                "checkpoint_path": str(ckpt_path),
            }
        )
        print(
            f"[ExperimentB] epoch={epoch:02d} | few_loss={metrics['few_loss']:.4f} | full_loss={metrics['full_loss']:.4f} | "
            f"test_loss={metrics['test_loss']:.4f} | few_acc={metrics['few_acc']:.2f} | test_acc={metrics['test_acc']:.2f}"
        )

    eval_and_save(0, init_state)

    start_time = time.perf_counter()
    amp_enabled = str(args.device).startswith("cuda")

    for epoch in range(1, args.max_epochs + 1):
        clip_model.train()
        for images, target in few_train_loader:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = clip_model.encode_text(class_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        if epoch in args.checkpoint_epochs:
            eval_and_save(epoch, get_trainable_state(clip_model))

    elapsed = time.perf_counter() - start_time
    print(f"[ExperimentB] Training finished in {elapsed:.1f}s")

    multipliers = torch.linspace(args.step_mult_min, args.step_mult_max, args.line_points).double().tolist()
    direction_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    safe_rows: List[Dict[str, object]] = []

    for row in trajectory_rows:
        epoch = int(row["epoch"])
        state_t, meta = load_trainable_checkpoint(Path(row["checkpoint_path"]))
        set_trainable_state(clip_model, state_t)

        sgd_dir, step_l2, grad_l2 = compute_sgd_step_state(
            clip_model,
            trainable_names,
            class_tokens,
            logit_scale,
            support_batch,
            args.device,
            args.lr,
        )
        direction_map: Dict[str, StateDictCPU] = {"sgd_step": sgd_dir}
        eigval_top = float("nan")

        if "random" in args.probe_directions:
            direction_map["random"] = _normalized_random_direction(sgd_dir, step_l2, args.seed + 1000 + epoch)

        if "hessian_top" in args.probe_directions:
            hdir, eigval_top = estimate_top_hessian_direction(
                clip_model,
                trainable_names,
                class_tokens,
                logit_scale,
                support_batch,
                args.device,
                step_l2,
                sgd_dir,
                args.hessian_power_iters,
                args.seed + 2000 + epoch,
                hessian_backend=args.hessian_backend,
                hessian_fd_eps=args.hessian_fd_eps,
            )
            direction_map["hessian_top"] = hdir

        for direction_name in args.probe_directions:
            dstate = direction_map[direction_name]
            dnorm = _state_l2(dstate)
            direction_rows.append(
                {
                    "epoch": epoch,
                    "direction": direction_name,
                    "step_l2": float(step_l2),
                    "direction_l2": float(dnorm),
                    "grad_l2": float(grad_l2),
                    "hessian_top_eigval": float(eigval_top) if direction_name == "hessian_top" else float("nan"),
                }
            )

            few_curve: List[float] = []
            full_curve: List[float] = []
            test_curve: List[float] = []
            for mult in multipliers:
                probe_state = _state_add(state_t, dstate, float(mult))
                set_trainable_state(clip_model, probe_state)
                metrics = compute_bundle_metrics(
                    clip_model,
                    few_eval_loader,
                    full_eval_loader,
                    test_loader,
                    class_tokens,
                    logit_scale,
                    args.device,
                )
                few_curve.append(metrics["few_loss"])
                full_curve.append(metrics["full_loss"])
                test_curve.append(metrics["test_loss"])
                curve_rows.append(
                    {
                        "epoch": epoch,
                        "direction": direction_name,
                        "step_multiplier": float(mult),
                        "absolute_step_radius": float(mult * step_l2),
                        "step_l2": float(step_l2),
                        **metrics,
                    }
                )

            few_stats = local_curve_summary(multipliers, few_curve)
            full_stats = local_curve_summary(multipliers, full_curve)
            test_stats = local_curve_summary(multipliers, test_curve)
            safe_rows.append(
                {
                    "epoch": epoch,
                    "direction": direction_name,
                    "step_l2": float(step_l2),
                    "grad_l2": float(grad_l2),
                    "few_baseline_loss": few_stats["baseline_loss"],
                    "few_multiplier_at_min": few_stats["multiplier_at_min"],
                    "few_loss_min": few_stats["loss_min"],
                    "few_safe_noninc": few_stats["safe_noninc"],
                    "few_safe_1pct": few_stats["safe_1pct"],
                    "few_safe_5pct": few_stats["safe_5pct"],
                    "few_loss_at_step1": few_stats["loss_at_step1"],
                    "full_baseline_loss": full_stats["baseline_loss"],
                    "full_multiplier_at_min": full_stats["multiplier_at_min"],
                    "full_loss_min": full_stats["loss_min"],
                    "full_safe_noninc": full_stats["safe_noninc"],
                    "full_safe_1pct": full_stats["safe_1pct"],
                    "full_safe_5pct": full_stats["safe_5pct"],
                    "full_loss_at_step1": full_stats["loss_at_step1"],
                    "test_baseline_loss": test_stats["baseline_loss"],
                    "test_multiplier_at_min": test_stats["multiplier_at_min"],
                    "test_loss_min": test_stats["loss_min"],
                    "test_safe_noninc": test_stats["safe_noninc"],
                    "test_safe_1pct": test_stats["safe_1pct"],
                    "test_safe_5pct": test_stats["safe_5pct"],
                    "test_loss_at_step1": test_stats["loss_at_step1"],
                }
            )

        set_trainable_state(clip_model, state_t)
        print(f"[ExperimentB] Probed checkpoint t={epoch:02d}")

    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with open(out_dir / f"{run_name}_direction_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(direction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(direction_rows)

    with open(out_dir / f"{run_name}_localstep_curves.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    with open(out_dir / f"{run_name}_safe_radius.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(safe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(safe_rows)

    summary = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "probe_directions": args.probe_directions,
        "line_points": args.line_points,
        "step_mult_min": args.step_mult_min,
        "step_mult_max": args.step_mult_max,
        "num_checkpoints_probed": len(trajectory_rows),
        "actual_sgd_equals_negative_gradient": True,
    }
    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ExperimentB] Wrote logs to: {out_dir}")
    print(f"[ExperimentB] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

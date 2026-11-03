#!/usr/bin/env python3
"""Experiment J — Noise-scale / batch-size landscape study for SGD.

Goal
----
Understand whether SGD's stochasticity helps it find flatter regions during
few-shot CLIP adaptation.

Protocol implemented here
-------------------------
This script follows the Experiment J protocol proposed for few-shot CLIP.
It now supports two budgeting modes:

- `epoch_matched`: original Experiment J, same number of epochs for every variant.
- `update_matched`: fairness-control follow-up, same total number of optimizer updates
  as the full-batch baseline.

Common setup:
- CLIP ViT-B/16 by default (backbone configurable).
- 1-shot adaptation by default.
- No augmentation (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default.
- Compare the following SGD variants on the *same* few-shot support set:
    1) full-support SGD without momentum          (baseline)
    2) half-support mini-batch SGD                (stochastic)
    3) quarter-support mini-batch SGD             (more stochastic)
    4) full-support SGD + additive gradient noise (algorithmic stochasticity)
    5) full-support SGD with momentum             (velocity / averaging effect)
- At each saved checkpoint, measure:
    * support / full-train / test loss and accuracy
    * full-train gradient norm
    * support gradient norm
    * local curvature proxy on full-train loss along:
        - the normalized full-train gradient direction
        - a normalized random direction
    * empirical batch-noise ratio on the support set
    * injected-noise ratio (for the noise-injection variant)
    * relative drift from the pretrained CLIP initialization
- After training, measure final basin connectivity by linear interpolation
  between the baseline final solution and every other variant.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_variant_final.csv
<out_dir>/<run_name>_connectivity.csv
<out_dir>/<run_name>_connectivity_summary.json
<out_dir>/checkpoints/<run_name>/<budget_protocol>/<variant>/*_trainable.pt

Companion plotting script:
  loss_landscape_experiment_files/plot_experimentJ_noise_scale.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
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


def _import_torchvision_safely():
    """Import torchvision without breaking environments that already provide NMS.

    Some CPU-only or mismatched torch/torchvision installations fail at import time
    with `operator torchvision::nms does not exist`. In those environments we
    define a dummy schema and retry. In normal CUDA builds we must *not* define
    the schema preemptively, otherwise torchvision later tries to register the
    real op and PyTorch raises a duplicate-registration error.
    """
    try:
        torchvision = importlib.import_module("torchvision")
        transforms = importlib.import_module("torchvision.transforms")
        return torchvision, transforms
    except RuntimeError as exc:
        if "operator torchvision::nms does not exist" not in str(exc):
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        torchvision = importlib.import_module("torchvision")
        transforms = importlib.import_module("torchvision.transforms")
        return torchvision, transforms


_torchvision, _transforms = _import_torchvision_safely()

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


TensorState = "OrderedDict[str, torch.Tensor]"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    batch_size: int
    momentum: float
    noise_rel: float


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument(
        "--full_support_train",
        action="store_true",
        help="Keep the baseline variant in full-support / one-batch-per-epoch mode.",
    )
    parser.add_argument(
        "--budget_protocol",
        type=str,
        default="epoch_matched",
        choices=["epoch_matched", "update_matched"],
        help=(
            "Budgeting protocol for Experiment J. `epoch_matched` reproduces the original experiment, "
            "where every variant trains for the same number of epochs. `update_matched` is the fairness "
            "control: every variant gets the same total number of optimizer updates as the full-batch baseline."
        ),
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=str,
        default="0,1,2,5,10,20,30",
        help="Comma-separated epochs t whose checkpoints are saved for epoch-matched runs.",
    )
    parser.add_argument(
        "--checkpoint_updates",
        type=str,
        default="0,1,2,5,10,20,30",
        help=(
            "Comma-separated optimizer-update counts saved for update-matched runs. "
            "The default mirrors the baseline full-batch schedule because the baseline makes one update per epoch."
        ),
    )
    parser.add_argument(
        "--target_total_updates",
        type=int,
        default=-1,
        help=(
            "Optional override for the total number of optimizer updates used in `update_matched` runs. "
            "If negative, the script uses the baseline full-batch update budget: baseline_updates_per_epoch * max_epochs."
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size used for full-train / test loss evaluation.",
    )
    parser.add_argument(
        "--mini_batch_divisors",
        type=str,
        default="2,4",
        help=(
            "Comma-separated divisors that define the stochastic mini-batch variants. "
            "For a support set of size N, divisor d creates a variant with batch size ceil(N/d)."
        ),
    )
    parser.add_argument(
        "--momentum_value",
        type=float,
        default=0.9,
        help="Momentum used by the full-support momentum variant.",
    )
    parser.add_argument(
        "--noise_rel_scale",
        type=float,
        default=0.10,
        help=(
            "Relative gradient-noise scale for the full-support noise-injection variant. "
            "Noise is added as sigma * ||g|| / sqrt(d) * N(0, I), so the expected noise-to-signal L2 ratio is sigma."
        ),
    )
    parser.add_argument(
        "--curvature_rel_radius",
        type=float,
        default=1e-4,
        help=(
            "Relative perturbation radius used in the finite-difference curvature proxy. "
            "The absolute radius is curvature_rel_radius * ||Theta_trainable||_2."
        ),
    )
    parser.add_argument(
        "--interp_points",
        type=int,
        default=61,
        help="Number of alpha points used for final baseline↔variant connectivity curves.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentJ_logs",
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

    args.checkpoint_updates = sorted({int(x) for x in args.checkpoint_updates.split(",") if x.strip()})
    if 0 not in args.checkpoint_updates:
        args.checkpoint_updates = [0] + args.checkpoint_updates
    args.checkpoint_updates = sorted(set(args.checkpoint_updates))

    args.mini_batch_divisors = [int(x) for x in args.mini_batch_divisors.split(",") if x.strip()]
    if args.interp_points < 2:
        raise ValueError("--interp_points must be >= 2")
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


def get_trainable_state(model: torch.nn.Module) -> TensorState:
    state: TensorState = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().clone()
    return state


def set_trainable_state(model: torch.nn.Module, state: MutableMapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def save_trainable_checkpoint(path: Path, state: MutableMapping[str, torch.Tensor], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": dict(state), "meta": meta}, path)


def _state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    s = 0.0
    for k in a.keys():
        s += float((a[k].double() * b[k].double()).sum().item())
    return s


def _state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return _state_dot(a, a)


def _state_norm(a: MutableMapping[str, torch.Tensor]) -> float:
    return math.sqrt(max(_state_sqnorm(a), 1e-32))


def _state_sub(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> TensorState:
    out: TensorState = OrderedDict()
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out


def _state_add_scaled(base: MutableMapping[str, torch.Tensor], direction: MutableMapping[str, torch.Tensor], scale: float) -> TensorState:
    out: TensorState = OrderedDict()
    for k in base.keys():
        out[k] = base[k] + scale * direction[k]
    return out


def interpolate_state(
    state_a: MutableMapping[str, torch.Tensor],
    state_b: MutableMapping[str, torch.Tensor],
    alpha: float,
) -> TensorState:
    out: TensorState = OrderedDict()
    for k in state_a.keys():
        out[k] = (1.0 - alpha) * state_a[k] + alpha * state_b[k]
    return out


def interpolation_barrier(alpha: torch.Tensor, values: torch.Tensor) -> Dict[str, float]:
    start = float(values[0].item())
    end = float(values[-1].item())
    linear = (1.0 - alpha) * start + alpha * end
    excess = values - linear
    idx_min = int(torch.argmin(values).item())
    idx_max = int(torch.argmax(excess).item())
    vmin = float(values[idx_min].item())
    width1_mask = values <= (vmin + 0.01 * max(start - vmin, 1e-12))
    width5_mask = values <= (vmin + 0.05 * max(start - vmin, 1e-12))

    def _width(mask: torch.Tensor) -> float:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0.0
        return float(alpha[idx[-1]].item() - alpha[idx[0]].item())

    return {
        "alpha_min": float(alpha[idx_min].item()),
        "loss_min": vmin,
        "barrier": float(excess[idx_max].item()),
        "alpha_barrier": float(alpha[idx_max].item()),
        "width_1pct": _width(width1_mask),
        "width_5pct": _width(width5_mask),
    }


def compute_dataset_gradient(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params: Sequence[Tuple[str, torch.nn.Parameter]],
    device: str,
) -> Tuple[TensorState, float]:
    was_training = model.training
    model.eval()
    total_size = len(loader.dataset)
    if total_size <= 0:
        raise RuntimeError("Empty loader in compute_dataset_gradient")

    grad_state: TensorState = OrderedDict((name, torch.zeros_like(p, dtype=torch.float32, device="cpu")) for name, p in params)

    for images, target in loader:
        batch_size = int(target.size(0))
        batch_weight = float(batch_size) / float(total_size)
        model.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features.t())
        loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()
        for name, p in params:
            if p.grad is not None:
                grad_state[name].add_(p.grad.detach().float().cpu(), alpha=batch_weight)

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return grad_state, _state_norm(grad_state)


def normalize_state_direction(state: MutableMapping[str, torch.Tensor]) -> TensorState:
    n = _state_norm(state)
    out: TensorState = OrderedDict()
    for k, v in state.items():
        out[k] = v / max(n, 1e-32)
    return out


def build_random_direction_like(ref_state: MutableMapping[str, torch.Tensor], seed: int) -> TensorState:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out: TensorState = OrderedDict()
    for k, v in ref_state.items():
        out[k] = torch.randn(v.shape, generator=gen, dtype=v.dtype)
    return normalize_state_direction(out)


def evaluate_loss_for_state(
    model,
    state: MutableMapping[str, torch.Tensor],
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> float:
    current = get_trainable_state(model)
    set_trainable_state(model, state)
    loss, _ = compute_loader_loss_acc(model, loader, class_tokens, logit_scale, device)
    set_trainable_state(model, current)
    return float(loss)


def finite_difference_curvature(
    model,
    base_state: MutableMapping[str, torch.Tensor],
    direction: MutableMapping[str, torch.Tensor],
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    base_loss: float,
    rel_radius: float,
) -> Tuple[float, float]:
    theta_norm = _state_norm(base_state)
    eps = rel_radius * max(theta_norm, 1e-12)
    plus_state = _state_add_scaled(base_state, direction, +eps)
    minus_state = _state_add_scaled(base_state, direction, -eps)
    l_plus = evaluate_loss_for_state(model, plus_state, loader, class_tokens, logit_scale, device)
    l_minus = evaluate_loss_for_state(model, minus_state, loader, class_tokens, logit_scale, device)
    curv = (l_plus + l_minus - 2.0 * base_loss) / max(eps * eps, 1e-32)
    return float(curv), float(eps)


def estimate_batch_noise_ratio(
    model,
    data_source,
    probe_batch_size: int,
    preprocess,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params: Sequence[Tuple[str, torch.nn.Parameter]],
    device: str,
    num_workers: int,
) -> Tuple[float, float]:
    total_items = len(data_source)
    if total_items <= 0:
        return float("nan"), float("nan")
    if probe_batch_size >= total_items:
        return 0.0, 0.0

    full_loader = build_loader_from_source(
        data_source,
        batch_size=total_items,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=num_workers,
    )
    full_grad, full_grad_norm = compute_dataset_gradient(model, full_loader, class_tokens, logit_scale, params, device)

    micro_loader = build_loader_from_source(
        data_source,
        batch_size=probe_batch_size,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=num_workers,
    )

    was_training = model.training
    model.eval()
    noise_sq = 0.0
    total_weight = 0.0

    for images, target in micro_loader:
        batch_items = int(target.size(0))
        weight = float(batch_items) / float(total_items)
        model.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features.t())
        loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()
        batch_diff_sq = 0.0
        for name, p in params:
            if p.grad is None:
                continue
            diff = p.grad.detach().float().cpu() - full_grad[name]
            batch_diff_sq += float((diff.double() * diff.double()).sum().item())
        noise_sq += weight * batch_diff_sq
        total_weight += weight

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    noise_rms = math.sqrt(max(noise_sq / max(total_weight, 1e-12), 0.0))
    ratio = noise_rms / max(full_grad_norm, 1e-32)
    return float(ratio), float(full_grad_norm)


def maybe_full_support_bs(num_items: int, full_support: bool) -> int:
    return max(1, num_items) if full_support else max(1, num_items)


def build_default_variants(args: argparse.Namespace, support_size: int) -> List[VariantSpec]:
    variants = [
        VariantSpec(
            name="fullbatch_nomom",
            label="full batch",
            batch_size=max(1, support_size) if args.full_support_train else max(1, support_size),
            momentum=0.0,
            noise_rel=0.0,
        )
    ]
    for d in args.mini_batch_divisors:
        bs = max(1, math.ceil(support_size / float(d)))
        tag = "half batch" if d == 2 else ("quarter batch" if d == 4 else f"1/{d} batch")
        variants.append(
            VariantSpec(
                name=f"div{d}_nomom",
                label=tag,
                batch_size=bs,
                momentum=0.0,
                noise_rel=0.0,
            )
        )
    variants.append(
        VariantSpec(
            name="fullbatch_noise",
            label=rf"full batch + noise",
            batch_size=max(1, support_size),
            momentum=0.0,
            noise_rel=float(args.noise_rel_scale),
        )
    )
    variants.append(
        VariantSpec(
            name="fullbatch_momentum",
            label=rf"full batch + mom",
            batch_size=max(1, support_size),
            momentum=float(args.momentum_value),
            noise_rel=0.0,
        )
    )
    return variants


def build_budget_schedule(
    args: argparse.Namespace,
    *,
    baseline_total_updates: int,
) -> Dict[str, object]:
    if args.budget_protocol == "epoch_matched":
        checkpoint_epochs = sorted(set(int(x) for x in args.checkpoint_epochs if x >= 0))
        if 0 not in checkpoint_epochs:
            checkpoint_epochs = [0] + checkpoint_epochs
        if args.max_epochs not in checkpoint_epochs:
            checkpoint_epochs.append(args.max_epochs)
        checkpoint_epochs = sorted(set(checkpoint_epochs))
        return {
            "protocol": "epoch_matched",
            "checkpoint_epochs": checkpoint_epochs,
            "checkpoint_updates": None,
            "target_total_updates": None,
            "baseline_total_updates": int(baseline_total_updates),
        }

    target_total_updates = int(args.target_total_updates) if int(args.target_total_updates) > 0 else int(baseline_total_updates)
    checkpoint_updates = sorted(set(int(x) for x in args.checkpoint_updates if 0 <= int(x) <= target_total_updates))
    if 0 not in checkpoint_updates:
        checkpoint_updates = [0] + checkpoint_updates
    if target_total_updates not in checkpoint_updates:
        checkpoint_updates.append(target_total_updates)
    checkpoint_updates = sorted(set(checkpoint_updates))
    return {
        "protocol": "update_matched",
        "checkpoint_epochs": None,
        "checkpoint_updates": checkpoint_updates,
        "target_total_updates": int(target_total_updates),
        "baseline_total_updates": int(baseline_total_updates),
    }


def add_gradient_noise_(params: Sequence[torch.nn.Parameter], noise_rel: float) -> float:
    if noise_rel <= 0.0:
        return 0.0
    sq = 0.0
    numel = 0
    for p in params:
        if p.grad is not None:
            sq += float(torch.sum(p.grad.detach().float() ** 2).item())
            numel += p.grad.numel()
    grad_norm = math.sqrt(max(sq, 1e-32))
    if numel <= 0:
        return 0.0
    std = noise_rel * grad_norm / math.sqrt(float(numel))
    for p in params:
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * std)
    return float(noise_rel)


def run_variant(
    args: argparse.Namespace,
    variant: VariantSpec,
    preprocess,
    few_dataset,
    full_dataset,
    out_dir: Path,
    run_name: str,
    *,
    budget_schedule: Dict[str, object],
    baseline_total_updates: int,
    baseline_total_examples: int,
) -> Tuple[List[Dict[str, object]], TensorState, TensorState]:
    set_random_seed(args.seed)
    clip_model, _ = load_clip_model(args.backbone, args.device)
    clip_model = clip_model.to(args.device)
    clip_model.eval()
    mark_as_trainable(clip_model, args)
    params = [(name, p) for name, p in clip_model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in params]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    optimizer = SGD(trainable_params, lr=args.lr, momentum=variant.momentum)
    amp_enabled = str(args.device).startswith("cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled)

    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)

    few_size = len(few_dataset.train_x)
    full_size = len(full_dataset.train_x)
    test_size = len(full_dataset.test)

    train_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=variant.batch_size,
        preprocess=preprocess,
        is_train=True,
        shuffle=(variant.batch_size < few_size),
        num_workers=args.num_workers,
    )
    few_eval_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=max(1, few_size),
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

    updates_per_epoch = int(len(train_loader))
    init_state = get_trainable_state(clip_model)
    ckpt_dir = out_dir / "checkpoints" / run_name / budget_schedule["protocol"] / variant.name

    rows: List[Dict[str, object]] = []

    def eval_and_save(
        *,
        epoch: int,
        global_update: int,
        examples_seen: int,
        state: MutableMapping[str, torch.Tensor],
    ) -> None:
        set_trainable_state(clip_model, state)
        few_loss, few_acc = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, 100.0, args.device)
        full_loss, full_acc = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, 100.0, args.device)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, 100.0, args.device)

        full_grad_state, full_grad_norm = compute_dataset_gradient(
            clip_model, full_eval_loader, class_tokens, 100.0, params, args.device
        )
        few_grad_state, few_grad_norm = compute_dataset_gradient(
            clip_model, few_eval_loader, class_tokens, 100.0, params, args.device
        )
        full_grad_dir = normalize_state_direction(full_grad_state)
        rand_dir = build_random_direction_like(state, seed=args.seed * 1000 + 17 * (global_update + 1) + len(variant.name))
        full_curv_grad, fd_eps = finite_difference_curvature(
            clip_model,
            state,
            full_grad_dir,
            full_eval_loader,
            class_tokens,
            100.0,
            args.device,
            full_loss,
            args.curvature_rel_radius,
        )
        full_curv_rand, _ = finite_difference_curvature(
            clip_model,
            state,
            rand_dir,
            full_eval_loader,
            class_tokens,
            100.0,
            args.device,
            full_loss,
            args.curvature_rel_radius,
        )

        batch_noise_ratio, support_full_grad_norm = estimate_batch_noise_ratio(
            clip_model,
            few_dataset.train_x,
            variant.batch_size,
            preprocess,
            class_tokens,
            100.0,
            params,
            args.device,
            args.num_workers,
        )
        injected_noise_ratio = float(variant.noise_rel)
        total_noise_ratio = math.sqrt(batch_noise_ratio * batch_noise_ratio + injected_noise_ratio * injected_noise_ratio)
        drift_rel = _state_norm(_state_sub(state, init_state)) / max(_state_norm(init_state), 1e-32)
        epochs_completed = float(global_update) / float(max(updates_per_epoch, 1))
        example_budget_ratio = float(examples_seen) / float(max(baseline_total_examples, 1))
        if budget_schedule["protocol"] == "update_matched":
            update_budget_ratio = float(global_update) / float(max(int(budget_schedule["target_total_updates"]), 1))
            ckpt_path = ckpt_dir / f"update_{global_update:03d}_trainable.pt"
            checkpoint_axis_value = global_update
            checkpoint_axis = "update"
        else:
            update_budget_ratio = float(global_update) / float(max(baseline_total_updates, 1))
            ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_trainable.pt"
            checkpoint_axis_value = epoch
            checkpoint_axis = "epoch"

        save_trainable_checkpoint(
            ckpt_path,
            state,
            {
                "epoch": int(epoch),
                "global_update": int(global_update),
                "epochs_completed": float(epochs_completed),
                "examples_seen": int(examples_seen),
                "budget_protocol": budget_schedule["protocol"],
                "checkpoint_axis": checkpoint_axis,
                "checkpoint_axis_value": int(checkpoint_axis_value),
                "variant": variant.name,
                "label": variant.label,
                "dataset": args.dataset,
                "shots": args.shots,
                "backbone": args.backbone,
                "position": args.position,
                "encoder": args.encoder,
                "train_batch_size": int(variant.batch_size),
                "momentum": float(variant.momentum),
                "noise_rel": float(variant.noise_rel),
            },
        )
        rows.append(
            {
                "budget_protocol": budget_schedule["protocol"],
                "variant": variant.name,
                "variant_label": variant.label,
                "epoch": int(epoch),
                "global_update": int(global_update),
                "epochs_completed": float(epochs_completed),
                "few_size": int(few_size),
                "full_train_size": int(full_size),
                "test_size": int(test_size),
                "train_batch_size": int(variant.batch_size),
                "updates_per_epoch": int(updates_per_epoch),
                "examples_seen": int(examples_seen),
                "example_budget_ratio": float(example_budget_ratio),
                "update_budget_ratio": float(update_budget_ratio),
                "checkpoint_axis": checkpoint_axis,
                "checkpoint_axis_value": int(checkpoint_axis_value),
                "momentum": float(variant.momentum),
                "noise_rel": float(variant.noise_rel),
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "few_grad_norm": float(few_grad_norm),
                "full_grad_norm": float(full_grad_norm),
                "support_fullbatch_grad_norm": float(support_full_grad_norm),
                "batch_noise_ratio": float(batch_noise_ratio),
                "injected_noise_ratio": float(injected_noise_ratio),
                "total_noise_ratio": float(total_noise_ratio),
                "full_curv_grad": float(full_curv_grad),
                "full_curv_rand": float(full_curv_rand),
                "fd_epsilon": float(fd_eps),
                "drift_rel": float(drift_rel),
                "checkpoint_path": str(ckpt_path),
            }
        )
        print(
            f"[ExperimentJ][{budget_schedule['protocol']}][{variant.name}] "
            f"epoch={epoch:02d} upd={global_update:03d} | full_loss={full_loss:.4f} | "
            f"test_acc={test_acc:.2f} | curv_full_grad={full_curv_grad:.3e} | noise={total_noise_ratio:.3f}"
        )

    eval_and_save(epoch=0, global_update=0, examples_seen=0, state=init_state)

    global_update = 0
    examples_seen = 0
    if budget_schedule["protocol"] == "epoch_matched":
        for epoch in range(1, args.max_epochs + 1):
            clip_model.train()
            for images, target in train_loader:
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=amp_enabled):
                    image_features = clip_model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(class_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * (image_features @ text_features.t())
                    loss = F.cross_entropy(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                add_gradient_noise_(trainable_params, variant.noise_rel)
                scaler.step(optimizer)
                scaler.update()
                global_update += 1
                examples_seen += int(target.size(0))

            if epoch in budget_schedule["checkpoint_epochs"]:
                eval_and_save(epoch=epoch, global_update=global_update, examples_seen=examples_seen, state=get_trainable_state(clip_model))
    else:
        target_total_updates = int(budget_schedule["target_total_updates"])
        checkpoint_updates = set(int(x) for x in budget_schedule["checkpoint_updates"])
        epoch = 0
        while global_update < target_total_updates:
            epoch += 1
            clip_model.train()
            for images, target in train_loader:
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=amp_enabled):
                    image_features = clip_model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(class_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * (image_features @ text_features.t())
                    loss = F.cross_entropy(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                add_gradient_noise_(trainable_params, variant.noise_rel)
                scaler.step(optimizer)
                scaler.update()
                global_update += 1
                examples_seen += int(target.size(0))

                if global_update in checkpoint_updates:
                    eval_and_save(epoch=epoch, global_update=global_update, examples_seen=examples_seen, state=get_trainable_state(clip_model))
                if global_update >= target_total_updates:
                    break

    final_state = get_trainable_state(clip_model)
    return rows, init_state, final_state


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets once so every variant uses the exact same support set.
    clip_model_probe, preprocess = load_clip_model(args.backbone, args.device)
    del clip_model_probe
    torch.cuda.empty_cache() if str(args.device).startswith("cuda") else None

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    support_size = len(few_dataset.train_x)
    variants = build_default_variants(args, support_size)
    baseline_variant = variants[0]
    baseline_updates_per_epoch = int(math.ceil(float(support_size) / float(max(baseline_variant.batch_size, 1))))
    baseline_total_updates = int(baseline_updates_per_epoch * args.max_epochs)
    baseline_total_examples = int(support_size * baseline_total_updates)
    budget_schedule = build_budget_schedule(args, baseline_total_updates=baseline_total_updates)

    if args.run_name is None:
        bb = args.backbone.replace("/", "-")
        budget_tag = "budget_epoch" if args.budget_protocol == "epoch_matched" else "budget_update"
        run_name = (
            f"experimentJ_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
            f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_{budget_tag}"
        )
    else:
        run_name = args.run_name

    cfg = {
        "seed": args.seed,
        "device": args.device,
        "root_path": args.root_path,
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "position": args.position,
        "encoder": args.encoder,
        "optimizer": "SGD",
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "full_support_train": bool(args.full_support_train),
        "budget_protocol": args.budget_protocol,
        "checkpoint_epochs": args.checkpoint_epochs,
        "checkpoint_updates": args.checkpoint_updates,
        "target_total_updates": int(args.target_total_updates),
        "resolved_budget_schedule": budget_schedule,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "out_dir": str(out_dir),
        "run_name": run_name,
        "few_size": support_size,
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "curvature_rel_radius": args.curvature_rel_radius,
        "mini_batch_divisors": args.mini_batch_divisors,
        "momentum_value": args.momentum_value,
        "noise_rel_scale": args.noise_rel_scale,
        "interp_points": args.interp_points,
        "baseline_updates_per_epoch": baseline_updates_per_epoch,
        "baseline_total_updates": baseline_total_updates,
        "baseline_total_examples": baseline_total_examples,
        "variants": [
            {
                "name": v.name,
                "label": v.label,
                "batch_size": v.batch_size,
                "momentum": v.momentum,
                "noise_rel": v.noise_rel,
            }
            for v in variants
        ],
        "no_augmentation": True,
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    all_rows: List[Dict[str, object]] = []
    final_states: Dict[str, TensorState] = {}
    init_ref: Optional[TensorState] = None

    t0 = time.perf_counter()
    for variant in variants:
        rows, init_state, final_state = run_variant(
            args,
            variant,
            preprocess,
            few_dataset,
            full_dataset,
            out_dir,
            run_name,
            budget_schedule=budget_schedule,
            baseline_total_updates=baseline_total_updates,
            baseline_total_examples=baseline_total_examples,
        )
        all_rows.extend(rows)
        final_states[variant.name] = final_state
        if init_ref is None:
            init_ref = init_state
    elapsed = time.perf_counter() - t0
    print(f"[ExperimentJ] All variants finished in {elapsed:.1f}s")

    traj_df = sorted(all_rows, key=lambda r: (r["variant"], r["global_update"], r["epoch"]))
    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_df[0].keys()))
        writer.writeheader()
        writer.writerows(traj_df)

    # Final checkpoint summary table.
    final_rows: List[Dict[str, object]] = []
    for variant in variants:
        sub = [r for r in traj_df if r["variant"] == variant.name]
        final_rows.append(sub[-1])
    with open(out_dir / f"{run_name}_variant_final.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    # Connectivity to the baseline final solution.
    baseline_name = variants[0].name
    baseline_state = final_states[baseline_name]
    clip_model_conn, preprocess_conn = load_clip_model(args.backbone, args.device)
    clip_model_conn = clip_model_conn.to(args.device)
    clip_model_conn.eval()
    mark_as_trainable(clip_model_conn, args)
    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)
    few_eval_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=max(1, len(few_dataset.train_x)),
        preprocess=preprocess_conn,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    full_eval_loader = build_loader_from_source(
        full_dataset.train_x,
        batch_size=args.eval_batch_size,
        preprocess=preprocess_conn,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader_from_source(
        full_dataset.test,
        batch_size=args.eval_batch_size,
        preprocess=preprocess_conn,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    connectivity_rows: List[Dict[str, object]] = []
    connectivity_summary: Dict[str, Dict[str, object]] = {}
    alphas = torch.linspace(0.0, 1.0, args.interp_points, dtype=torch.float64)

    for variant in variants[1:]:
        target_state = final_states[variant.name]
        pair_key = f"{baseline_name}__to__{variant.name}"
        pair_rows: List[Dict[str, object]] = []
        for alpha in alphas.tolist():
            interp_state = interpolate_state(baseline_state, target_state, float(alpha))
            set_trainable_state(clip_model_conn, interp_state)
            few_loss, few_acc = compute_loader_loss_acc(clip_model_conn, few_eval_loader, class_tokens, 100.0, args.device)
            full_loss, full_acc = compute_loader_loss_acc(clip_model_conn, full_eval_loader, class_tokens, 100.0, args.device)
            test_loss, test_acc = compute_loader_loss_acc(clip_model_conn, test_loader, class_tokens, 100.0, args.device)
            row = {
                "budget_protocol": args.budget_protocol,
                "pair": pair_key,
                "baseline_variant": baseline_name,
                "target_variant": variant.name,
                "target_label": variant.label,
                "alpha": float(alpha),
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }
            connectivity_rows.append(row)
            pair_rows.append(row)
        few_curve = torch.tensor([r["few_loss"] for r in pair_rows], dtype=torch.float64)
        full_curve = torch.tensor([r["full_loss"] for r in pair_rows], dtype=torch.float64)
        test_curve = torch.tensor([r["test_loss"] for r in pair_rows], dtype=torch.float64)
        connectivity_summary[pair_key] = {
            "budget_protocol": args.budget_protocol,
            "baseline_variant": baseline_name,
            "target_variant": variant.name,
            "target_label": variant.label,
            "few_loss_curve": interpolation_barrier(alphas, few_curve),
            "full_loss_curve": interpolation_barrier(alphas, full_curve),
            "test_loss_curve": interpolation_barrier(alphas, test_curve),
        }

    with open(out_dir / f"{run_name}_connectivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(connectivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(connectivity_rows)

    with open(out_dir / f"{run_name}_connectivity_summary.json", "w") as f:
        json.dump(connectivity_summary, f, indent=2)

    print(f"[ExperimentJ] Wrote logs to: {out_dir}")
    print(f"[ExperimentJ] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

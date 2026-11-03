#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment H — Support-vs-full landscape mismatch.

Goal
----
Understand whether SGD works because it stays in regions where the support and
full-data losses are still geometrically aligned during few-shot CLIP
adaptation.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (but backbone is configurable).
- 1-shot adaptation by default.
- Full-support training: one batch per epoch when --full_support_train is set.
- No augmentation for geometry experiments.
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default.
- At each saved checkpoint, compare:
    1) support-vs-full gradients,
    2) dominant support-vs-full Hessian eigendirections (per encoder),
    3) local 2D support/full loss surfaces in the plane spanned by the support
       gradient and the orthogonal component of the full gradient.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_encoder_alignment.csv
<out_dir>/<run_name>_block_alignment.csv
<out_dir>/<run_name>_surface.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

Companion plotting script:
  loss_landscape_experiment_files/plot_experimentH_support_full_mismatch.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext, contextmanager
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader


def _prepare_torchvision_import() -> None:
    """Import torchvision safely across environments with/without the NMS op."""
    try:
        importlib.import_module("torchvision")
        return
    except RuntimeError as exc:
        if "operator torchvision::nms does not exist" not in str(exc):
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        importlib.import_module("torchvision")


_prepare_torchvision_import()

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root
from loss_landscape_experiment_files.clip_block_utils import build_trainable_meta, group_names_by_block, group_names_by_encoder, subset_state

EPS = 1e-12


@contextmanager
def _math_sdpa_context(device: str):
    """Force the unfused math SDPA backend for higher-order derivatives.

    This helper is intentionally defensive: some environments ignore the newer
    sdpa_kernel() context for nn.MultiheadAttention unless the legacy CUDA SDPA
    flags are also disabled explicitly. We therefore apply both when available.
    """
    dev_type = torch.device(device).type
    if dev_type != "cuda":
        yield
        return

    ctx = nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch >= 2.1
        ctx = sdpa_kernel([SDPBackend.MATH])
    except Exception:
        try:
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            ctx = nullcontext()

    prev = {}
    cuda_bk = getattr(torch.backends, "cuda", None)
    if cuda_bk is not None:
        for nm in ["flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"]:
            fn = getattr(cuda_bk, nm, None)
            if callable(fn):
                try:
                    prev[nm] = fn()
                except Exception:
                    pass
        for nm, val in [("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)]:
            fn = getattr(cuda_bk, nm, None)
            if callable(fn):
                try:
                    fn(val)
                except Exception:
                    pass

    try:
        with ctx:
            yield
    finally:
        if cuda_bk is not None and prev:
            restore = {
                "flash_sdp_enabled": ("enable_flash_sdp", prev.get("flash_sdp_enabled")),
                "mem_efficient_sdp_enabled": ("enable_mem_efficient_sdp", prev.get("mem_efficient_sdp_enabled")),
                "math_sdp_enabled": ("enable_math_sdp", prev.get("math_sdp_enabled")),
            }
            for _, (setter_name, val) in restore.items():
                if val is None:
                    continue
                fn = getattr(cuda_bk, setter_name, None)
                if callable(fn):
                    try:
                        fn(val)
                    except Exception:
                        pass


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
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD"])
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
        help="Comma-separated epochs t whose trainable checkpoints are saved and analysed.",
    )
    parser.add_argument(
        "--surface_epochs",
        type=str,
        default=None,
        help="Comma-separated checkpoint epochs for which 2D surfaces are evaluated. Defaults to --checkpoint_epochs.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="CPU-side dataloader batch size used for full-train / test evaluation.",
    )
    parser.add_argument(
        "--analysis_microbatch_size",
        type=int,
        default=64,
        help=(
            "GPU microbatch size used inside analysis passes (loss / accuracy / gradient). "
            "Large CPU batches are split into chunks of this size before being sent to the GPU."
        ),
    )
    parser.add_argument(
        "--hessian_microbatch_size",
        type=int,
        default=4,
        help=(
            "GPU microbatch size used inside Hessian-vector-product computations. "
            "Lower this first if Experiment H runs out of memory."
        ),
    )
    parser.add_argument(
        "--hessian_power_iters",
        type=int,
        default=6,
        help="Power-iteration steps used for the dominant Hessian eigendirection per encoder and per loss.",
    )
    parser.add_argument(
        "--surface_points",
        type=int,
        default=11,
        help="Number of points per axis for the 2D local loss surfaces.",
    )
    parser.add_argument(
        "--surface_radius_steps",
        type=float,
        default=2.0,
        help="Half-width of the surface grid, measured in units of the encoder's actual SGD step norm.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentH_logs",
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
    if args.surface_epochs is None:
        args.surface_epochs = list(args.checkpoint_epochs)
    else:
        args.surface_epochs = sorted({int(x) for x in args.surface_epochs.split(",") if x.strip()})
    if args.surface_points < 3:
        raise ValueError("--surface_points must be >= 3")
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


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


def _iter_microbatches(images: torch.Tensor, target: torch.Tensor, microbatch_size: int):
    mb = max(1, int(microbatch_size))
    n = int(target.size(0))
    for start in range(0, n, mb):
        end = min(start + mb, n)
        yield images[start:end], target[start:end]


def _maybe_empty_cuda_cache(device: str) -> None:
    if torch.device(device).type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def tokenize_class_texts(classnames: Sequence[str], template: Sequence[str], device: str) -> torch.Tensor:
    prompt_template = template[0]
    texts = [prompt_template.format(name.replace("_", " ")) for name in classnames]
    return clip.tokenize(texts).to(device)


def get_trainable_state(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().clone().float()
    return state


def set_trainable_state(model: torch.nn.Module, state: MutableMapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def save_trainable_checkpoint(path: Path, state: MutableMapping[str, torch.Tensor], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": dict(state), "meta": meta}, path)


def load_trainable_checkpoint(path: Path) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, object]]:
    obj = torch.load(path, map_location="cpu")
    state = OrderedDict((k, v.float()) for k, v in obj["trainable_state"].items())
    meta = dict(obj.get("meta", {}))
    return state, meta


def state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    return sum(float((a[k].double() * b[k].double()).sum().item()) for k in a.keys())


def state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return state_dot(a, a)


def state_norm(a: MutableMapping[str, torch.Tensor]) -> float:
    return math.sqrt(max(state_sqnorm(a), 0.0))


def state_add(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor], beta: float = 1.0) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    b_keys = set(b.keys())
    for k, v in a.items():
        if k in b_keys:
            out[k] = v + beta * b[k]
        else:
            out[k] = v.clone()
    for k, v in b.items():
        if k not in out:
            out[k] = beta * v
    return out


def state_sub(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out


def state_scale(a: MutableMapping[str, torch.Tensor], alpha: float) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, alpha * v) for k, v in a.items())


def state_cosine(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    denom = state_norm(a) * state_norm(b)
    if denom <= EPS:
        return float("nan")
    return state_dot(a, b) / denom


def normalize_state(a: MutableMapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    nrm = state_norm(a)
    if nrm <= EPS:
        return OrderedDict((k, torch.zeros_like(v)) for k, v in a.items())
    return state_scale(a, 1.0 / nrm)


def orthogonalize_state(vec: MutableMapping[str, torch.Tensor], basis: MutableMapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    coeff = state_dot(vec, basis)
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in vec.keys():
        out[k] = vec[k] - coeff * basis[k]
    return out


def random_like_state(names_to_params: MutableMapping[str, torch.nn.Parameter], seed: int) -> "OrderedDict[str, torch.Tensor]":
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in names_to_params.items():
        out[name] = torch.randn(p.shape, generator=gen, dtype=torch.float32)
    return out


def combine_basis(d1: MutableMapping[str, torch.Tensor], d2: MutableMapping[str, torch.Tensor], a: float, b: float) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    keys = list(d1.keys())
    for k in keys:
        out[k] = a * d1[k] + b * d2[k]
    return out


def sign_agreement(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    same = 0
    total = 0
    for k in a.keys():
        x = a[k].reshape(-1)
        y = b[k].reshape(-1)
        mask = (x != 0) | (y != 0)
        if mask.any():
            total += int(mask.sum().item())
            same += int((torch.sign(x[mask]) == torch.sign(y[mask])).sum().item())
    if total == 0:
        return float("nan")
    return float(same) / float(total)


def compute_loader_loss_acc(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    microbatch_size: int = 64,
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
            for img_mb, tgt_mb in _iter_microbatches(images, target, microbatch_size):
                img_mb = img_mb.to(device, non_blocking=True)
                tgt_mb = tgt_mb.to(device, non_blocking=True)
                image_features = model.encode_image(img_mb)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, tgt_mb, reduction="mean")
                pred = logits.argmax(dim=1)
                total_loss += float(loss.item()) * int(tgt_mb.size(0))
                total_correct += int((pred == tgt_mb).sum().item())
                total_items += int(tgt_mb.size(0))
                del logits, loss, pred, image_features, img_mb, tgt_mb
    model.train(was_training)
    _maybe_empty_cuda_cache(device)
    if total_items == 0:
        return float("nan"), float("nan")
    return total_loss / total_items, 100.0 * total_correct / total_items


def compute_dataset_gradient(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params_map: MutableMapping[str, torch.nn.Parameter],
    device: str,
    microbatch_size: int = 64,
) -> Tuple["OrderedDict[str, torch.Tensor]", float, float]:
    was_training = model.training
    model.eval()
    total_items = len(loader.dataset)
    if total_items <= 0:
        raise RuntimeError("Gradient loader is empty.")
    grad_acc: "OrderedDict[str, torch.Tensor]" = OrderedDict((n, torch.zeros_like(p, dtype=torch.float32, device="cpu")) for n, p in params_map.items())
    total_loss = 0.0
    total_correct = 0
    params = list(params_map.values())
    names = list(params_map.keys())
    for images, target in loader:
        for img_mb, tgt_mb in _iter_microbatches(images, target, microbatch_size):
            bs = int(tgt_mb.size(0))
            weight = float(bs) / float(total_items)
            img_mb = img_mb.to(device, non_blocking=True)
            tgt_mb = tgt_mb.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = model.encode_image(img_mb)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, tgt_mb, reduction="mean")
            grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
            for name, g in zip(names, grads):
                grad_acc[name].add_(g.detach().float().cpu(), alpha=weight)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == tgt_mb).sum().item())
            del grads, logits, loss, image_features, text_features, img_mb, tgt_mb
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    _maybe_empty_cuda_cache(device)
    return grad_acc, total_loss / float(total_items), 100.0 * total_correct / float(total_items)


def compute_dataset_hvp(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params_map: MutableMapping[str, torch.nn.Parameter],
    vec_state: MutableMapping[str, torch.Tensor],
    device: str,
    microbatch_size: int = 4,
) -> "OrderedDict[str, torch.Tensor]":
    was_training = model.training
    model.eval()
    total_items = len(loader.dataset)
    if total_items <= 0:
        raise RuntimeError("Hessian loader is empty.")
    params = list(params_map.values())
    names = list(params_map.keys())
    hvp_acc: "OrderedDict[str, torch.Tensor]" = OrderedDict((n, torch.zeros_like(p, dtype=torch.float32, device="cpu")) for n, p in params_map.items())
    vec_dev = {n: vec_state[n].to(device=device, dtype=params_map[n].dtype) for n in names}
    for images, target in loader:
        for img_mb, tgt_mb in _iter_microbatches(images, target, microbatch_size):
            bs = int(tgt_mb.size(0))
            weight = float(bs) / float(total_items)
            img_mb = img_mb.to(device, non_blocking=True)
            tgt_mb = tgt_mb.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with _math_sdpa_context(device):
                text_features = model.encode_text(class_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features = model.encode_image(img_mb)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, tgt_mb, reduction="mean")
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                dot = 0.0
                for n, g in zip(names, grads):
                    dot = dot + torch.sum(g * vec_dev[n])
                hvps = torch.autograd.grad(dot, params, retain_graph=False, create_graph=False)
            for name, hv in zip(names, hvps):
                hvp_acc[name].add_(hv.detach().float().cpu(), alpha=weight)
            del grads, hvps, dot, logits, loss, image_features, text_features, img_mb, tgt_mb
            _maybe_empty_cuda_cache(device)
    del vec_dev
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    _maybe_empty_cuda_cache(device)
    return hvp_acc


def power_iteration_top_hessian(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params_map: MutableMapping[str, torch.nn.Parameter],
    device: str,
    num_iters: int,
    seed: int,
    microbatch_size: int = 4,
) -> Tuple[float, "OrderedDict[str, torch.Tensor]", float]:
    vec = normalize_state(random_like_state(params_map, seed=seed))
    eigval = 0.0
    for _ in range(num_iters):
        hv = compute_dataset_hvp(model, loader, class_tokens, logit_scale, params_map, vec, device, microbatch_size=microbatch_size)
        hv_norm = state_norm(hv)
        if hv_norm <= EPS:
            return 0.0, normalize_state(hv), 0.0
        eigval = state_dot(vec, hv)
        vec = normalize_state(hv)
    hv = compute_dataset_hvp(model, loader, class_tokens, logit_scale, params_map, vec, device, microbatch_size=microbatch_size)
    eigval = state_dot(vec, hv)
    # residual ||Hv - lambda v|| / ||Hv||
    res_vec = state_add(hv, state_scale(vec, -eigval), beta=1.0)
    residual = state_norm(res_vec) / max(state_norm(hv), EPS)
    return float(eigval), vec, float(residual)


def directional_curvature(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params_map: MutableMapping[str, torch.nn.Parameter],
    direction: MutableMapping[str, torch.Tensor],
    device: str,
    microbatch_size: int = 4,
) -> float:
    d = normalize_state(direction)
    if state_norm(d) <= EPS:
        return float("nan")
    hv = compute_dataset_hvp(model, loader, class_tokens, logit_scale, params_map, d, device, microbatch_size=microbatch_size)
    return state_dot(d, hv)


def get_step_scale_for_surface(trajectory_df, epoch: int, encoder: str) -> float:
    col = f"{encoder}_step_l2"
    epochs = set(int(x) for x in trajectory_df["epoch"].tolist())
    # Prefer the outgoing step from t -> t+1 when available.
    if (epoch + 1) in epochs:
        row = trajectory_df.loc[trajectory_df["epoch"] == (epoch + 1)].iloc[0]
        val = float(row[col])
        if val > 0:
            return val
    # Otherwise fall back to the incoming step to epoch t.
    if epoch in epochs:
        row = trajectory_df.loc[trajectory_df["epoch"] == epoch].iloc[0]
        val = float(row[col])
        if val > 0:
            return val
    positives = trajectory_df[col].to_numpy(dtype=float)
    positives = positives[positives > 0]
    if positives.size == 0:
        return 1.0
    return float(np.median(positives))


def make_surface_basis(
    g_few: MutableMapping[str, torch.Tensor],
    g_full: MutableMapping[str, torch.Tensor],
    fallback: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> Tuple["OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]", float, float]:
    e1 = normalize_state(g_few)
    full_hat = normalize_state(g_full)
    proj = state_dot(full_hat, e1)
    orth = orthogonalize_state(full_hat, e1)
    orth_n = state_norm(orth)
    if orth_n <= 1e-8:
        if fallback is None:
            raise RuntimeError("Could not build a non-degenerate 2D basis; full gradient is collinear with the support gradient and no fallback direction was provided.")
        orth = orthogonalize_state(normalize_state(fallback), e1)
        orth_n = state_norm(orth)
        if orth_n <= 1e-8:
            raise RuntimeError("Fallback direction is also collinear with the support gradient.")
    e2 = state_scale(orth, 1.0 / orth_n)
    return e1, e2, float(proj), float(orth_n)


def evaluate_surface_for_encoder(
    model,
    base_state: MutableMapping[str, torch.Tensor],
    encoder_names: Sequence[str],
    d1: MutableMapping[str, torch.Tensor],
    d2: MutableMapping[str, torch.Tensor],
    scale: float,
    surface_alphas: Sequence[float],
    few_loader,
    full_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    epoch: int,
    encoder: str,
    grad_proj_x: float,
    grad_proj_y: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for beta in surface_alphas:
        for alpha in surface_alphas:
            delta = combine_basis(d1, d2, float(alpha) * scale, float(beta) * scale)
            pert_state = OrderedDict((k, v.clone()) for k, v in base_state.items())
            for name in encoder_names:
                pert_state[name] = pert_state[name] + delta[name]
            set_trainable_state(model, pert_state)
            few_loss, few_acc = compute_loader_loss_acc(model, few_loader, class_tokens, logit_scale, device, microbatch_size=64)
            full_loss, full_acc = compute_loader_loss_acc(model, full_loader, class_tokens, logit_scale, device, microbatch_size=64)
            rows.append(
                {
                    "epoch": int(epoch),
                    "encoder": encoder,
                    "alpha_units": float(alpha),
                    "beta_units": float(beta),
                    "alpha_abs": float(alpha) * float(scale),
                    "beta_abs": float(beta) * float(scale),
                    "step_scale": float(scale),
                    "few_loss": float(few_loss),
                    "few_acc": float(few_acc),
                    "full_loss": float(full_loss),
                    "full_acc": float(full_acc),
                    "support_arrow_x": 1.0,
                    "support_arrow_y": 0.0,
                    "full_arrow_x": float(grad_proj_x),
                    "full_arrow_y": float(grad_proj_y),
                }
            )
    set_trainable_state(model, base_state)
    return rows


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = load_clip_model(args.backbone, args.device)
    clip_model = clip_model.float()  # full precision for second-order geometry diagnostics
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    if args.run_name is None:
        bb = args.backbone.replace("/", "-")
        run_name = (
            f"experimentH_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
            f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
        )
    else:
        run_name = args.run_name

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

    clip_model = clip_model.to(args.device)
    mark_as_trainable(clip_model, args)
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")
    param_map_model = OrderedDict((name, p) for name, p in clip_model.named_parameters() if p.requires_grad)
    meta = build_trainable_meta(clip_model)
    encoder_groups = {k: v for k, v in group_names_by_encoder(meta).items() if k in {"vision", "text"}}
    block_groups = {(enc, blk): names for (enc, blk), names in group_names_by_block(meta).items() if enc in {"vision", "text"} and blk >= 0}

    optimizer = SGD(trainable_params, lr=args.lr)

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
        "surface_epochs": args.surface_epochs,
        "eval_batch_size": args.eval_batch_size,
        "analysis_microbatch_size": args.analysis_microbatch_size,
        "hessian_microbatch_size": args.hessian_microbatch_size,
        "hessian_power_iters": args.hessian_power_iters,
        "surface_points": args.surface_points,
        "surface_radius_steps": args.surface_radius_steps,
        "num_workers": args.num_workers,
        "out_dir": str(out_dir),
        "run_name": run_name,
        "few_size": len(few_dataset.train_x),
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "logit_scale": logit_scale,
        "no_augmentation": True,
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    trajectory_rows: List[Dict[str, object]] = []
    init_state = get_trainable_state(clip_model)
    save_trainable_checkpoint(
        ckpt_dir / "epoch_00_trainable.pt",
        init_state,
        {
            "epoch": 0,
            "dataset": args.dataset,
            "shots": args.shots,
            "backbone": args.backbone,
            "optimizer": args.optimizer,
            "position": args.position,
            "encoder": args.encoder,
        },
    )
    trajectory_rows.append({
        "epoch": 0,
        "checkpoint_saved": 1,
        "all_step_l2": 0.0,
        "vision_step_l2": 0.0,
        "text_step_l2": 0.0,
    })

    start_time = time.perf_counter()
    clip_model.train()
    for epoch in range(1, args.max_epochs + 1):
        prev_state = get_trainable_state(clip_model)
        for images, target in few_train_loader:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            text_features = clip_model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        curr_state = get_trainable_state(clip_model)
        delta = state_sub(curr_state, prev_state)
        row = {
            "epoch": int(epoch),
            "checkpoint_saved": int(epoch in args.checkpoint_epochs),
            "all_step_l2": float(state_norm(delta)),
        }
        for enc in ["vision", "text"]:
            if enc in encoder_groups:
                row[f"{enc}_step_l2"] = float(state_norm(subset_state(delta, encoder_groups[enc])))
            else:
                row[f"{enc}_step_l2"] = float("nan")
        trajectory_rows.append(row)
        if epoch in args.checkpoint_epochs:
            save_trainable_checkpoint(
                ckpt_dir / f"epoch_{epoch:02d}_trainable.pt",
                curr_state,
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
        print(f"[ExperimentH] finished epoch {epoch:02d} | all_step_l2={row['all_step_l2']:.4e}")
    elapsed_train = time.perf_counter() - start_time
    print(f"[ExperimentH] Training finished in {elapsed_train:.1f}s")

    # Analyse saved checkpoints.
    trajectory_df = None
    import pandas as pd
    trajectory_df = pd.DataFrame(trajectory_rows)

    encoder_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    surface_rows: List[Dict[str, object]] = []

    for epoch in args.checkpoint_epochs:
        state, meta_ckpt = load_trainable_checkpoint(ckpt_dir / f"epoch_{epoch:02d}_trainable.pt")
        set_trainable_state(clip_model, state)
        few_loss, few_acc = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        full_loss, full_acc = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        print(f"[ExperimentH] checkpoint t={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | test_loss={test_loss:.4f}")

        grad_few_all, _, _ = compute_dataset_gradient(clip_model, few_eval_loader, class_tokens, logit_scale, param_map_model, args.device, microbatch_size=args.analysis_microbatch_size)
        grad_full_all, _, _ = compute_dataset_gradient(clip_model, full_eval_loader, class_tokens, logit_scale, param_map_model, args.device, microbatch_size=args.analysis_microbatch_size)

        for enc in ["vision", "text"]:
            if enc not in encoder_groups:
                continue
            names = encoder_groups[enc]
            params_enc = OrderedDict((n, param_map_model[n]) for n in names)
            g_few = subset_state(grad_few_all, names)
            g_full = subset_state(grad_full_all, names)
            few_grad_l2 = state_norm(g_few)
            full_grad_l2 = state_norm(g_full)
            grad_cos = state_cosine(g_few, g_full)
            grad_sign = sign_agreement(g_few, g_full)

            eig_few, vec_few, res_few = power_iteration_top_hessian(
                clip_model,
                few_eval_loader,
                class_tokens,
                logit_scale,
                params_enc,
                args.device,
                num_iters=args.hessian_power_iters,
                seed=1000 + 17 * epoch + (0 if enc == "vision" else 1),
                microbatch_size=args.hessian_microbatch_size,
            )
            eig_full, vec_full, res_full = power_iteration_top_hessian(
                clip_model,
                full_eval_loader,
                class_tokens,
                logit_scale,
                params_enc,
                args.device,
                num_iters=args.hessian_power_iters,
                seed=2000 + 17 * epoch + (0 if enc == "vision" else 1),
                microbatch_size=args.hessian_microbatch_size,
            )
            eigvec_abs_cos = abs(state_cosine(vec_few, vec_full))
            curv_few_on_gfew = directional_curvature(clip_model, few_eval_loader, class_tokens, logit_scale, params_enc, g_few, args.device, microbatch_size=args.hessian_microbatch_size)
            curv_full_on_gfew = directional_curvature(clip_model, full_eval_loader, class_tokens, logit_scale, params_enc, g_few, args.device, microbatch_size=args.hessian_microbatch_size)
            curv_few_on_gfull = directional_curvature(clip_model, few_eval_loader, class_tokens, logit_scale, params_enc, g_full, args.device, microbatch_size=args.hessian_microbatch_size)
            curv_full_on_gfull = directional_curvature(clip_model, full_eval_loader, class_tokens, logit_scale, params_enc, g_full, args.device, microbatch_size=args.hessian_microbatch_size)

            step_scale = get_step_scale_for_surface(trajectory_df, epoch, enc)

            e1, e2, full_proj_x, full_proj_y = make_surface_basis(g_few, g_full, fallback=vec_full)

            encoder_rows.append({
                "epoch": int(epoch),
                "encoder": enc,
                "numel": int(sum(int(param_map_model[n].numel()) for n in names)),
                "few_size": len(few_dataset.train_x),
                "full_train_size": len(full_dataset.train_x),
                "test_size": len(full_dataset.test),
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "step_scale": float(step_scale),
                "grad_cosine": float(grad_cos),
                "grad_sign_agreement": float(grad_sign),
                "few_grad_l2": float(few_grad_l2),
                "full_grad_l2": float(full_grad_l2),
                "grad_norm_ratio": float(few_grad_l2 / max(full_grad_l2, EPS)),
                "hessian_top_eig_few": float(eig_few),
                "hessian_top_eig_full": float(eig_full),
                "hessian_eigvec_abs_cosine": float(eigvec_abs_cos),
                "hessian_residual_few": float(res_few),
                "hessian_residual_full": float(res_full),
                "curv_few_on_gfew": float(curv_few_on_gfew),
                "curv_full_on_gfew": float(curv_full_on_gfew),
                "curv_few_on_gfull": float(curv_few_on_gfull),
                "curv_full_on_gfull": float(curv_full_on_gfull),
                "surface_full_grad_proj_x": float(full_proj_x),
                "surface_full_grad_proj_y": float(full_proj_y),
            })

            if epoch in args.surface_epochs:
                surface_axis = torch.linspace(-args.surface_radius_steps, args.surface_radius_steps, args.surface_points).tolist()
                surface_rows.extend(
                    evaluate_surface_for_encoder(
                        clip_model,
                        state,
                        names,
                        e1,
                        e2,
                        step_scale,
                        surface_axis,
                        few_eval_loader,
                        full_eval_loader,
                        class_tokens,
                        logit_scale,
                        args.device,
                        epoch,
                        enc,
                        full_proj_x,
                        full_proj_y,
                    )
                )

        for (enc, blk), names in sorted(block_groups.items(), key=lambda x: (x[0][0], x[0][1])):
            g_few_blk = subset_state(grad_few_all, names)
            g_full_blk = subset_state(grad_full_all, names)
            few_grad_l2 = state_norm(g_few_blk)
            full_grad_l2 = state_norm(g_full_blk)
            block_rows.append({
                "epoch": int(epoch),
                "encoder": enc,
                "block": int(blk),
                "numel": int(sum(int(param_map_model[n].numel()) for n in names)),
                "few_loss": float(few_loss),
                "full_loss": float(full_loss),
                "test_loss": float(test_loss),
                "grad_cosine": float(state_cosine(g_few_blk, g_full_blk)),
                "grad_sign_agreement": float(sign_agreement(g_few_blk, g_full_blk)),
                "few_grad_l2": float(few_grad_l2),
                "full_grad_l2": float(full_grad_l2),
                "grad_norm_ratio": float(few_grad_l2 / max(full_grad_l2, EPS)),
            })

    # Write CSV bundles.
    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    if encoder_rows:
        with open(out_dir / f"{run_name}_encoder_alignment.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(encoder_rows[0].keys()))
            writer.writeheader()
            writer.writerows(encoder_rows)

    if block_rows:
        with open(out_dir / f"{run_name}_block_alignment.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
            writer.writeheader()
            writer.writerows(block_rows)

    if surface_rows:
        with open(out_dir / f"{run_name}_surface.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(surface_rows[0].keys()))
            writer.writeheader()
            writer.writerows(surface_rows)

    # Summary.
    import pandas as pd
    enc_df = pd.DataFrame(encoder_rows)
    summary = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "checkpoint_epochs": args.checkpoint_epochs,
        "surface_epochs": args.surface_epochs,
        "train_time_seconds": elapsed_train,
        "encoder_summary": {},
    }
    if not enc_df.empty:
        for enc in sorted(enc_df["encoder"].unique().tolist()):
            sub = enc_df[enc_df["encoder"] == enc].sort_values("epoch")
            summary["encoder_summary"][enc] = {
                "grad_cosine_start": float(sub["grad_cosine"].iloc[0]),
                "grad_cosine_end": float(sub["grad_cosine"].iloc[-1]),
                "hessian_eigvec_abs_cosine_start": float(sub["hessian_eigvec_abs_cosine"].iloc[0]),
                "hessian_eigvec_abs_cosine_end": float(sub["hessian_eigvec_abs_cosine"].iloc[-1]),
                "grad_norm_ratio_start": float(sub["grad_norm_ratio"].iloc[0]),
                "grad_norm_ratio_end": float(sub["grad_norm_ratio"].iloc[-1]),
            }
    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ExperimentH] Wrote logs to: {out_dir}")
    print(f"[ExperimentH] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

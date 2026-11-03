#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment E — Directional curvature decomposition on the SGD trajectory.

Goal
----
Understand which directions are actually dangerous during few-shot CLIP
adaptation by measuring directional curvature

    kappa(d) = d^T H d

at checkpoints t in {0,1,2,5,10,20,30}.

Protocol implemented here
-------------------------
- SGD trajectory (the experiment is defined for SGD).
- CLIP ViT-B/16 by default (configurable, but defaults match the paper plan).
- 1-shot adaptation by default.
- Full-support training when --full_support_train is set (one batch per epoch).
- No augmentation for the geometry experiment.
- Train both encoders, all selected blocks trainable.
- Representative datasets: EuroSAT / Oxford-Pets / DTD (user chooses via --dataset).
- Checkpoints are saved at the requested epochs and analysed after training.

Directions analysed
-------------------
1) sgd_step        : the actual SGD step direction at the checkpoint.
2) few_grad        : normalized support-gradient direction.
3) full_grad       : normalized full-train gradient direction.
4) top_eig_full    : approximate top Hessian eigen-direction of the full-train loss.
5) random_i        : random normalized probe directions.

Under the exact default protocol here (vanilla SGD, no momentum, one batch per
epoch), sgd_step and few_grad are identical by construction. We still log both so
that the output matches the experimental design exactly.

Losses whose Hessian is probed
------------------------------
- few  : support / few-shot train loss
- full : full train-split loss
- test : test loss

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_directional_curvature.csv
<out_dir>/<run_name>_block_curvature.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

Companion plotting script
-------------------------
loss_landscape_experiment_files/plot_experimentE_directional_curvature.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import torch
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader


def _import_torchvision_safely() -> None:
    """Make torchvision import robust across environments.

    Some environments are missing the torchvision::nms operator registration at
    import time. We first try a normal import. Only if the specific missing-op
    error appears do we register a dummy schema and retry.
    """
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
        lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        importlib.import_module("torchvision")


_import_torchvision_safely()

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


State = "OrderedDict[str, torch.Tensor]"
ParamItems = List[Tuple[str, torch.nn.Parameter]]
GroupMap = Dict[str, Tuple[str, int]]


def higher_order_sdpa_math_context():
    """Force PyTorch to use the math SDPA kernel for second-order derivatives.

    CLIP uses nn.MultiheadAttention. On recent PyTorch builds this can route to the
    efficient scaled-dot-product attention kernel, but that kernel does not support
    the double backward needed by Hessian-vector products. Experiment E hits that
    path inside curvature_on_loader(...).

    The fix is to use the math SDPA backend only during higher-order sections. This
    keeps normal training/evaluation fast while making HVPs differentiable.
    """
    if not torch.cuda.is_available():
        return nullcontext()

    # Newer PyTorch API.
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        pass

    # Backward-compatible API.
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    except Exception:
        return nullcontext()


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
        help="Experiment E is defined for the SGD trajectory.",
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
        help="Comma-separated checkpoint epochs t.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Loader batch size used for full-train / test evaluation. Actual GPU work is micro-batched by --analysis_microbatch_size.",
    )
    parser.add_argument(
        "--power_iters",
        type=int,
        default=6,
        help="Power-iteration count used to approximate the top full-loss Hessian eigen-direction.",
    )
    parser.add_argument(
        "--analysis_microbatch_size",
        type=int,
        default=2,
        help=(
            "Micro-batch size used inside loss / gradient / curvature evaluation. "
            "This is the main memory-safety knob for local A6000 runs; 2 is the safer default when using the math SDPA kernel for HVPs."
        ),
    )
    parser.add_argument(
        "--empty_cache_between_phases",
        action="store_true",
        help="Call torch.cuda.empty_cache() after expensive gradient / curvature phases to reduce fragmentation.",
    )
    parser.add_argument(
        "--num_random_dirs",
        type=int,
        default=3,
        help="Number of random normalized probe directions.",
    )
    parser.add_argument(
        "--random_seed_offset",
        type=int,
        default=1000,
        help="Offset used when sampling random directions per checkpoint.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentE_logs",
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
    if args.power_iters < 1:
        raise ValueError("--power_iters must be >= 1")
    if args.num_random_dirs < 1:
        raise ValueError("--num_random_dirs must be >= 1")
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


def iterate_microbatches(images: torch.Tensor, target: torch.Tensor, microbatch_size: int):
    if microbatch_size <= 0 or microbatch_size >= int(target.size(0)):
        yield images, target
        return
    for start in range(0, int(target.size(0)), int(microbatch_size)):
        end = min(start + int(microbatch_size), int(target.size(0)))
        yield images[start:end], target[start:end]


def maybe_empty_cache(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_loader_loss_acc(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    *,
    microbatch_size: int,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for images_cpu, target_cpu in loader:
            for images_mb_cpu, target_mb_cpu in iterate_microbatches(images_cpu, target_cpu, microbatch_size):
                images = images_mb_cpu.to(device, non_blocking=True)
                target = target_mb_cpu.to(device, non_blocking=True)
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, target, reduction="mean")
                pred = logits.argmax(dim=1)
                total_loss += float(loss.item()) * int(target.size(0))
                total_correct += int((pred == target).sum().item())
                total_items += int(target.size(0))
                del pred, logits, loss, image_features, images, target
    model.train(was_training)
    if total_items == 0:
        return float("nan"), float("nan")
    return total_loss / total_items, 100.0 * total_correct / total_items


def get_trainable_items(model: torch.nn.Module) -> ParamItems:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def get_trainable_state(model: torch.nn.Module) -> State:
    state: State = OrderedDict()
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


def load_trainable_checkpoint(path: Path) -> Tuple[State, Dict[str, object]]:
    obj = torch.load(path, map_location="cpu")
    state = OrderedDict((k, v) for k, v in obj["trainable_state"].items())
    meta = dict(obj.get("meta", {}))
    return state, meta


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


def infer_group_from_name(param_name: str) -> Tuple[str, int]:
    if param_name.startswith("transformer.resblocks."):
        parts = param_name.split(".")
        return "text", int(parts[2])
    if param_name.startswith("visual.transformer.resblocks."):
        parts = param_name.split(".")
        return "vision", int(parts[3])
    return "other", -1


def build_group_map(param_items: ParamItems) -> GroupMap:
    return {name: infer_group_from_name(name) for name, _ in param_items}


def state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    out = 0.0
    for k in a.keys():
        out += float((a[k].double() * b[k].double()).sum().item())
    return out


def state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return state_dot(a, a)


def normalize_state(a: MutableMapping[str, torch.Tensor], eps: float = 1e-32) -> State:
    norm = math.sqrt(max(state_sqnorm(a), eps))
    out: State = OrderedDict()
    for k, v in a.items():
        out[k] = (v / norm).clone()
    return out


def random_unit_state(param_items: ParamItems, seed: int) -> State:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    out: State = OrderedDict()
    for name, p in param_items:
        out[name] = torch.randn(tuple(p.shape), generator=g, dtype=torch.float32)
    return normalize_state(out)


def zero_state_like(param_items: ParamItems) -> State:
    return OrderedDict((name, torch.zeros_like(p, device="cpu", dtype=torch.float32)) for name, p in param_items)


def clone_state(state: MutableMapping[str, torch.Tensor]) -> State:
    return OrderedDict((k, v.clone()) for k, v in state.items())


def _build_logits(model, images: torch.Tensor, class_tokens: torch.Tensor, logit_scale: float) -> torch.Tensor:
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = model.encode_text(class_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return logit_scale * (image_features @ text_features.t())


def compute_dataset_gradient(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    param_items: ParamItems,
    device: str,
    *,
    microbatch_size: int,
) -> Tuple[float, float, State]:
    was_training = model.training
    model.eval()
    total_items = len(loader.dataset)
    if total_items <= 0:
        raise RuntimeError("Gradient loader is empty.")

    grad_state = zero_state_like(param_items)
    total_loss = 0.0
    total_correct = 0

    params = [p for _, p in param_items]
    for images_cpu, target_cpu in loader:
        for images_mb_cpu, target_mb_cpu in iterate_microbatches(images_cpu, target_cpu, microbatch_size):
            batch_size = int(target_mb_cpu.size(0))
            weight = float(batch_size) / float(total_items)
            model.zero_grad(set_to_none=True)
            images = images_mb_cpu.to(device, non_blocking=True)
            target = target_mb_cpu.to(device, non_blocking=True)
            logits = _build_logits(model, images, class_tokens, logit_scale)
            loss = F.cross_entropy(logits, target, reduction="mean")
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)

            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == target).sum().item())

            for (name, _), g in zip(param_items, grads):
                if g is not None:
                    grad_state[name].add_(g.detach().float().cpu(), alpha=weight)

            del grads, logits, loss, images, target

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return total_loss / float(total_items), 100.0 * total_correct / float(total_items), grad_state


def curvature_on_loader(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    param_items: ParamItems,
    direction_state: MutableMapping[str, torch.Tensor],
    group_map: GroupMap,
    device: str,
    *,
    microbatch_size: int,
    return_hvp_state: bool = False,
) -> Tuple[float, Dict[Tuple[str, int], float], State | None]:
    was_training = model.training
    model.eval()
    total_items = len(loader.dataset)
    if total_items <= 0:
        raise RuntimeError("Curvature loader is empty.")

    params = [p for _, p in param_items]
    block_contrib: DefaultDict[Tuple[str, int], float] = defaultdict(float)
    hvp_state: State | None = zero_state_like(param_items) if return_hvp_state else None
    total_curvature = 0.0

    for images_cpu, target_cpu in loader:
        for images_mb_cpu, target_mb_cpu in iterate_microbatches(images_cpu, target_cpu, microbatch_size):
            batch_size = int(target_mb_cpu.size(0))
            weight = float(batch_size) / float(total_items)
            model.zero_grad(set_to_none=True)
            images = images_mb_cpu.to(device, non_blocking=True)
            target = target_mb_cpu.to(device, non_blocking=True)
            with higher_order_sdpa_math_context():
                logits = _build_logits(model, images, class_tokens, logit_scale)
                loss = F.cross_entropy(logits, target, reduction="mean")
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
                g_dot_d = 0.0
                for (name, _), g in zip(param_items, grads):
                    if g is not None:
                        d = direction_state[name].to(device=device, dtype=torch.float32, non_blocking=True)
                        g_dot_d = g_dot_d + torch.sum(g * d)
                hvps = torch.autograd.grad(g_dot_d, params, create_graph=False, retain_graph=False, allow_unused=True)
            for (name, _), hvp in zip(param_items, hvps):
                if hvp is None:
                    continue
                d = direction_state[name].to(device=device, dtype=torch.float32, non_blocking=True)
                contrib = float(torch.sum(hvp.detach() * d).item())
                total_curvature += weight * contrib
                block_contrib[group_map[name]] += weight * contrib
                if hvp_state is not None:
                    hvp_state[name].add_(hvp.detach().float().cpu(), alpha=weight)
            del hvps, grads, logits, loss, images, target, g_dot_d
            model.zero_grad(set_to_none=True)
    model.train(was_training)
    return total_curvature, dict(block_contrib), hvp_state


def power_iteration_top_eig_full(
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    param_items: ParamItems,
    group_map: GroupMap,
    device: str,
    *,
    num_iters: int,
    seed: int,
    microbatch_size: int,
) -> Tuple[State, float]:
    direction = random_unit_state(param_items, seed=seed)
    rayleigh = float("nan")
    for _ in range(num_iters):
        rayleigh, _, hvp_state = curvature_on_loader(
            model,
            loader,
            class_tokens,
            logit_scale,
            param_items,
            direction,
            group_map,
            device,
            microbatch_size=microbatch_size,
            return_hvp_state=True,
        )
        if hvp_state is None:
            break
        direction = normalize_state(hvp_state)
    rayleigh, _, _ = curvature_on_loader(
        model,
        loader,
        class_tokens,
        logit_scale,
        param_items,
        direction,
        group_map,
        device,
        microbatch_size=microbatch_size,
        return_hvp_state=False,
    )
    return direction, rayleigh


def summarize_direction_name(direction: str) -> Tuple[str, str]:
    if direction.startswith("random_"):
        return "random", direction
    return direction, direction


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    if args.run_name is None:
        bb = args.backbone.replace("/", "-")
        run_name = (
            f"experimentE_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
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
    clip_model.eval()
    mark_as_trainable(clip_model, args)
    param_items = get_trainable_items(clip_model)
    if len(param_items) == 0:
        raise RuntimeError("No trainable parameters were selected.")
    group_map = build_group_map(param_items)

    optimizer = SGD([p for _, p in param_items], lr=args.lr)
    amp_enabled = str(args.device).startswith("cuda")
    scaler = GradScaler("cuda") if amp_enabled else GradScaler("cpu")

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
        "eval_batch_size": args.eval_batch_size,
        "power_iters": args.power_iters,
        "analysis_microbatch_size": args.analysis_microbatch_size,
        "num_random_dirs": args.num_random_dirs,
        "num_workers": args.num_workers,
        "out_dir": str(out_dir),
        "run_name": run_name,
        "few_size": len(few_dataset.train_x),
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "logit_scale": logit_scale,
        "no_augmentation": True,
        "top_eig_direction_loss": "full",
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    trajectory_rows: List[Dict[str, object]] = []

    def eval_and_save(epoch: int, state: MutableMapping[str, torch.Tensor]) -> None:
        set_trainable_state(clip_model, state)
        few_loss, few_acc = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        full_loss, full_acc = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
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
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "checkpoint_path": str(ckpt_path),
            }
        )
        print(
            f"[ExperimentE] epoch={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | "
            f"test_loss={test_loss:.4f} | few_acc={few_acc:.2f} | test_acc={test_acc:.2f}"
        )

    init_state = get_trainable_state(clip_model)
    eval_and_save(0, init_state)

    start_time = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        clip_model.train()
        for images, target in few_train_loader:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                logits = _build_logits(clip_model, images, class_tokens, logit_scale)
                loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        if epoch in args.checkpoint_epochs:
            eval_and_save(epoch, get_trainable_state(clip_model))

    elapsed = time.perf_counter() - start_time
    print(f"[ExperimentE] Training finished in {elapsed:.1f}s")

    # Write trajectory CSV first.
    import csv

    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    # Directional-curvature diagnostics.
    global_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []

    for row in trajectory_rows:
        epoch = int(row["epoch"])
        ckpt_path = Path(row["checkpoint_path"])
        state, _ = load_trainable_checkpoint(ckpt_path)
        set_trainable_state(clip_model, state)
        print(f"[ExperimentE] checkpoint t={epoch:02d}: computing gradients / curvature directions ...")

        few_loss, few_acc, few_grad = compute_dataset_gradient(
            clip_model, few_eval_loader, class_tokens, logit_scale, param_items, args.device,
            microbatch_size=args.analysis_microbatch_size
        )
        if args.empty_cache_between_phases:
            maybe_empty_cache(args.device)
        full_loss, full_acc, full_grad = compute_dataset_gradient(
            clip_model, full_eval_loader, class_tokens, logit_scale, param_items, args.device,
            microbatch_size=args.analysis_microbatch_size
        )
        if args.empty_cache_between_phases:
            maybe_empty_cache(args.device)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device, microbatch_size=args.analysis_microbatch_size)
        if args.empty_cache_between_phases:
            maybe_empty_cache(args.device)

        few_grad_dir = normalize_state(few_grad)
        sgd_step_dir = clone_state(few_grad_dir)  # identical under the exact default protocol here.
        full_grad_dir = normalize_state(full_grad)
        top_eig_dir, top_eig_rayleigh = power_iteration_top_eig_full(
            clip_model,
            full_eval_loader,
            class_tokens,
            logit_scale,
            param_items,
            group_map,
            args.device,
            num_iters=args.power_iters,
            seed=args.seed + epoch,
            microbatch_size=args.analysis_microbatch_size,
        )

        directions: OrderedDict[str, State] = OrderedDict()
        directions["sgd_step"] = sgd_step_dir
        directions["few_grad"] = few_grad_dir
        directions["full_grad"] = full_grad_dir
        directions["top_eig_full"] = top_eig_dir
        for ridx in range(1, args.num_random_dirs + 1):
            directions[f"random_{ridx}"] = random_unit_state(param_items, seed=args.seed + args.random_seed_offset * epoch + ridx)

        loss_specs = [
            ("few", few_eval_loader, float(few_loss), float(few_acc)),
            ("full", full_eval_loader, float(full_loss), float(full_acc)),
            ("test", test_loader, float(test_loss), float(test_acc)),
        ]

        for loss_name, loader, scalar_loss, scalar_acc in loss_specs:
            for direction_name, direction in directions.items():
                curvature, block_contrib, _ = curvature_on_loader(
                    clip_model,
                    loader,
                    class_tokens,
                    logit_scale,
                    param_items,
                    direction,
                    group_map,
                    args.device,
                    microbatch_size=args.analysis_microbatch_size,
                    return_hvp_state=False,
                )
                family, display_key = summarize_direction_name(direction_name)
                global_rows.append(
                    {
                        "epoch": epoch,
                        "loss_name": loss_name,
                        "direction": direction_name,
                        "direction_family": family,
                        "curvature": float(curvature),
                        "scalar_loss": scalar_loss,
                        "scalar_acc": scalar_acc,
                        "few_loss": float(few_loss),
                        "full_loss": float(full_loss),
                        "test_loss": float(test_loss),
                        "few_acc": float(few_acc),
                        "full_acc": float(full_acc),
                        "test_acc": float(test_acc),
                        "top_eig_full_rayleigh": float(top_eig_rayleigh),
                    }
                )
                for (encoder, block), contrib in sorted(block_contrib.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                    if encoder not in {"vision", "text"} or block < 0:
                        continue
                    frac = float("nan")
                    if abs(curvature) > 1e-32:
                        frac = float(contrib / curvature)
                    block_rows.append(
                        {
                            "epoch": epoch,
                            "loss_name": loss_name,
                            "direction": direction_name,
                            "direction_family": family,
                            "encoder": encoder,
                            "block": int(block),
                            "curvature_contrib": float(contrib),
                            "curvature_frac": frac,
                            "global_curvature": float(curvature),
                        }
                    )
                print(
                    f"[ExperimentE] t={epoch:02d} | loss={loss_name:>4s} | dir={direction_name:>11s} | curvature={curvature:+.6e}"
                )
                if args.empty_cache_between_phases:
                    maybe_empty_cache(args.device)

    with open(out_dir / f"{run_name}_directional_curvature.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
        writer.writeheader()
        writer.writerows(global_rows)

    with open(out_dir / f"{run_name}_block_curvature.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    # Compact summary.
    summary = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "checkpoint_epochs": args.checkpoint_epochs,
        "power_iters": args.power_iters,
        "analysis_microbatch_size": args.analysis_microbatch_size,
        "num_random_dirs": args.num_random_dirs,
        "dominant_direction_by_loss": {},
    }
    for loss_name in ["few", "full", "test"]:
        subset = [r for r in global_rows if r["loss_name"] == loss_name]
        if not subset:
            continue
        by_direction: DefaultDict[str, List[float]] = defaultdict(list)
        for r in subset:
            by_direction[r["direction_family"]].append(float(r["curvature"]))
        summary["dominant_direction_by_loss"][loss_name] = {
            k: float(sum(v) / len(v)) for k, v in by_direction.items()
        }

    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ExperimentE] Wrote logs to: {out_dir}")
    print(f"[ExperimentE] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

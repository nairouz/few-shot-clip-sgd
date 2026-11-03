#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment F — Per-block loss landscape and sensitivity map.

Goal
----
Identify which CLIP blocks / encoders are safe to move and which are fragile
under the SGD geometry used in few-shot adaptation.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (configurable).
- 1-shot adaptation by default.
- SGD trajectory only.
- Full-support training when --full_support_train is set (one batch per epoch).
- No augmentation by default (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Checkpoints saved at t in {0,1,2,5,10,20,30} by default.
- At each checkpoint t, compute the full-support few-shot gradient and form the
  actual SGD block step
      Delta_t^(ell) = - lr * g_t^(ell)
  where only the parameters of block ell are kept.
- Probe the local loss landscape
      L(Theta_t + alpha * Delta_t^(ell))
  block by block for alpha in a symmetric grid.
- Measure support / full-train / test losses for each probed point.

This follows the exact protocol recommended for Experiment F:
  "Perturb one block at a time around the SGD checkpoint: Theta_t + alpha d_t^(ell),
   where d_t^(ell) only affects block ell. For each block / encoder, measure local
   loss increase, curvature, safe step radius, and support/full/test mismatch."

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_block_landscape.csv
<out_dir>/<run_name>_block_summary.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

Companion plotting script:
  loss_landscape_experiment_files/plot_experimentF_block_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
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

    Some CPU-only or mismatched torch/torchvision installs raise
    `RuntimeError: operator torchvision::nms does not exist` during import.
    We first try a normal import. Only for that specific error do we define a
    placeholder schema and retry. Normal CUDA builds keep the real operator.
    """
    try:
        importlib.import_module("torchvision")
        return
    except RuntimeError as exc:
        msg = str(exc)
        if "operator torchvision::nms does not exist" not in msg:
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        importlib.import_module("torchvision")


_import_torchvision_safely()

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


BlockKey = Tuple[str, int]  # (encoder, block)


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
        help="Experiment F is defined for the SGD trajectory.",
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
        help="Comma-separated epochs t whose trainable checkpoints are saved and analysed.",
    )
    parser.add_argument("--alpha_min", type=float, default=-2.0)
    parser.add_argument("--alpha_max", type=float, default=2.0)
    parser.add_argument(
        "--alpha_points",
        type=int,
        default=21,
        help="Number of alpha values in [alpha_min, alpha_max]. Must be odd so that alpha=0 is included.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size used for full-train / test loss evaluation.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--safe_tol_rel",
        type=float,
        default=0.01,
        help="Relative loss increase tolerance used for the positive safe radius.",
    )
    parser.add_argument(
        "--safe_tol_abs",
        type=float,
        default=1e-3,
        help="Absolute loss increase tolerance used for the positive safe radius.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentF_logs",
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
    if args.alpha_points < 3 or args.alpha_points % 2 == 0:
        raise ValueError("--alpha_points must be an odd integer >= 3 so that alpha=0 is included.")
    if not (args.alpha_min < 0 < args.alpha_max):
        raise ValueError("Require alpha_min < 0 < alpha_max.")
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
    amp_enabled = str(device).startswith("cuda")
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        with autocast("cuda", enabled=amp_enabled):
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast("cuda", enabled=amp_enabled):
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


def get_trainable_state(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
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


def load_trainable_checkpoint(path: Path) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, object]]:
    obj = torch.load(path, map_location="cpu")
    state = OrderedDict((k, v) for k, v in obj["trainable_state"].items())
    meta = dict(obj.get("meta", {}))
    return state, meta


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


def infer_block_key(param_name: str) -> Optional[BlockKey]:
    if param_name.startswith("visual.transformer.resblocks."):
        parts = param_name.split(".")
        return ("vision", int(parts[3]))
    if param_name.startswith("transformer.resblocks."):
        parts = param_name.split(".")
        return ("text", int(parts[2]))
    return None


def collect_block_to_param_names(model: torch.nn.Module) -> "OrderedDict[BlockKey, List[str]]":
    block_to_names: Dict[BlockKey, List[str]] = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = infer_block_key(name)
        if key is not None:
            block_to_names[key].append(name)
    ordered = OrderedDict()
    for enc in ["vision", "text"]:
        keys = [k for k in block_to_names.keys() if k[0] == enc]
        for key in sorted(keys, key=lambda x: x[1]):
            ordered[key] = block_to_names[key]
    return ordered


def compute_support_step_state(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    lr: float,
) -> "OrderedDict[str, torch.Tensor]":
    amp_enabled = str(device).startswith("cuda")
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast("cuda", enabled=amp_enabled):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()
        break  # full-support loader: one batch by protocol

    step_state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            grad = torch.zeros_like(p) if p.grad is None else p.grad.detach()
            step_state[name] = (-lr * grad).cpu().clone()

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return step_state


def interpolate_at(alpha_grid: np.ndarray, loss_grid: np.ndarray, alpha_query: float) -> float:
    import numpy as np

    return float(np.interp(alpha_query, alpha_grid, loss_grid))


def safe_positive_radius(alpha_grid, loss_grid, *, tol_rel: float, tol_abs: float) -> float:
    import numpy as np

    alpha = np.asarray(alpha_grid, dtype=float)
    loss = np.asarray(loss_grid, dtype=float)
    mask_pos = alpha >= 0.0
    alpha_pos = alpha[mask_pos]
    loss_pos = loss[mask_pos]
    if alpha_pos.size == 0:
        return 0.0
    loss0 = float(loss_pos[alpha_pos == 0.0][0]) if (alpha_pos == 0.0).any() else float(interpolate_at(alpha, loss, 0.0))
    tol = max(float(tol_abs), float(tol_rel) * max(abs(loss0), 1e-12))
    safe_mask = loss_pos <= (loss0 + tol)
    if safe_mask.all():
        return float(alpha_pos[-1])
    idx_first_unsafe = int(np.argmax(~safe_mask))
    if idx_first_unsafe == 0:
        return 0.0
    a0, a1 = float(alpha_pos[idx_first_unsafe - 1]), float(alpha_pos[idx_first_unsafe])
    l0, l1 = float(loss_pos[idx_first_unsafe - 1]), float(loss_pos[idx_first_unsafe])
    if abs(l1 - l0) < 1e-12:
        return a0
    target = loss0 + tol
    frac = (target - l0) / (l1 - l0)
    frac = min(max(frac, 0.0), 1.0)
    return float(a0 + frac * (a1 - a0))


def central_curvature(alpha_grid, loss_grid) -> float:
    import numpy as np

    alpha = np.asarray(alpha_grid, dtype=float)
    loss = np.asarray(loss_grid, dtype=float)
    idx0 = int(np.argmin(np.abs(alpha)))
    if abs(alpha[idx0]) > 1e-12:
        raise ValueError("alpha grid must contain 0")
    if idx0 == 0 or idx0 == len(alpha) - 1:
        return float("nan")
    h_left = abs(alpha[idx0] - alpha[idx0 - 1])
    h_right = abs(alpha[idx0 + 1] - alpha[idx0])
    if abs(h_left - h_right) > 1e-8:
        h = min(h_left, h_right)
        l_neg = interpolate_at(alpha, loss, -h)
        l_pos = interpolate_at(alpha, loss, +h)
        l0 = interpolate_at(alpha, loss, 0.0)
        return float((l_pos - 2.0 * l0 + l_neg) / (h * h))
    h = h_left
    return float((loss[idx0 + 1] - 2.0 * loss[idx0] + loss[idx0 - 1]) / (h * h))


def curve_summary(alpha_grid, loss_grid, *, tol_rel: float, tol_abs: float) -> Dict[str, float]:
    import numpy as np

    alpha = np.asarray(alpha_grid, dtype=float)
    loss = np.asarray(loss_grid, dtype=float)
    idx_min = int(np.argmin(loss))
    loss0 = float(interpolate_at(alpha, loss, 0.0))
    return {
        "loss0": loss0,
        "alpha_min": float(alpha[idx_min]),
        "loss_min": float(loss[idx_min]),
        "delta_at_1": float(interpolate_at(alpha, loss, 1.0) - loss0),
        "curvature0": central_curvature(alpha, loss),
        "safe_radius": safe_positive_radius(alpha, loss, tol_rel=tol_rel, tol_abs=tol_abs),
    }


def make_run_name(args: argparse.Namespace) -> str:
    bb = args.backbone.replace("/", "-")
    return (
        f"experimentF_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
        f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
    )


def run_experiment(args: argparse.Namespace) -> None:
    import numpy as np

    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    run_name = args.run_name if args.run_name is not None else make_run_name(args)
    ckpt_dir = out_dir / "checkpoints" / run_name
    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)

    # Geometry experiment: no augmentation anywhere.
    few_train_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=maybe_full_support_bs(len(few_dataset.train_x), args),
        preprocess=preprocess,
        is_train=True,
        shuffle=not args.full_support_train and args.batch_size > 0,
        num_workers=args.num_workers,
    )
    # Full-support support loader for the actual SGD block step direction.
    few_grad_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=max(1, len(few_dataset.train_x)),
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
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
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    block_to_names = collect_block_to_param_names(clip_model)
    name_to_param = dict(clip_model.named_parameters())
    optimizer = SGD(trainable_params, lr=args.lr)
    scaler = GradScaler("cuda") if str(args.device).startswith("cuda") else GradScaler("cpu")
    amp_enabled = str(args.device).startswith("cuda")

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
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_points": args.alpha_points,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "out_dir": str(out_dir),
        "run_name": run_name,
        "few_size": len(few_dataset.train_x),
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "logit_scale": logit_scale,
        "no_augmentation": True,
        "safe_tol_rel": args.safe_tol_rel,
        "safe_tol_abs": args.safe_tol_abs,
        "blocks": [{"encoder": enc, "block": blk, "num_params": len(names)} for (enc, blk), names in block_to_names.items()],
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    trajectory_rows: List[Dict[str, object]] = []

    def eval_and_save(epoch: int, state: MutableMapping[str, torch.Tensor]) -> None:
        set_trainable_state(clip_model, state)
        few_loss, few_acc = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, logit_scale, args.device)
        full_loss, full_acc = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, logit_scale, args.device)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device)
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
            f"[ExperimentF] epoch={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | "
            f"test_loss={test_loss:.4f} | few_acc={few_acc:.2f} | test_acc={test_acc:.2f}"
        )

    # t = 0
    eval_and_save(0, init_state)

    start_time = time.perf_counter()
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
    print(f"[ExperimentF] Training finished in {elapsed:.1f}s")

    # Save trajectory csv
    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    # Measurement phase.
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points, dtype=np.float64)
    curve_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    traj_by_epoch = {int(r["epoch"]): r for r in trajectory_rows}

    for epoch in args.checkpoint_epochs:
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_trainable.pt"
        state_t, meta = load_trainable_checkpoint(ckpt_path)
        set_trainable_state(clip_model, state_t)

        # Baseline losses at alpha = 0.
        baseline = traj_by_epoch[int(epoch)]
        base_few_loss = float(baseline["few_loss"])
        base_full_loss = float(baseline["full_loss"])
        base_test_loss = float(baseline["test_loss"])

        # Actual SGD block-step direction from the current support gradient.
        step_state = compute_support_step_state(clip_model, few_grad_loader, class_tokens, logit_scale, args.device, args.lr)

        for (enc, blk), names in block_to_names.items():
            base_block = [name_to_param[n].detach().clone() for n in names]
            step_block = [step_state[n].to(device=args.device, dtype=name_to_param[n].dtype) for n in names]
            step_norm_sq = 0.0
            for s in step_block:
                step_norm_sq += float(torch.sum(s.detach().float() * s.detach().float()).item())
            step_norm = math.sqrt(max(step_norm_sq, 0.0))

            # alpha = 0 row can be written without re-evaluation.
            curve_rows.append(
                {
                    "epoch": int(epoch),
                    "encoder": enc,
                    "block": int(blk),
                    "alpha": 0.0,
                    "step_norm": float(step_norm),
                    "few_loss": base_few_loss,
                    "full_loss": base_full_loss,
                    "test_loss": base_test_loss,
                }
            )

            # Probe non-zero alphas.
            for alpha in alpha_grid.tolist():
                if abs(alpha) < 1e-12:
                    continue
                with torch.no_grad():
                    for p_name, p, p_base, p_step in zip(names, [name_to_param[n] for n in names], base_block, step_block):
                        p.copy_(p_base + float(alpha) * p_step)

                few_loss, _ = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, logit_scale, args.device)
                full_loss, _ = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, logit_scale, args.device)
                test_loss, _ = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device)
                curve_rows.append(
                    {
                        "epoch": int(epoch),
                        "encoder": enc,
                        "block": int(blk),
                        "alpha": float(alpha),
                        "step_norm": float(step_norm),
                        "few_loss": float(few_loss),
                        "full_loss": float(full_loss),
                        "test_loss": float(test_loss),
                    }
                )

            # Restore checkpoint state for this block.
            with torch.no_grad():
                for p, p_base in zip([name_to_param[n] for n in names], base_block):
                    p.copy_(p_base)

            # Summaries for this block.
            rows_this = [r for r in curve_rows if int(r["epoch"]) == int(epoch) and r["encoder"] == enc and int(r["block"]) == int(blk)]
            rows_this = sorted(rows_this, key=lambda x: float(x["alpha"]))
            a = np.array([float(r["alpha"]) for r in rows_this], dtype=np.float64)
            few_curve = np.array([float(r["few_loss"]) for r in rows_this], dtype=np.float64)
            full_curve = np.array([float(r["full_loss"]) for r in rows_this], dtype=np.float64)
            test_curve = np.array([float(r["test_loss"]) for r in rows_this], dtype=np.float64)
            few_sum = curve_summary(a, few_curve, tol_rel=args.safe_tol_rel, tol_abs=args.safe_tol_abs)
            full_sum = curve_summary(a, full_curve, tol_rel=args.safe_tol_rel, tol_abs=args.safe_tol_abs)
            test_sum = curve_summary(a, test_curve, tol_rel=args.safe_tol_rel, tol_abs=args.safe_tol_abs)

            summary_rows.append(
                {
                    "epoch": int(epoch),
                    "encoder": enc,
                    "block": int(blk),
                    "step_norm": float(step_norm),
                    "few_loss0": float(few_sum["loss0"]),
                    "full_loss0": float(full_sum["loss0"]),
                    "test_loss0": float(test_sum["loss0"]),
                    "few_delta_at1": float(few_sum["delta_at_1"]),
                    "full_delta_at1": float(full_sum["delta_at_1"]),
                    "test_delta_at1": float(test_sum["delta_at_1"]),
                    "few_curvature0": float(few_sum["curvature0"]),
                    "full_curvature0": float(full_sum["curvature0"]),
                    "test_curvature0": float(test_sum["curvature0"]),
                    "few_safe_radius": float(few_sum["safe_radius"]),
                    "full_safe_radius": float(full_sum["safe_radius"]),
                    "test_safe_radius": float(test_sum["safe_radius"]),
                    "few_alpha_min": float(few_sum["alpha_min"]),
                    "full_alpha_min": float(full_sum["alpha_min"]),
                    "test_alpha_min": float(test_sum["alpha_min"]),
                    "few_loss_min": float(few_sum["loss_min"]),
                    "full_loss_min": float(full_sum["loss_min"]),
                    "test_loss_min": float(test_sum["loss_min"]),
                    "gap_full_minus_few_at1": float(full_sum["delta_at_1"] - few_sum["delta_at_1"]),
                    "gap_test_minus_few_at1": float(test_sum["delta_at_1"] - few_sum["delta_at_1"]),
                    "gap_alpha_min_full_minus_few": float(full_sum["alpha_min"] - few_sum["alpha_min"]),
                    "gap_alpha_min_test_minus_few": float(test_sum["alpha_min"] - few_sum["alpha_min"]),
                }
            )

        print(f"[ExperimentF] Analysed epoch {epoch:02d}")

    # Write measurement bundles.
    with open(out_dir / f"{run_name}_block_landscape.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    with open(out_dir / f"{run_name}_block_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Compact summary: safest / most fragile blocks by full_delta_at1 and full_safe_radius.
    by_epoch_encoder: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_epoch_encoder[(int(row["epoch"]), str(row["encoder"]))].append(row)

    compact = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "checkpoint_epochs": args.checkpoint_epochs,
        "alpha_grid": alpha_grid.tolist(),
        "per_epoch_encoder": {},
    }
    for (epoch, enc), rows in sorted(by_epoch_encoder.items()):
        rows_sorted_fragile = sorted(rows, key=lambda r: float(r["full_delta_at1"]), reverse=True)
        rows_sorted_safe = sorted(rows, key=lambda r: float(r["full_delta_at1"]))
        rows_sorted_radius = sorted(rows, key=lambda r: float(r["full_safe_radius"]))
        compact["per_epoch_encoder"][f"epoch_{epoch:02d}_{enc}"] = {
            "most_fragile_full_delta_at1": {"block": int(rows_sorted_fragile[0]["block"]), "value": float(rows_sorted_fragile[0]["full_delta_at1"])},
            "safest_full_delta_at1": {"block": int(rows_sorted_safe[0]["block"]), "value": float(rows_sorted_safe[0]["full_delta_at1"])},
            "smallest_full_safe_radius": {"block": int(rows_sorted_radius[0]["block"]), "value": float(rows_sorted_radius[0]["full_safe_radius"])},
            "largest_full_safe_radius": {"block": int(rows_sorted_radius[-1]["block"]), "value": float(rows_sorted_radius[-1]["full_safe_radius"])},
        }

    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(compact, f, indent=2)

    print(f"[ExperimentF] Wrote logs to: {out_dir}")
    print(f"[ExperimentF] Run name: {run_name}")


if __name__ == "__main__":
    run_experiment(parse_args())

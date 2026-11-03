#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment A — 1D interpolation from initialization to final SGD solution.

Goal
----
Understand whether SGD moves into a broad basin or into a narrow support-fitting
basin during few-shot CLIP adaptation.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (but backbone is configurable).
- 1-shot adaptation by default.
- Full-support training: one batch per epoch when --full_support_train is set.
- No augmentation by default (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default.
- Measure support / full-train / test losses along the line
    Theta(alpha) = (1-alpha) * Theta_0 + alpha * Theta_final
  where only the trainable parameters are interpolated; frozen parameters stay at
  the pretrained CLIP values.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_checkpoint_projection.csv
<out_dir>/<run_name>_interpolation.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

The companion plotting script is:
  loss_landscape_experiment_files/plot_experimentA_interpolation.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


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
        help="Experiment A is defined for the SGD trajectory.",
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
        help="Comma-separated epochs t whose trainable checkpoints are saved.",
    )
    parser.add_argument(
        "--interp_points",
        type=int,
        default=101,
        help="Number of alpha points in the 1D interpolation curve.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size used for full-train / test loss evaluation.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentA_logs",
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
        # ImageFolder-like dataset (not used in the three target datasets, but handled for robustness)
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


def _state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    s = 0.0
    for k in a.keys():
        s += float((a[k].double() * b[k].double()).sum().item())
    return s


def _state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return _state_dot(a, a)


def _state_sub(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out


def project_checkpoint_onto_line(
    state_t: MutableMapping[str, torch.Tensor],
    init_state: MutableMapping[str, torch.Tensor],
    final_state: MutableMapping[str, torch.Tensor],
) -> Tuple[float, float]:
    d = _state_sub(final_state, init_state)
    x = _state_sub(state_t, init_state)
    denom = _state_sqnorm(d)
    if denom <= 0.0:
        return 0.0, 0.0
    alpha = _state_dot(x, d) / denom
    orth_sq = 0.0
    for k in d.keys():
        r = x[k].double() - alpha * d[k].double()
        orth_sq += float((r * r).sum().item())
    orth_rel = math.sqrt(orth_sq / max(denom, 1e-32))
    return float(alpha), float(orth_rel)


def interpolate_state(
    init_state: MutableMapping[str, torch.Tensor],
    final_state: MutableMapping[str, torch.Tensor],
    alpha: float,
) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in init_state.keys():
        out[k] = (1.0 - alpha) * init_state[k] + alpha * final_state[k]
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


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


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
            f"experimentA_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
            f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
        )
    else:
        run_name = args.run_name

    ckpt_dir = out_dir / "checkpoints" / run_name
    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)

    # Geometry experiment: no augmentation, so preprocess is used everywhere.
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
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
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
        "interp_points": args.interp_points,
        "eval_batch_size": args.eval_batch_size,
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

    # Save / evaluate t = 0.
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
            f"[ExperimentA] epoch={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | "
            f"test_loss={test_loss:.4f} | few_acc={few_acc:.2f} | test_acc={test_acc:.2f}"
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
    print(f"[ExperimentA] Training finished in {elapsed:.1f}s")

    # Prepare interpolation from init to final SGD solution.
    final_state = get_trainable_state(clip_model)
    alphas = torch.linspace(0.0, 1.0, args.interp_points)
    interp_rows: List[Dict[str, object]] = []

    for alpha in alphas.tolist():
        interp_state = interpolate_state(init_state, final_state, alpha)
        set_trainable_state(clip_model, interp_state)
        few_loss, few_acc = compute_loader_loss_acc(clip_model, few_eval_loader, class_tokens, logit_scale, args.device)
        full_loss, full_acc = compute_loader_loss_acc(clip_model, full_eval_loader, class_tokens, logit_scale, args.device)
        test_loss, test_acc = compute_loader_loss_acc(clip_model, test_loader, class_tokens, logit_scale, args.device)
        interp_rows.append(
            {
                "alpha": float(alpha),
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }
        )

    # Restore final state for completeness.
    set_trainable_state(clip_model, final_state)

    # Checkpoint projection table.
    projection_rows: List[Dict[str, object]] = []
    for row in trajectory_rows:
        state_t, meta = load_trainable_checkpoint(Path(row["checkpoint_path"]))
        alpha_proj, orth_rel = project_checkpoint_onto_line(state_t, init_state, final_state)
        projection_rows.append(
            {
                "epoch": int(row["epoch"]),
                "alpha_proj": float(alpha_proj),
                "alpha_proj_clipped": float(min(max(alpha_proj, 0.0), 1.0)),
                "orth_rel": float(orth_rel),
                "few_loss": float(row["few_loss"]),
                "full_loss": float(row["full_loss"]),
                "test_loss": float(row["test_loss"]),
                "few_acc": float(row["few_acc"]),
                "full_acc": float(row["full_acc"]),
                "test_acc": float(row["test_acc"]),
            }
        )

    # Write CSV bundles.
    import csv

    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with open(out_dir / f"{run_name}_interpolation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(interp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(interp_rows)

    with open(out_dir / f"{run_name}_checkpoint_projection.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(projection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(projection_rows)

    # Summary metrics for basin shape.
    few_curve = torch.tensor([r["few_loss"] for r in interp_rows], dtype=torch.float64)
    full_curve = torch.tensor([r["full_loss"] for r in interp_rows], dtype=torch.float64)
    test_curve = torch.tensor([r["test_loss"] for r in interp_rows], dtype=torch.float64)
    summary = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "few_loss_curve": interpolation_barrier(alphas.double(), few_curve),
        "full_loss_curve": interpolation_barrier(alphas.double(), full_curve),
        "test_loss_curve": interpolation_barrier(alphas.double(), test_curve),
    }
    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ExperimentA] Wrote logs to: {out_dir}")
    print(f"[ExperimentA] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

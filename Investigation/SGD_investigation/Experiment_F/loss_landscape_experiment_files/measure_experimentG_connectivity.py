#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment G — Basin connectivity across SGD seeds.

Goal
----
Understand whether full-support SGD trajectories for few-shot CLIP adaptation
end in a broad connected basin or in narrow support-fitting basins.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (backbone configurable).
- 1-shot adaptation by default.
- Full-support training: one batch per epoch when --full_support_train is set.
- No augmentation by default (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default, for every seed.
- Train one SGD trajectory per requested seed.
- For every seed pair (i, j), linearly interpolate only the trainable
  parameters between the final seed solutions:
      Theta(alpha) = (1-alpha) * Theta_i + alpha * Theta_j.
- Evaluate, along that path:
      * few-pair loss   = 0.5 * (L_few^i + L_few^j)
      * full-train loss = L_full
      * test loss       = L_test

Notes
-----
Under this geometry protocol (full-support, no augmentation, deterministic CLIP
initialization), changing the seed mainly changes the sampled few-shot support
set. Therefore the pairwise support loss reported here is the *mean of the two
endpoint support tasks*, which is the natural pairwise analogue of L_few.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_pairwise_interpolation.csv
<out_dir>/<run_name>_pairwise_summary.csv
<out_dir>/checkpoints/<run_name>/seed_XX/epoch_YY_trainable.pt

The companion plotting script is:
  loss_landscape_experiment_files/plot_experimentG_connectivity.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated SGD seeds / few-shot task seeds.")
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
        help="Experiment G is defined for SGD trajectories.",
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
        help="Comma-separated epochs t whose trainable checkpoints are saved for every seed.",
    )
    parser.add_argument(
        "--interp_points",
        type=int,
        default=101,
        help="Number of alpha points in each pairwise interpolation curve.",
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
        default="./results_without_aug/experimentG_logs",
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
    args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(args.seeds) < 2:
        raise ValueError("Experiment G needs at least two seeds, e.g. --seeds 1,2,3")
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


def interpolate_state(
    state_a: MutableMapping[str, torch.Tensor],
    state_b: MutableMapping[str, torch.Tensor],
    alpha: float,
) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
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
    width1_mask = values <= (vmin + 0.01 * max(max(start, end) - vmin, 1e-12))
    width5_mask = values <= (vmin + 0.05 * max(max(start, end) - vmin, 1e-12))

    def _width(mask: torch.Tensor) -> float:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0.0
        return float(alpha[idx[-1]].item() - alpha[idx[0]].item())

    return {
        "start": start,
        "end": end,
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


def pair_distance_metrics(
    state_a: MutableMapping[str, torch.Tensor],
    state_b: MutableMapping[str, torch.Tensor],
    init_state: MutableMapping[str, torch.Tensor],
) -> Dict[str, float]:
    d_ab = _state_sub(state_a, state_b)
    d_a0 = _state_sub(state_a, init_state)
    d_b0 = _state_sub(state_b, init_state)
    dist_ab = math.sqrt(_state_sqnorm(d_ab))
    dist_a0 = math.sqrt(_state_sqnorm(d_a0))
    dist_b0 = math.sqrt(_state_sqnorm(d_b0))
    rel = dist_ab / max(0.5 * (dist_a0 + dist_b0), 1e-32)
    return {
        "pair_l2_distance": float(dist_ab),
        "seed_a_l2_from_init": float(dist_a0),
        "seed_b_l2_from_init": float(dist_b0),
        "pair_rel_distance": float(rel),
    }


def train_one_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    preprocess,
    full_dataset,
    class_tokens: torch.Tensor,
    logit_scale: float,
    full_eval_loader,
    test_loader,
    out_dir: Path,
    run_name: str,
) -> Dict[str, object]:
    set_random_seed(seed)
    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)

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

    clip_model, _ = load_clip_model(args.backbone, args.device)
    clip_model = clip_model.to(args.device)
    clip_model.eval()
    mark_as_trainable(clip_model, args)
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    optimizer = SGD(trainable_params, lr=args.lr)
    amp_enabled = str(args.device).startswith("cuda")
    scaler = GradScaler("cuda") if amp_enabled else GradScaler("cpu")

    init_state = get_trainable_state(clip_model)
    ckpt_dir = out_dir / "checkpoints" / run_name / f"seed_{seed:02d}"
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
                "seed": int(seed),
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
                "seed": int(seed),
                "epoch": int(epoch),
                "few_loss": float(few_loss),
                "few_acc": float(few_acc),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "few_size": int(len(few_dataset.train_x)),
                "checkpoint_path": str(ckpt_path),
            }
        )
        print(
            f"[ExperimentG][seed={seed}] epoch={epoch:02d} | few_loss={few_loss:.4f} | "
            f"full_loss={full_loss:.4f} | test_loss={test_loss:.4f} | "
            f"few_acc={few_acc:.2f} | test_acc={test_acc:.2f}"
        )

    eval_and_save(0, init_state)

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

    final_state = get_trainable_state(clip_model)
    return {
        "seed": int(seed),
        "init_state": init_state,
        "final_state": final_state,
        "trajectory_rows": trajectory_rows,
        "few_eval_loader": few_eval_loader,
        "few_size": int(len(few_dataset.train_x)),
    }


def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a CPU CLIP instance just to obtain the preprocess pipeline without
    # occupying GPU memory before the per-seed runs start.
    _cpu_model, preprocess = load_clip_model(args.backbone, "cpu")
    del _cpu_model

    # The full-train and test objectives are common to all seed-specific runs.
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)
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
    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)
    logit_scale = 100.0

    if args.run_name is None:
        bb = args.backbone.replace("/", "-")
        seeds_tag = "-".join(str(s) for s in args.seeds)
        run_name = (
            f"experimentG_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
            f"seeds_{seeds_tag}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
        )
    else:
        run_name = args.run_name

    cfg = {
        "seeds": args.seeds,
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
        "full_train_size": len(full_dataset.train_x),
        "test_size": len(full_dataset.test),
        "logit_scale": logit_scale,
        "no_augmentation": True,
        "pair_support_definition": "few_pair_loss = 0.5 * (few_loss_seed_a + few_loss_seed_b)",
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    start_time = time.perf_counter()
    seed_runs: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            args=args,
            preprocess=preprocess,
            full_dataset=full_dataset,
            class_tokens=class_tokens,
            logit_scale=logit_scale,
            full_eval_loader=full_eval_loader,
            test_loader=test_loader,
            out_dir=out_dir,
            run_name=run_name,
        )
        seed_runs.append(result)
        trajectory_rows.extend(result["trajectory_rows"])

    elapsed = time.perf_counter() - start_time
    print(f"[ExperimentG] Training {len(args.seeds)} seeds finished in {elapsed:.1f}s")

    # Fresh evaluation model for pairwise interpolation.
    eval_model, _ = load_clip_model(args.backbone, args.device)
    eval_model = eval_model.to(args.device)
    eval_model.eval()
    mark_as_trainable(eval_model, args)

    alphas = torch.linspace(0.0, 1.0, args.interp_points, dtype=torch.float64)
    pairwise_rows: List[Dict[str, object]] = []
    pairwise_summary_rows: List[Dict[str, object]] = []

    for run_a, run_b in itertools.combinations(seed_runs, 2):
        seed_a = int(run_a["seed"])
        seed_b = int(run_b["seed"])
        pair_label = f"{seed_a}↔{seed_b}"
        print(f"[ExperimentG] Measuring pairwise interpolation {pair_label}")
        state_a = run_a["final_state"]
        state_b = run_b["final_state"]
        init_state = run_a["init_state"]
        pair_metrics = pair_distance_metrics(state_a, state_b, init_state)

        pair_curve_rows: List[Dict[str, object]] = []
        for alpha in alphas.tolist():
            interp_state = interpolate_state(state_a, state_b, float(alpha))
            set_trainable_state(eval_model, interp_state)

            few_a_loss, few_a_acc = compute_loader_loss_acc(
                eval_model, run_a["few_eval_loader"], class_tokens, logit_scale, args.device
            )
            few_b_loss, few_b_acc = compute_loader_loss_acc(
                eval_model, run_b["few_eval_loader"], class_tokens, logit_scale, args.device
            )
            full_loss, full_acc = compute_loader_loss_acc(
                eval_model, full_eval_loader, class_tokens, logit_scale, args.device
            )
            test_loss, test_acc = compute_loader_loss_acc(
                eval_model, test_loader, class_tokens, logit_scale, args.device
            )

            row = {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "pair_label": pair_label,
                "alpha": float(alpha),
                "few_loss_seed_a": float(few_a_loss),
                "few_acc_seed_a": float(few_a_acc),
                "few_loss_seed_b": float(few_b_loss),
                "few_acc_seed_b": float(few_b_acc),
                "few_pair_loss": 0.5 * (float(few_a_loss) + float(few_b_loss)),
                "few_pair_acc": 0.5 * (float(few_a_acc) + float(few_b_acc)),
                "full_loss": float(full_loss),
                "full_acc": float(full_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                **pair_metrics,
            }
            pair_curve_rows.append(row)
            pairwise_rows.append(row)

        few_pair_curve = torch.tensor([r["few_pair_loss"] for r in pair_curve_rows], dtype=torch.float64)
        full_curve = torch.tensor([r["full_loss"] for r in pair_curve_rows], dtype=torch.float64)
        test_curve = torch.tensor([r["test_loss"] for r in pair_curve_rows], dtype=torch.float64)
        pairwise_summary_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "pair_label": pair_label,
                **pair_metrics,
                **{f"few_pair_{k}": v for k, v in interpolation_barrier(alphas, few_pair_curve).items()},
                **{f"full_{k}": v for k, v in interpolation_barrier(alphas, full_curve).items()},
                **{f"test_{k}": v for k, v in interpolation_barrier(alphas, test_curve).items()},
            }
        )

    # Write CSV bundles.
    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with open(out_dir / f"{run_name}_pairwise_interpolation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    with open(out_dir / f"{run_name}_pairwise_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_summary_rows)

    print(f"[ExperimentG] Wrote logs to: {out_dir}")
    print(f"[ExperimentG] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

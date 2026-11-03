#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment 6: feature-drift audit for few-shot CLIP adaptation.

Goal
----
Answer the question:
    "Does Adam fail because it pushes CLIP too far from the pretrained
    solution too fast?"

This script mirrors the design of Experiment 2 as closely as possible:
- it uses the same dataset construction,
- the same full-support training option,
- the same SGD / Adam trajectory interface,
- the same trainable-block selection logic,
- and it writes a compact log bundle that can be compared with a dedicated
  plotting script.

Protocol (exactly as recommended)
---------------------------------
Along an SGD or Adam trajectory, at every diagnostic step, we track:
1. parameter drift relative to the pretrained CLIP solution,
2. per-block parameter drift,
3. image-feature drift on the support / full-train / test splits,
4. text-feature drift,
5. logit drift and class-margin drift,
6. support / full-train / test loss and accuracy.

Outputs
-------
For each run, the script writes:
  - <run_name>_config.json
  - <run_name>_per_iteration.csv
  - <run_name>_per_block.csv

These files are consumed by plot_experiment6_feature_drift.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from second_moment_experiment_files.measure_second_moment import (
    _str2bool,
    build_dataset_bundle,
    build_loaders,
    load_clip_model,
    make_optimizer,
    tokenize_class_texts,
)
from run_utils import set_random_seed
from train import mark_as_trainable
from second_moment_experiment_files.clip_block_utils import BlockKey, build_param_to_block_map

EPS = 1e-12
LOGIT_SCALE = 100.0


# ============================================================================
# CLI
# ============================================================================

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
    parser.add_argument("--shots", type=int, default=1, help="Few-shot support size K.")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
        help="Which CLIP transformer blocks to adapt (same convention as train.py).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="both",
        choices=["vision", "text", "both"],
        help="Which encoder(s) to adapt and analyse.",
    )

    parser.add_argument(
        "--traj_optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam"],
        help="Optimizer used for the training trajectory.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size. Set <=0 or use --full_support_train for one full-support batch per epoch.",
    )
    parser.add_argument(
        "--full_support_train",
        action="store_true",
        help="Force one full-support batch per epoch for the training trajectory.",
    )
    parser.add_argument("--use_aug", action="store_true", help="Use the same augmentation as main.py for the trajectory.")

    parser.add_argument(
        "--ref_mode",
        type=str,
        default="full_train",
        choices=["kshot", "full_train"],
        help="Reference-set construction.",
    )
    parser.add_argument(
        "--ref_shots",
        type=int,
        default=16,
        help="Reference K when --ref_mode=kshot.",
    )
    parser.add_argument(
        "--max_ref_per_class",
        type=int,
        default=-1,
        help="Optional cap per class for --ref_mode=full_train (-1 = use all available images).",
    )
    parser.add_argument(
        "--repeat_if_needed",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="Repeat images when a class has fewer than K examples (default: True).",
    )

    parser.add_argument(
        "--diag_every",
        type=int,
        default=1,
        help="Run the feature-drift diagnostic every N optimizer steps.",
    )
    parser.add_argument(
        "--max_diag_steps",
        type=int,
        default=-1,
        help="Stop logging after this many diagnostic checkpoints (-1 = log whenever scheduled).",
    )
    parser.add_argument(
        "--diag_batch_size",
        type=int,
        default=256,
        help="Evaluation batch size for support / full-train / test diagnostics. Set <=0 for full-support.",
    )
    parser.add_argument(
        "--diag_eval_mode",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="If True, compute diagnostics in model.eval() mode.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="feature_drift_logs",
        help="Directory where the log bundle will be written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, the script creates one automatically.",
    )

    args = parser.parse_args()

    # Keep the exact same root-resolution behaviour as Experiment 2.
    from utils import resolve_root
    args.root_path = resolve_root(args.root_path)
    if args.ref_mode == "kshot" and args.ref_shots < args.shots:
        raise ValueError("--ref_shots must be >= --shots when --ref_mode=kshot")
    return args


# ============================================================================
# Small numeric helpers
# ============================================================================

def _safe_div(num: float, den: float, eps: float = EPS) -> float:
    return float(num) / float(den + eps)


@torch.no_grad()
def _mean_row_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(x, y, dim=1).mean().item())


@torch.no_grad()
def _ground_truth_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gt = logits.gather(1, target.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked[torch.arange(logits.size(0), device=logits.device), target] = -float("inf")
    best_other = masked.max(dim=1).values
    return gt - best_other


# ============================================================================
# Drift computation helpers
# ============================================================================

def snapshot_initial_params(
    trainable_params: Sequence[torch.nn.Parameter],
) -> Dict[torch.nn.Parameter, torch.Tensor]:
    return {p: p.detach().float().cpu().clone() for p in trainable_params}


@torch.no_grad()
def compute_parameter_drift(
    trainable_params: Sequence[torch.nn.Parameter],
    init_params: Dict[torch.nn.Parameter, torch.Tensor],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    enc_sq: Dict[str, Dict[str, float]] = {
        "vision": {"delta_sq": 0.0, "init_sq": 0.0},
        "text": {"delta_sq": 0.0, "init_sq": 0.0},
        "all": {"delta_sq": 0.0, "init_sq": 0.0},
    }
    block_sq: MutableMapping[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: {"delta_sq": 0.0, "init_sq": 0.0})

    for p in trainable_params:
        p0 = init_params[p]
        cur = p.detach().float().cpu()
        delta = cur - p0
        delta_sq = float(torch.sum(delta * delta).item())
        init_sq = float(torch.sum(p0 * p0).item())

        enc_sq["all"]["delta_sq"] += delta_sq
        enc_sq["all"]["init_sq"] += init_sq

        enc, blk = param_to_block.get(p, ("other", -1))
        if enc in {"vision", "text"}:
            enc_sq[enc]["delta_sq"] += delta_sq
            enc_sq[enc]["init_sq"] += init_sq
            block_sq[(enc, blk)]["delta_sq"] += delta_sq
            block_sq[(enc, blk)]["init_sq"] += init_sq

    enc_rows = {}
    for enc, vals in enc_sq.items():
        delta_norm = math.sqrt(vals["delta_sq"])
        init_norm = math.sqrt(vals["init_sq"])
        enc_rows[enc] = {
            "param_delta_norm": delta_norm,
            "param_init_norm": init_norm,
            "param_drift_rel": _safe_div(delta_norm, init_norm),
        }

    block_rows: List[Dict[str, float]] = []
    for (enc, blk), vals in sorted(block_sq.items(), key=lambda x: (x[0][0], x[0][1])):
        delta_norm = math.sqrt(vals["delta_sq"])
        init_norm = math.sqrt(vals["init_sq"])
        block_rows.append({
            "encoder": enc,
            "block": int(blk),
            "param_delta_norm": delta_norm,
            "param_init_norm": init_norm,
            "param_drift_rel": _safe_div(delta_norm, init_norm),
        })
    return enc_rows, block_rows


@torch.no_grad()
def encode_text_features(model, class_tokens: torch.Tensor) -> torch.Tensor:
    text_features = model.encode_text(class_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def build_split_cache(
    *,
    model,
    loader: DataLoader,
    class_tokens: torch.Tensor,
    device: str,
    logit_scale: float,
    eval_mode: bool,
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    if eval_mode:
        model.eval()
    else:
        model.train()

    text_features_0 = encode_text_features(model, class_tokens)
    image_feats: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    margins_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features_0.t())
        margins = _ground_truth_margin(logits, target)

        image_feats.append(image_features.detach().cpu())
        logits_list.append(logits.detach().cpu())
        margins_list.append(margins.detach().cpu())
        targets_list.append(target.detach().cpu())

    model.train(was_training)
    return {
        "image_features_0": torch.cat(image_feats, dim=0),
        "logits_0": torch.cat(logits_list, dim=0),
        "gt_margin_0": torch.cat(margins_list, dim=0),
        "targets": torch.cat(targets_list, dim=0),
        "text_features_0": text_features_0.detach().cpu(),
    }


@torch.no_grad()
def evaluate_split_with_cache(
    *,
    model,
    loader: DataLoader,
    text_features_cur: torch.Tensor,
    cache: Dict[str, torch.Tensor],
    device: str,
    logit_scale: float,
    eval_mode: bool,
    prefix: str,
) -> Dict[str, float]:
    was_training = model.training
    if eval_mode:
        model.eval()
    else:
        model.train()

    loss_sum = 0.0
    correct = 0.0
    n = 0
    image_cos_sum = 0.0
    logit_abs_delta_sum = 0.0
    margin_abs_delta_sum = 0.0
    margin_sum = 0.0
    n_logit_entries = 0

    offset = 0
    for images, target in loader:
        bs = int(target.size(0))
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features_cur.t())
        margins = _ground_truth_margin(logits, target)

        loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
        correct += float((logits.argmax(dim=1) == target).sum().item())
        margin_sum += float(margins.sum().item())
        n += bs
        n_logit_entries += int(logits.numel())

        init_img = cache["image_features_0"][offset:offset + bs].to(device, non_blocking=True)
        init_logits = cache["logits_0"][offset:offset + bs].to(device, non_blocking=True)
        init_margin = cache["gt_margin_0"][offset:offset + bs].to(device, non_blocking=True)

        image_cos_sum += float((image_features * init_img).sum(dim=1).sum().item())
        logit_abs_delta_sum += float((logits - init_logits).abs().sum().item())
        margin_abs_delta_sum += float((margins - init_margin).abs().sum().item())
        offset += bs

    model.train(was_training)
    image_cos = _safe_div(image_cos_sum, n)
    return {
        f"{prefix}_loss": _safe_div(loss_sum, n),
        f"{prefix}_acc": 100.0 * _safe_div(correct, n),
        f"{prefix}_image_feature_cos_to_init": image_cos,
        f"{prefix}_image_feature_drift": 1.0 - image_cos,
        f"{prefix}_logit_delta_abs_mean": _safe_div(logit_abs_delta_sum, n_logit_entries),
        f"{prefix}_gt_margin_mean": _safe_div(margin_sum, n),
        f"{prefix}_gt_margin_delta_abs_mean": _safe_div(margin_abs_delta_sum, n),
    }


# ============================================================================
# Logging
# ============================================================================

class FeatureDriftLogger:
    def __init__(self, out_dir: Path, run_name: str):
        self.out_dir = out_dir
        self.run_name = run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.iter_path = self.out_dir / f"{run_name}_per_iteration.csv"
        self.block_path = self.out_dir / f"{run_name}_per_block.csv"

        self._iter_f = open(self.iter_path, "w", newline="")
        self._block_f = open(self.block_path, "w", newline="")
        self._iter_w: Optional[csv.DictWriter] = None
        self._block_w: Optional[csv.DictWriter] = None

    def log_iteration(self, row: Dict[str, object]) -> None:
        if self._iter_w is None:
            self._iter_w = csv.DictWriter(self._iter_f, fieldnames=list(row.keys()))
            self._iter_w.writeheader()
        self._iter_w.writerow(row)
        self._iter_f.flush()

    def log_blocks(self, rows: Sequence[Dict[str, object]]) -> None:
        for row in rows:
            if self._block_w is None:
                self._block_w = csv.DictWriter(self._block_f, fieldnames=list(row.keys()))
                self._block_w.writeheader()
            self._block_w.writerow(row)
        self._block_f.flush()

    def close(self) -> None:
        self._iter_f.close()
        self._block_f.close()


# ============================================================================
# Main experiment
# ============================================================================

def default_run_name(args: argparse.Namespace) -> str:
    bb = args.backbone.replace("/", "-")
    if args.ref_mode == "kshot":
        ref_tag = f"kshot_{args.ref_shots}"
    else:
        ref_tag = "full"
        if args.max_ref_per_class > 0:
            ref_tag += f"_cap_{args.max_ref_per_class}"
    return (
        f"feature_drift_dataset_{args.dataset}_noisyshots_{args.shots}_ref_{ref_tag}"
        f"_backbone_{bb}_trajopt_{args.traj_optimizer.lower()}_seed_{args.seed}"
        f"_pos_{args.position}_enc_{args.encoder}"
    )


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    model, preprocess = load_clip_model(args.backbone, args.device)
    bundle = build_dataset_bundle(args, preprocess)
    train_loader, few_diag_loader, ref_diag_loader, test_loader = build_loaders(bundle, args, preprocess)

    mark_as_trainable(model, args)
    model = model.to(args.device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected. Check --encoder and --position.")

    optimizer = make_optimizer(args.traj_optimizer, trainable_params, args.lr, 1e-8)
    param_to_block = build_param_to_block_map(model, only_trainable=True)
    class_tokens = tokenize_class_texts(bundle.classnames, bundle.template, args.device)

    # Snapshot the pretrained solution (only trainable parameters are needed for the drift audit).
    init_params = snapshot_initial_params(trainable_params)

    print("[FeatureDrift] Precomputing initial support / full-train / test caches ...")
    init_support_cache = build_split_cache(
        model=model,
        loader=few_diag_loader,
        class_tokens=class_tokens,
        device=args.device,
        logit_scale=LOGIT_SCALE,
        eval_mode=args.diag_eval_mode,
    )
    init_ref_cache = build_split_cache(
        model=model,
        loader=ref_diag_loader,
        class_tokens=class_tokens,
        device=args.device,
        logit_scale=LOGIT_SCALE,
        eval_mode=args.diag_eval_mode,
    )
    init_test_cache = build_split_cache(
        model=model,
        loader=test_loader,
        class_tokens=class_tokens,
        device=args.device,
        logit_scale=LOGIT_SCALE,
        eval_mode=args.diag_eval_mode,
    )
    init_text_features = init_support_cache["text_features_0"].to(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)

    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger = FeatureDriftLogger(out_dir=out_dir, run_name=run_name)

    print(f"[FeatureDrift] Logging bundle to: {out_dir}")
    print(f"[FeatureDrift] Run name: {run_name}")
    print(
        f"[FeatureDrift] trajectory={args.traj_optimizer} | full_support_train={args.full_support_train or args.batch_size <= 0} "
        f"| ref_mode={args.ref_mode}"
    )

    global_step = 0
    diag_count = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for step_in_epoch, (images, target) in enumerate(train_loader):
            should_log = (global_step % max(1, args.diag_every) == 0)
            if args.max_diag_steps > 0:
                should_log = should_log and (diag_count < args.max_diag_steps)

            if should_log:
                print(
                    f"[FeatureDrift] step={global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"measuring parameter / feature / logit drift ..."
                )

                enc_drift, block_rows = compute_parameter_drift(trainable_params, init_params, param_to_block)

                was_training = model.training
                if args.diag_eval_mode:
                    model.eval()
                text_features_cur = encode_text_features(model, class_tokens)
                text_cos = float(F.cosine_similarity(text_features_cur, init_text_features, dim=1).mean().item())
                text_drift = 1.0 - text_cos
                if was_training:
                    model.train()

                few_metrics = evaluate_split_with_cache(
                    model=model,
                    loader=few_diag_loader,
                    text_features_cur=text_features_cur,
                    cache=init_support_cache,
                    device=args.device,
                    logit_scale=LOGIT_SCALE,
                    eval_mode=args.diag_eval_mode,
                    prefix="few",
                )
                ref_metrics = evaluate_split_with_cache(
                    model=model,
                    loader=ref_diag_loader,
                    text_features_cur=text_features_cur,
                    cache=init_ref_cache,
                    device=args.device,
                    logit_scale=LOGIT_SCALE,
                    eval_mode=args.diag_eval_mode,
                    prefix="ref",
                )
                test_metrics = evaluate_split_with_cache(
                    model=model,
                    loader=test_loader,
                    text_features_cur=text_features_cur,
                    cache=init_test_cache,
                    device=args.device,
                    logit_scale=LOGIT_SCALE,
                    eval_mode=args.diag_eval_mode,
                    prefix="test",
                )

                iter_row: Dict[str, object] = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "traj_optimizer": args.traj_optimizer,
                    "dataset": args.dataset,
                    "shots": args.shots,
                    "ref_mode": args.ref_mode,
                    "ref_shots": args.ref_shots,
                    "backbone": args.backbone,
                    "position": args.position,
                    "encoder": args.encoder,
                    "few_size": len(few_diag_loader.dataset),
                    "ref_size": len(ref_diag_loader.dataset),
                    "test_size": len(test_loader.dataset),
                    "vision_param_delta_norm": enc_drift.get("vision", {}).get("param_delta_norm", float("nan")),
                    "vision_param_init_norm": enc_drift.get("vision", {}).get("param_init_norm", float("nan")),
                    "vision_param_drift_rel": enc_drift.get("vision", {}).get("param_drift_rel", float("nan")),
                    "text_param_delta_norm": enc_drift.get("text", {}).get("param_delta_norm", float("nan")),
                    "text_param_init_norm": enc_drift.get("text", {}).get("param_init_norm", float("nan")),
                    "text_param_drift_rel": enc_drift.get("text", {}).get("param_drift_rel", float("nan")),
                    "all_param_delta_norm": enc_drift.get("all", {}).get("param_delta_norm", float("nan")),
                    "all_param_init_norm": enc_drift.get("all", {}).get("param_init_norm", float("nan")),
                    "all_param_drift_rel": enc_drift.get("all", {}).get("param_drift_rel", float("nan")),
                    "text_feature_cos_to_init": text_cos,
                    "text_feature_drift": text_drift,
                }
                iter_row.update(few_metrics)
                iter_row.update(ref_metrics)
                iter_row.update(test_metrics)
                logger.log_iteration(iter_row)

                out_block_rows: List[Dict[str, object]] = []
                for row in block_rows:
                    out_row = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "scope": "block",
                        "traj_optimizer": args.traj_optimizer,
                        "dataset": args.dataset,
                        "shots": args.shots,
                        "ref_mode": args.ref_mode,
                        "ref_shots": args.ref_shots,
                        "backbone": args.backbone,
                        "position": args.position,
                        "few_size": len(few_diag_loader.dataset),
                        "ref_size": len(ref_diag_loader.dataset),
                    }
                    out_row.update(row)
                    out_block_rows.append(out_row)
                logger.log_blocks(out_block_rows)
                diag_count += 1

            optimizer.zero_grad(set_to_none=True)
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = LOGIT_SCALE * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            bs = int(target.size(0))
            epoch_loss += float(loss.item()) * bs
            epoch_samples += bs
            global_step += 1

        mean_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"[Epoch {epoch:02d}] training loss = {mean_epoch_loss:.4f}")

    logger.close()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

r"""Experiment 7: temporal staleness of the few-shot gradient.

This script follows the exact temporal-staleness protocol proposed for the
Adam-collapse study.

Question
--------
Does the few-shot gradient itself become stale over time?

Protocol
--------
At diagnostic checkpoints \Theta_t and \Theta_{t+\Delta}, we compare:

    cos(m_t^few,  m_{t+\Delta}^few)
    cos(m_t^few,  m_{t+\Delta}^full)

for small temporal lags \Delta in {1, 2, 5} by default.

Interpretation:
  - cos(m_t^few, m_{t+\Delta}^few) measures temporal autocorrelation of the
    few-shot gradient itself.
  - cos(m_t^few, m_{t+\Delta}^full) measures whether the few-shot gradient at
    time t still aligns with the *future* full-data objective.

The experiment is designed to be run for *both* an SGD trajectory and an Adam
trajectory. The measurement code is identical across both trajectories; only the
training optimizer changes.

Design constraints
------------------
This implementation is intentionally minimal and reuses the Experiment 2 /
second-moment code as much as possible:
  - same dataset / loader / CLIP setup,
  - same trainable-block selection,
  - same full-support training option,
  - same diagnostic few/full loaders.

Outputs
-------
For each run, the script writes:

  - <run_name>_config.json
  - <run_name>_temporal_staleness.csv

The CSV contains both encoder-level rows (scope='encoder') and per-block rows
(scope='block'). The companion plotting script
``plot_experiment7_temporal_staleness.py`` compares one SGD run and one Adam
run for the same task setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import deque, defaultdict
from pathlib import Path
from typing import Deque, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from second_moment_experiment_files.measure_second_moment import (
    _str2bool,
    build_dataset_bundle,
    build_loaders,
    load_clip_model,
    make_optimizer,
    tokenize_class_texts,
)
from run_utils import set_random_seed
from train import evaluate_CLIP, mark_as_trainable
from utils import resolve_root
from second_moment_experiment_files.clip_block_utils import BlockKey, build_param_to_block_map

EPS = 1e-12


# ============================================================================
# Argument parsing
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
        help="Reference-set construction. For the temporal-staleness study, the important signal is usually full_train.",
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
        help="Run the gradient-staleness diagnostic every N optimizer steps. Use 1 for the exact protocol.",
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
        help="Diagnostic mini-batch size. Set <=0 for full-support diagnostics.",
    )
    parser.add_argument(
        "--diag_grad_eval_mode",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="If True, compute diagnostics in model.eval() mode (dropout off).",
    )

    parser.add_argument(
        "--lags",
        type=str,
        default="1,2,5",
        help="Comma-separated temporal lags Δ for cos(m_t^few, m_{t+Δ}^few) and cos(m_t^few, m_{t+Δ}^full).",
    )
    parser.add_argument(
        "--history_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help=(
            "Storage dtype for the rolling history buffer of past few-shot gradients. "
            "float16 greatly reduces RAM and is usually sufficient; set float32 for exact history storage."
        ),
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="gradient_staleness_logs",
        help="Directory where the Experiment 7 CSV bundle will be written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, the script creates one automatically.",
    )
    parser.add_argument("--eval_final", action="store_true", help="Evaluate the final adapted model on the test split.")

    args = parser.parse_args()
    args.root_path = resolve_root(args.root_path)
    args.lags = sorted({int(x) for x in args.lags.split(",") if x.strip()})
    if not args.lags or min(args.lags) <= 0:
        raise ValueError("--lags must contain positive integers, e.g. --lags 1,2,5")
    if args.ref_mode == "kshot" and args.ref_shots < args.shots:
        raise ValueError("--ref_shots must be >= --shots when --ref_mode=kshot")
    return args


# ============================================================================
# Helpers
# ============================================================================


def default_run_name(args: argparse.Namespace) -> str:
    bb = args.backbone.replace("/", "-")
    if args.ref_mode == "kshot":
        ref_tag = f"kshot_{args.ref_shots}"
    else:
        ref_tag = "full"
        if args.max_ref_per_class > 0:
            ref_tag += f"_cap_{args.max_ref_per_class}"
    lag_tag = "-".join(str(x) for x in args.lags)
    return (
        f"gradient_staleness_dataset_{args.dataset}_noisyshots_{args.shots}_ref_{ref_tag}"
        f"_backbone_{bb}_trajopt_{args.traj_optimizer.lower()}_seed_{args.seed}"
        f"_pos_{args.position}_enc_{args.encoder}_lags_{lag_tag}"
    )



def cosine_from_sums(dot_sum: float, norm_x_sq: float, norm_y_sq: float, eps: float = EPS) -> float:
    denom = math.sqrt(max(norm_x_sq, 0.0)) * math.sqrt(max(norm_y_sq, 0.0))
    if denom <= eps:
        return float("nan")
    return float(dot_sum / (denom + eps))



def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> float:
    x = x.float().cpu()
    y = y.float().cpu()
    x_sq = float(torch.dot(x, x).item())
    y_sq = float(torch.dot(y, y).item())
    dot = float(torch.dot(x, y).item())
    return cosine_from_sums(dot, x_sq, y_sq, eps)



def _storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported history dtype: {name}")



def compute_dataset_gradient_only(
    *,
    model,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params: Sequence[torch.nn.Parameter],
    device: str,
    grad_eval_mode: bool,
) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], float]:
    """Estimate the dataset-level mean gradient m_t(D) on a data source D."""
    was_training = model.training
    if grad_eval_mode:
        model.eval()
    else:
        model.train()

    total_size = len(loader.dataset)
    if total_size <= 0:
        raise RuntimeError("Diagnostic loader is empty; cannot estimate gradients.")

    m_est = {p: torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params}
    total_loss = 0.0

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
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
        loss.backward()

        total_loss += float(loss.item()) * batch_size

        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            m_est[p].add_(g, alpha=batch_weight)

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return m_est, total_loss / float(total_size)



def aggregate_gradients_by_block(
    *,
    gradients: Dict[torch.nn.Parameter, torch.Tensor],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
) -> Dict[BlockKey, torch.Tensor]:
    by_block: Dict[BlockKey, List[torch.Tensor]] = defaultdict(list)
    for p, g in gradients.items():
        bk = param_to_block.get(p)
        if bk is None:
            continue
        by_block[bk].append(g.reshape(-1))
    out: Dict[BlockKey, torch.Tensor] = {}
    for bk, tensors in by_block.items():
        out[bk] = torch.cat(tensors, dim=0).float().cpu()
    return out



def compute_temporal_rows(
    *,
    past_step: int,
    current_step: int,
    past_epoch: int,
    current_epoch: int,
    lag: int,
    past_few_blocks: Dict[BlockKey, torch.Tensor],
    past_few_norm_sq: Dict[BlockKey, float],
    current_few_blocks: Dict[BlockKey, torch.Tensor],
    current_ref_blocks: Dict[BlockKey, torch.Tensor],
    current_few_norm_sq: Dict[BlockKey, float],
    current_ref_norm_sq: Dict[BlockKey, float],
    args: argparse.Namespace,
    few_size: int,
    ref_size: int,
    past_few_loss: float,
    past_ref_loss: float,
    current_few_loss: float,
    current_ref_loss: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return encoder rows and block rows for one lag comparison t -> t+Δ."""
    enc_acc = {
        "vision": {"dot_ff": 0.0, "dot_fF": 0.0, "norm_past": 0.0, "norm_curr_few": 0.0, "norm_curr_ref": 0.0, "numel": 0},
        "text": {"dot_ff": 0.0, "dot_fF": 0.0, "norm_past": 0.0, "norm_curr_few": 0.0, "norm_curr_ref": 0.0, "numel": 0},
    }
    block_rows: List[Dict[str, object]] = []

    for (enc, blk), past_vec_stored in sorted(past_few_blocks.items(), key=lambda x: (x[0][0], x[0][1])):
        curr_few = current_few_blocks[(enc, blk)]
        curr_ref = current_ref_blocks[(enc, blk)]
        past_vec = past_vec_stored.float()

        dot_ff = float(torch.dot(past_vec, curr_few).item())
        dot_fF = float(torch.dot(past_vec, curr_ref).item())
        past_sq = float(past_few_norm_sq[(enc, blk)])
        curr_few_sq = float(current_few_norm_sq[(enc, blk)])
        curr_ref_sq = float(current_ref_norm_sq[(enc, blk)])

        cos_ff = cosine_from_sums(dot_ff, past_sq, curr_few_sq)
        cos_fF = cosine_from_sums(dot_fF, past_sq, curr_ref_sq)

        enc_acc[enc]["dot_ff"] += dot_ff
        enc_acc[enc]["dot_fF"] += dot_fF
        enc_acc[enc]["norm_past"] += past_sq
        enc_acc[enc]["norm_curr_few"] += curr_few_sq
        enc_acc[enc]["norm_curr_ref"] += curr_ref_sq
        enc_acc[enc]["numel"] += int(curr_few.numel())

        block_rows.append({
            "step_t": int(past_step),
            "step_t_plus_lag": int(current_step),
            "epoch_t": int(past_epoch),
            "epoch_t_plus_lag": int(current_epoch),
            "lag": int(lag),
            "scope": "block",
            "encoder": enc,
            "block": int(blk),
            "numel": int(curr_few.numel()),
            "few_size": int(few_size),
            "ref_size": int(ref_size),
            "few_loss_t": float(past_few_loss),
            "ref_loss_t": float(past_ref_loss),
            "few_loss_t_plus_lag": float(current_few_loss),
            "ref_loss_t_plus_lag": float(current_ref_loss),
            "traj_optimizer": args.traj_optimizer,
            "dataset": args.dataset,
            "shots": int(args.shots),
            "ref_mode": args.ref_mode,
            "ref_shots": int(args.ref_shots),
            "corr_few_t_to_few_t_plus_lag": cos_ff,
            "corr_few_t_to_full_t_plus_lag": cos_fF,
            "few_grad_l2_t": math.sqrt(max(past_sq, 0.0)),
            "few_grad_l2_t_plus_lag": math.sqrt(max(curr_few_sq, 0.0)),
            "full_grad_l2_t_plus_lag": math.sqrt(max(curr_ref_sq, 0.0)),
        })

    encoder_rows: List[Dict[str, object]] = []
    for enc in ["vision", "text"]:
        if enc_acc[enc]["numel"] <= 0:
            continue
        cos_ff = cosine_from_sums(enc_acc[enc]["dot_ff"], enc_acc[enc]["norm_past"], enc_acc[enc]["norm_curr_few"])
        cos_fF = cosine_from_sums(enc_acc[enc]["dot_fF"], enc_acc[enc]["norm_past"], enc_acc[enc]["norm_curr_ref"])
        encoder_rows.append({
            "step_t": int(past_step),
            "step_t_plus_lag": int(current_step),
            "epoch_t": int(past_epoch),
            "epoch_t_plus_lag": int(current_epoch),
            "lag": int(lag),
            "scope": "encoder",
            "encoder": enc,
            "block": -1,
            "numel": int(enc_acc[enc]["numel"]),
            "few_size": int(few_size),
            "ref_size": int(ref_size),
            "few_loss_t": float(past_few_loss),
            "ref_loss_t": float(past_ref_loss),
            "few_loss_t_plus_lag": float(current_few_loss),
            "ref_loss_t_plus_lag": float(current_ref_loss),
            "traj_optimizer": args.traj_optimizer,
            "dataset": args.dataset,
            "shots": int(args.shots),
            "ref_mode": args.ref_mode,
            "ref_shots": int(args.ref_shots),
            "corr_few_t_to_few_t_plus_lag": cos_ff,
            "corr_few_t_to_full_t_plus_lag": cos_fF,
            "few_grad_l2_t": math.sqrt(max(enc_acc[enc]["norm_past"], 0.0)),
            "few_grad_l2_t_plus_lag": math.sqrt(max(enc_acc[enc]["norm_curr_few"], 0.0)),
            "full_grad_l2_t_plus_lag": math.sqrt(max(enc_acc[enc]["norm_curr_ref"], 0.0)),
        })

    return encoder_rows, block_rows


class TemporalStalenessLogger:
    def __init__(self, out_dir: Path, run_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.out_dir / f"{self.run_name}_temporal_staleness.csv"
        fieldnames = [
            "step_t",
            "step_t_plus_lag",
            "epoch_t",
            "epoch_t_plus_lag",
            "lag",
            "scope",
            "encoder",
            "block",
            "numel",
            "few_size",
            "ref_size",
            "few_loss_t",
            "ref_loss_t",
            "few_loss_t_plus_lag",
            "ref_loss_t_plus_lag",
            "traj_optimizer",
            "dataset",
            "shots",
            "ref_mode",
            "ref_shots",
            "corr_few_t_to_few_t_plus_lag",
            "corr_few_t_to_full_t_plus_lag",
            "few_grad_l2_t",
            "few_grad_l2_t_plus_lag",
            "full_grad_l2_t_plus_lag",
        ]
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()

    def log_rows(self, rows: Iterable[Dict[str, object]]) -> None:
        for row in rows:
            self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger = TemporalStalenessLogger(out_dir=out_dir, run_name=run_name)
    print(f"[GradientStaleness] Logging CSV to: {logger.csv_path}")
    print(f"[GradientStaleness] Run name: {run_name}")
    print(
        f"[GradientStaleness] trajectory={args.traj_optimizer} | full_support_train={args.full_support_train or args.batch_size <= 0} "
        f"| ref_mode={args.ref_mode} | lags={args.lags} | history_dtype={args.history_dtype}"
    )

    hist_dtype = _storage_dtype(args.history_dtype)
    max_lag = max(args.lags)
    history: Deque[Dict[str, object]] = deque(maxlen=max_lag)

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
                    f"[GradientStaleness] step={global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"estimating few-shot/full gradients ..."
                )
                m_few, few_loss = compute_dataset_gradient_only(
                    model=model,
                    loader=few_diag_loader,
                    class_tokens=class_tokens,
                    logit_scale=logit_scale,
                    params=trainable_params,
                    device=args.device,
                    grad_eval_mode=args.diag_grad_eval_mode,
                )
                m_ref, ref_loss = compute_dataset_gradient_only(
                    model=model,
                    loader=ref_diag_loader,
                    class_tokens=class_tokens,
                    logit_scale=logit_scale,
                    params=trainable_params,
                    device=args.device,
                    grad_eval_mode=args.diag_grad_eval_mode,
                )

                current_few_blocks = aggregate_gradients_by_block(gradients=m_few, param_to_block=param_to_block)
                current_ref_blocks = aggregate_gradients_by_block(gradients=m_ref, param_to_block=param_to_block)
                current_few_norm_sq = {bk: float(torch.dot(vec, vec).item()) for bk, vec in current_few_blocks.items()}
                current_ref_norm_sq = {bk: float(torch.dot(vec, vec).item()) for bk, vec in current_ref_blocks.items()}

                # Compare current checkpoint t+Δ to the stored past checkpoints t.
                rows_to_write: List[Dict[str, object]] = []
                for past in history:
                    lag = int(global_step - int(past["step"]))
                    if lag not in args.lags:
                        continue
                    enc_rows, blk_rows = compute_temporal_rows(
                        past_step=int(past["step"]),
                        current_step=int(global_step),
                        past_epoch=int(past["epoch"]),
                        current_epoch=int(epoch),
                        lag=lag,
                        past_few_blocks=past["few_blocks"],
                        past_few_norm_sq=past["few_norm_sq"],
                        current_few_blocks=current_few_blocks,
                        current_ref_blocks=current_ref_blocks,
                        current_few_norm_sq=current_few_norm_sq,
                        current_ref_norm_sq=current_ref_norm_sq,
                        args=args,
                        few_size=len(few_diag_loader.dataset),
                        ref_size=len(ref_diag_loader.dataset),
                        past_few_loss=float(past["few_loss"]),
                        past_ref_loss=float(past["ref_loss"]),
                        current_few_loss=float(few_loss),
                        current_ref_loss=float(ref_loss),
                    )
                    rows_to_write.extend(enc_rows)
                    rows_to_write.extend(blk_rows)

                if rows_to_write:
                    logger.log_rows(rows_to_write)
                diag_count += 1

                # Push current few-shot gradient into the rolling history buffer.
                history.append({
                    "step": int(global_step),
                    "epoch": int(epoch),
                    "few_loss": float(few_loss),
                    "ref_loss": float(ref_loss),
                    "few_blocks": {bk: vec.to(dtype=hist_dtype) for bk, vec in current_few_blocks.items()},
                    "few_norm_sq": current_few_norm_sq,
                })

            optimizer.zero_grad(set_to_none=True)
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            batch_size = int(target.size(0))
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size
            global_step += 1

        mean_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"[Epoch {epoch:02d}] training loss = {mean_epoch_loss:.4f}")

    logger.close()

    if args.eval_final:
        model.eval()
        acc = evaluate_CLIP(model, test_loader, bundle.dataset_obj, args.device)
        print(f"[Final] test accuracy = {acc:.2f}")
        with open(out_dir / f"{run_name}_final_eval.json", "w") as f:
            json.dump({"test_acc": acc}, f, indent=2)



def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

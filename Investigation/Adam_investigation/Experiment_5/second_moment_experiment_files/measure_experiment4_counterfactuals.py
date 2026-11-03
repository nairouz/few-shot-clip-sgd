#!/usr/bin/env python3
r"""Experiment 4 — Frozen-denominator and oracle-denominator counterfactuals.

This experiment answers:

    If you only fix the denominator, how much of Adam can be rescued?

At each logged checkpoint \Theta_t along a chosen trajectory (SGD or Adam), we
compute the same few-shot and reference statistics as Experiment 2:

    m_few, v_few, m_ref, v_ref

and then form the following adaptive directions:

    u_ff   = m_few / sqrt(v_few + eps)
    u_fr   = m_few / sqrt(v_ref + eps)
    u_rr   = m_ref / sqrt(v_ref + eps)
    u_f0   = m_few / sqrt(v_frozen + eps)
    u_flag = m_few / sqrt(v_lag + eps)

where:
  - v_frozen is the *few-shot* denominator frozen at initialization (step 0),
  - v_lag is a lagged few-shot denominator from t - tau.

The script logs two families of outputs:

(1) Counterfactual direction diagnostics
    - cosine to the full reference direction u_rr
    - norm ratios relative to u_rr
    - per-encoder and per-block summaries

(2) Strong-version norm-matched probe steps
    For each counterfactual direction d in {u_ff, u_fr, u_f0, u_flag, u_rr},
    the script applies a temporary, norm-matched one-step perturbation and
    measures the resulting few-shot / reference / test losses.

    By default, all probe directions are matched to the global norm of u_ff,
    which makes the comparison answer the intended causal question directly:

        how much of Adam can be rescued by changing only the denominator,
        under the same adaptive update budget?

This script is intentionally implemented as a *new* measurement file rather
than modifying Experiment 2, so the original pipeline remains untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import Experiment 2 utilities first so its robust torchvision import path is
# executed before importing train.py / clip.py.
from second_moment_experiment_files.measure_second_moment import (  # noqa: E402
    EPS,
    _str2bool,
    build_dataset_bundle,
    build_loaders,
    compute_dataset_gradient_and_second_moment,
    load_clip_model,
    make_optimizer,
    safe_log10_ratio,
    topk_overlap,
    density_to_col,
    tokenize_class_texts,
)
from second_moment_experiment_files.clip_block_utils import BlockKey, build_param_to_block_map  # noqa: E402
from run_utils import set_random_seed  # noqa: E402
from train import evaluate_CLIP, mark_as_trainable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


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
    parser.add_argument("--shots", type=int, default=1)
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
        default="SGD",
        choices=["SGD", "Adam"],
        help="Optimizer used for the adaptation trajectory whose checkpoints are audited.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--full_support_train", action="store_true")
    parser.add_argument("--use_aug", action="store_true")

    parser.add_argument(
        "--ref_mode",
        type=str,
        default="full_train",
        choices=["kshot", "full_train"],
        help="Reference-set construction.",
    )
    parser.add_argument("--ref_shots", type=int, default=16)
    parser.add_argument("--max_ref_per_class", type=int, default=-1)
    parser.add_argument(
        "--repeat_if_needed",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="Repeat images when a class has fewer than K examples (default: True).",
    )

    parser.add_argument(
        "--v_estimator",
        type=str,
        default="batch_mean_square",
        choices=["batch_mean_square", "full_support_square"],
        help="How to estimate the diagonal second moment v_t(D).",
    )
    parser.add_argument("--diag_every", type=int, default=1)
    parser.add_argument("--diag_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--diag_grad_eval_mode",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="If True, compute diagnostics in model.eval() mode (dropout off).",
    )
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    parser.add_argument(
        "--overlap_densities",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated densities rho for top-k overlap of denominators.",
    )
    parser.add_argument("--logratio_min", type=float, default=-12.0)
    parser.add_argument("--logratio_max", type=float, default=12.0)
    parser.add_argument("--logratio_bins", type=int, default=120)

    # Experiment 4 specific arguments
    parser.add_argument(
        "--lag_steps",
        type=int,
        default=5,
        help="Tau for the lagged few-shot denominator v_{t-tau}.",
    )
    parser.add_argument(
        "--probe_alpha",
        type=float,
        default=1.0,
        help=(
            "Global multiplier applied after norm-matching probe directions. "
            "The final probe step is - lr * probe_alpha * d_matched."
        ),
    )
    parser.add_argument(
        "--probe_match_norm_to",
        type=str,
        default="u_ff",
        choices=["u_ff", "u_rr"],
        help="Which counterfactual direction defines the target probe-step norm.",
    )
    parser.add_argument(
        "--probe_eval_test",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="If True, also measure one-step probe loss on the test split.",
    )
    parser.add_argument(
        "--max_diag_steps",
        type=int,
        default=-1,
        help="If > 0, stop logging after this many diagnostic checkpoints.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experiment4_logs",
        help="Directory where the Experiment 4 logs are written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, a descriptive name is generated automatically.",
    )
    parser.add_argument("--eval_final", action="store_true")

    args = parser.parse_args()
    args.overlap_densities = [float(x) for x in args.overlap_densities.split(",") if x.strip()]
    if args.lag_steps < 1:
        raise ValueError("--lag_steps must be >= 1")
    return args


# ============================================================================
# Utility helpers
# ============================================================================


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    nx = torch.norm(x, p=2)
    ny = torch.norm(y, p=2)
    denom = float((nx * ny).item())
    if denom <= eps:
        return float("nan")
    return float(torch.dot(x, y).item() / (denom + eps))



def clone_tensor_dict(x: Dict[torch.nn.Parameter, torch.Tensor]) -> Dict[torch.nn.Parameter, torch.Tensor]:
    return {p: t.clone() for p, t in x.items()}



def flatten_tensor_dict(x: Dict[torch.nn.Parameter, torch.Tensor], *, order: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([x[p].reshape(-1).float().cpu() for p in order], dim=0)



def norm_tensor_dict(x: Dict[torch.nn.Parameter, torch.Tensor], *, order: Sequence[torch.nn.Parameter]) -> float:
    flat = flatten_tensor_dict(x, order=order)
    return float(torch.norm(flat, p=2).item())



def scale_tensor_dict(x: Dict[torch.nn.Parameter, torch.Tensor], scale: float) -> Dict[torch.nn.Parameter, torch.Tensor]:
    return {p: t * scale for p, t in x.items()}



def build_direction_dict(
    m: Dict[torch.nn.Parameter, torch.Tensor],
    v: Dict[torch.nn.Parameter, torch.Tensor],
    eps: float,
) -> Dict[torch.nn.Parameter, torch.Tensor]:
    return {p: m[p] / torch.sqrt(v[p] + eps) for p in m.keys()}



def evaluate_loss_on_loader(
    *,
    model,
    loader: DataLoader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    grad_eval_mode: bool,
) -> float:
    was_training = model.training
    if grad_eval_mode:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(class_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target, reduction="mean")
            bs = int(target.size(0))
            total_loss += float(loss.item()) * bs
            total_n += bs

    model.train(was_training)
    return total_loss / max(total_n, 1)



def apply_delta_inplace(model, delta: Dict[torch.nn.Parameter, torch.Tensor], *, alpha: float) -> None:
    with torch.no_grad():
        for p, d in delta.items():
            p.add_(d.to(device=p.device, dtype=p.dtype), alpha=alpha)



def build_scope_tensors_counterfactual(
    *,
    m_few: Dict[torch.nn.Parameter, torch.Tensor],
    v_few: Dict[torch.nn.Parameter, torch.Tensor],
    m_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_frozen: Dict[torch.nn.Parameter, torch.Tensor],
    v_lag: Optional[Dict[torch.nn.Parameter, torch.Tensor]],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
) -> Tuple[Dict[BlockKey, Dict[str, List[torch.Tensor]]], Dict[str, Dict[str, List[torch.Tensor]]]]:
    by_block: Dict[BlockKey, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    by_encoder: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    for p, mf in m_few.items():
        bk = param_to_block.get(p)
        if bk is None:
            continue
        enc, _ = bk
        tensors = {
            "m_few": mf.reshape(-1),
            "v_few": v_few[p].reshape(-1),
            "m_ref": m_ref[p].reshape(-1),
            "v_ref": v_ref[p].reshape(-1),
            "v_frozen": v_frozen[p].reshape(-1),
        }
        if v_lag is not None:
            tensors["v_lag"] = v_lag[p].reshape(-1)
        for key, value in tensors.items():
            by_block[bk][key].append(value)
            by_encoder[enc][key].append(value)

    return by_block, by_encoder



def safe_percentile_cpu(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    arr = x.detach().contiguous().numpy()
    return float(np.percentile(arr, q * 100.0))



def compute_counterfactual_metric_row(
    *,
    m_few_vec: torch.Tensor,
    v_few_vec: torch.Tensor,
    m_ref_vec: torch.Tensor,
    v_ref_vec: torch.Tensor,
    v_frozen_vec: torch.Tensor,
    v_lag_vec: Optional[torch.Tensor],
    args: argparse.Namespace,
    scope: str,
    encoder: str,
    block: int,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    few_size: int,
    ref_size: int,
    few_loss: float,
    ref_loss: float,
) -> Dict[str, object]:
    m_few_vec = m_few_vec.float().cpu()
    v_few_vec = v_few_vec.float().cpu()
    m_ref_vec = m_ref_vec.float().cpu()
    v_ref_vec = v_ref_vec.float().cpu()
    v_frozen_vec = v_frozen_vec.float().cpu()
    if v_lag_vec is not None:
        v_lag_vec = v_lag_vec.float().cpu()

    numel = int(m_few_vec.numel())
    log_ratio = safe_log10_ratio(v_few_vec, v_ref_vec, args.adam_eps).abs()

    u_ff = m_few_vec / torch.sqrt(v_few_vec + args.adam_eps)
    u_fr = m_few_vec / torch.sqrt(v_ref_vec + args.adam_eps)
    u_rr = m_ref_vec / torch.sqrt(v_ref_vec + args.adam_eps)
    u_f0 = m_few_vec / torch.sqrt(v_frozen_vec + args.adam_eps)
    u_flag = None if v_lag_vec is None else (m_few_vec / torch.sqrt(v_lag_vec + args.adam_eps))

    row: Dict[str, object] = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "scope": scope,
        "encoder": encoder,
        "block": int(block),
        "numel": numel,
        "few_size": int(few_size),
        "ref_size": int(ref_size),
        "few_loss": float(few_loss),
        "ref_loss": float(ref_loss),
        "traj_optimizer": args.traj_optimizer,
        "dataset": args.dataset,
        "shots": int(args.shots),
        "ref_mode": args.ref_mode,
        "ref_shots": int(args.ref_shots),
        "v_estimator": args.v_estimator,
        "lag_steps": int(args.lag_steps),
        "lag_available": v_lag_vec is not None,
        "grad_cosine": cosine_similarity(m_few_vec, m_ref_vec),
        "few_grad_l2": float(torch.norm(m_few_vec, p=2).item()),
        "ref_grad_l2": float(torch.norm(m_ref_vec, p=2).item()),
        "few_v_l2": float(torch.norm(v_few_vec, p=2).item()),
        "ref_v_l2": float(torch.norm(v_ref_vec, p=2).item()),
        "v_mean_abs_log10_ratio": float(torch.mean(log_ratio).item()),
        "v_median_abs_log10_ratio": float(torch.median(log_ratio).item()),
        "v_p90_abs_log10_ratio": safe_percentile_cpu(log_ratio, 0.9),
        "cos_u_ff_u_rr": cosine_similarity(u_ff, u_rr),
        "cos_u_fr_u_rr": cosine_similarity(u_fr, u_rr),
        "cos_u_f0_u_rr": cosine_similarity(u_f0, u_rr),
        "cos_u_ff_u_fr": cosine_similarity(u_ff, u_fr),
        "l2_u_ff": float(torch.norm(u_ff, p=2).item()),
        "l2_u_fr": float(torch.norm(u_fr, p=2).item()),
        "l2_u_f0": float(torch.norm(u_f0, p=2).item()),
        "l2_u_rr": float(torch.norm(u_rr, p=2).item()),
        "normratio_u_ff": float(torch.norm(u_ff, p=2).item() / (torch.norm(u_rr, p=2).item() + EPS)),
        "normratio_u_fr": float(torch.norm(u_fr, p=2).item() / (torch.norm(u_rr, p=2).item() + EPS)),
        "normratio_u_f0": float(torch.norm(u_f0, p=2).item() / (torch.norm(u_rr, p=2).item() + EPS)),
    }
    if u_flag is None:
        row.update({
            "cos_u_flag_u_rr": float("nan"),
            "cos_u_flag_u_fr": float("nan"),
            "l2_u_flag": float("nan"),
            "normratio_u_flag": float("nan"),
        })
    else:
        row.update({
            "cos_u_flag_u_rr": cosine_similarity(u_flag, u_rr),
            "cos_u_flag_u_fr": cosine_similarity(u_flag, u_fr),
            "l2_u_flag": float(torch.norm(u_flag, p=2).item()),
            "normratio_u_flag": float(torch.norm(u_flag, p=2).item() / (torch.norm(u_rr, p=2).item() + EPS)),
        })
    for density in args.overlap_densities:
        row[f"frozen_overlap_{density_to_col(density)}"] = topk_overlap(v_frozen_vec, v_ref_vec, density)
        if v_lag_vec is None:
            row[f"lagged_overlap_{density_to_col(density)}"] = float("nan")
        else:
            row[f"lagged_overlap_{density_to_col(density)}"] = topk_overlap(v_lag_vec, v_ref_vec, density)
    return row



def compute_counterfactual_rows(
    *,
    m_few: Dict[torch.nn.Parameter, torch.Tensor],
    v_few: Dict[torch.nn.Parameter, torch.Tensor],
    m_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_frozen: Dict[torch.nn.Parameter, torch.Tensor],
    v_lag: Optional[Dict[torch.nn.Parameter, torch.Tensor]],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    few_size: int,
    ref_size: int,
    few_loss: float,
    ref_loss: float,
) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    by_block, by_encoder = build_scope_tensors_counterfactual(
        m_few=m_few,
        v_few=v_few,
        m_ref=m_ref,
        v_ref=v_ref,
        v_frozen=v_frozen,
        v_lag=v_lag,
        param_to_block=param_to_block,
    )

    encoder_rows: Dict[str, Dict[str, object]] = {}
    for enc in ["vision", "text"]:
        if enc not in by_encoder:
            continue
        tensors = by_encoder[enc]
        lag_vec = None if "v_lag" not in tensors else torch.cat(tensors["v_lag"], dim=0)
        encoder_rows[enc] = compute_counterfactual_metric_row(
            m_few_vec=torch.cat(tensors["m_few"], dim=0),
            v_few_vec=torch.cat(tensors["v_few"], dim=0),
            m_ref_vec=torch.cat(tensors["m_ref"], dim=0),
            v_ref_vec=torch.cat(tensors["v_ref"], dim=0),
            v_frozen_vec=torch.cat(tensors["v_frozen"], dim=0),
            v_lag_vec=lag_vec,
            args=args,
            scope="encoder",
            encoder=enc,
            block=-1,
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            few_size=few_size,
            ref_size=ref_size,
            few_loss=few_loss,
            ref_loss=ref_loss,
        )

    block_rows: List[Dict[str, object]] = []
    for (enc, block), tensors in sorted(by_block.items(), key=lambda x: (x[0][0], x[0][1])):
        lag_vec = None if "v_lag" not in tensors else torch.cat(tensors["v_lag"], dim=0)
        row = compute_counterfactual_metric_row(
            m_few_vec=torch.cat(tensors["m_few"], dim=0),
            v_few_vec=torch.cat(tensors["v_few"], dim=0),
            m_ref_vec=torch.cat(tensors["m_ref"], dim=0),
            v_ref_vec=torch.cat(tensors["v_ref"], dim=0),
            v_frozen_vec=torch.cat(tensors["v_frozen"], dim=0),
            v_lag_vec=lag_vec,
            args=args,
            scope="block",
            encoder=enc,
            block=block,
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            few_size=few_size,
            ref_size=ref_size,
            few_loss=few_loss,
            ref_loss=ref_loss,
        )
        block_rows.append(row)

    return encoder_rows, block_rows



def evaluate_probe_bundle(
    *,
    model,
    directions: Dict[str, Dict[torch.nn.Parameter, torch.Tensor]],
    order: Sequence[torch.nn.Parameter],
    class_tokens: torch.Tensor,
    logit_scale: float,
    few_loader: DataLoader,
    ref_loader: DataLoader,
    test_loader: Optional[DataLoader],
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    base_few_loss: float,
    base_ref_loss: float,
    base_test_loss: Optional[float],
) -> Dict[str, object]:
    if args.probe_match_norm_to not in directions:
        raise KeyError(f"Unknown probe-match direction: {args.probe_match_norm_to}")

    target_norm = norm_tensor_dict(directions[args.probe_match_norm_to], order=order)
    row: Dict[str, object] = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "traj_optimizer": args.traj_optimizer,
        "dataset": args.dataset,
        "shots": int(args.shots),
        "ref_mode": args.ref_mode,
        "ref_shots": int(args.ref_shots),
        "v_estimator": args.v_estimator,
        "lag_steps": int(args.lag_steps),
        "probe_alpha": float(args.probe_alpha),
        "probe_match_norm_to": args.probe_match_norm_to,
        "probe_eval_test": bool(args.probe_eval_test),
        "base_few_loss": float(base_few_loss),
        "base_ref_loss": float(base_ref_loss),
        "base_test_loss": float(base_test_loss) if base_test_loss is not None else float("nan"),
    }

    for name, d in directions.items():
        d_norm = norm_tensor_dict(d, order=order)
        if (not math.isfinite(d_norm)) or d_norm <= EPS or (not math.isfinite(target_norm)) or target_norm <= EPS:
            scale = float("nan")
            few_loss = float("nan")
            ref_loss = float("nan")
            test_loss = float("nan")
        else:
            scale = target_norm / d_norm
            d_scaled = scale_tensor_dict(d, scale)
            alpha = -args.lr * args.probe_alpha
            apply_delta_inplace(model, d_scaled, alpha=alpha)
            try:
                few_loss = evaluate_loss_on_loader(
                    model=model,
                    loader=few_loader,
                    class_tokens=class_tokens,
                    logit_scale=logit_scale,
                    device=args.device,
                    grad_eval_mode=args.diag_grad_eval_mode,
                )
                ref_loss = evaluate_loss_on_loader(
                    model=model,
                    loader=ref_loader,
                    class_tokens=class_tokens,
                    logit_scale=logit_scale,
                    device=args.device,
                    grad_eval_mode=args.diag_grad_eval_mode,
                )
                if args.probe_eval_test and test_loader is not None:
                    test_loss = evaluate_loss_on_loader(
                        model=model,
                        loader=test_loader,
                        class_tokens=class_tokens,
                        logit_scale=logit_scale,
                        device=args.device,
                        grad_eval_mode=args.diag_grad_eval_mode,
                    )
                else:
                    test_loss = float("nan")
            finally:
                apply_delta_inplace(model, d_scaled, alpha=-alpha)

        row[f"scale_{name}"] = float(scale) if math.isfinite(scale) else float("nan")
        row[f"few_loss_{name}"] = float(few_loss)
        row[f"ref_loss_{name}"] = float(ref_loss)
        row[f"test_loss_{name}"] = float(test_loss)
        row[f"delta_few_loss_{name}"] = float(few_loss - base_few_loss) if math.isfinite(few_loss) else float("nan")
        row[f"delta_ref_loss_{name}"] = float(ref_loss - base_ref_loss) if math.isfinite(ref_loss) else float("nan")
        row[f"delta_test_loss_{name}"] = float(test_loss - base_test_loss) if (base_test_loss is not None and math.isfinite(test_loss)) else float("nan")

    return row



def default_run_name(args: argparse.Namespace) -> str:
    bb = args.backbone.replace("/", "-")
    ref_tag = "full" if args.ref_mode == "full_train" else f"kshot_{args.ref_shots}"
    vest = args.v_estimator.replace("_", "-")
    return (
        f"experiment4_dataset_{args.dataset}_noisyshots_{args.shots}_ref_{ref_tag}"
        f"_backbone_{bb}_trajopt_{args.traj_optimizer.lower()}_seed_{args.seed}"
        f"_pos_{args.position}_enc_{args.encoder}_vest_{vest}_lag_{args.lag_steps}"
    )


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

    optimizer = make_optimizer(args.traj_optimizer, trainable_params, args.lr, args.adam_eps)
    param_to_block = build_param_to_block_map(model, only_trainable=True)
    class_tokens = tokenize_class_texts(bundle.classnames, bundle.template, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)

    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # CSV writers
    counterfactual_fieldnames = [
        "global_step", "epoch", "step_in_epoch", "scope", "encoder", "block",
        "numel", "few_size", "ref_size", "few_loss", "ref_loss", "traj_optimizer",
        "dataset", "shots", "ref_mode", "ref_shots", "v_estimator", "lag_steps",
        "lag_available", "grad_cosine", "few_grad_l2", "ref_grad_l2", "few_v_l2",
        "ref_v_l2", "v_mean_abs_log10_ratio", "v_median_abs_log10_ratio",
        "v_p90_abs_log10_ratio", "cos_u_ff_u_rr", "cos_u_fr_u_rr", "cos_u_f0_u_rr",
        "cos_u_flag_u_rr", "cos_u_ff_u_fr", "cos_u_flag_u_fr", "l2_u_ff", "l2_u_fr",
        "l2_u_f0", "l2_u_flag", "l2_u_rr", "normratio_u_ff", "normratio_u_fr",
        "normratio_u_f0", "normratio_u_flag",
    ] + [
        f"frozen_overlap_{density_to_col(d)}" for d in args.overlap_densities
    ] + [
        f"lagged_overlap_{density_to_col(d)}" for d in args.overlap_densities
    ]

    probe_names = ["u_ff", "u_fr", "u_f0", "u_flag", "u_rr"]
    probe_fieldnames = [
        "global_step", "epoch", "step_in_epoch", "traj_optimizer", "dataset", "shots",
        "ref_mode", "ref_shots", "v_estimator", "lag_steps", "probe_alpha",
        "probe_match_norm_to", "probe_eval_test", "base_few_loss", "base_ref_loss",
        "base_test_loss",
    ]
    for name in probe_names:
        probe_fieldnames += [
            f"scale_{name}", f"few_loss_{name}", f"ref_loss_{name}", f"test_loss_{name}",
            f"delta_few_loss_{name}", f"delta_ref_loss_{name}", f"delta_test_loss_{name}",
        ]

    cf_iter_path = out_dir / f"{run_name}_counterfactual_per_iteration.csv"
    cf_block_path = out_dir / f"{run_name}_counterfactual_per_block.csv"
    probe_path = out_dir / f"{run_name}_probe_per_iteration.csv"

    with open(cf_iter_path, "w", newline="") as f_iter, \
         open(cf_block_path, "w", newline="") as f_block, \
         open(probe_path, "w", newline="") as f_probe:

        iter_writer = csv.DictWriter(f_iter, fieldnames=counterfactual_fieldnames)
        block_writer = csv.DictWriter(f_block, fieldnames=counterfactual_fieldnames)
        probe_writer = csv.DictWriter(f_probe, fieldnames=probe_fieldnames)
        iter_writer.writeheader()
        block_writer.writeheader()
        probe_writer.writeheader()

        print(f"[Exp4] Logging bundle to: {out_dir}")
        print(f"[Exp4] Run name: {run_name}")
        print(
            f"[Exp4] trajectory={args.traj_optimizer} | lag_steps={args.lag_steps} | "
            f"probe_match_norm_to={args.probe_match_norm_to} | ref_mode={args.ref_mode}"
        )

        global_step = 0
        diag_count = 0
        v_frozen: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None
        v_history: deque[Dict[torch.nn.Parameter, torch.Tensor]] = deque()

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
                        f"[Exp4] step={global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                        f"estimating few-shot/reference moments and counterfactuals ..."
                    )
                    m_few, v_few, few_loss = compute_dataset_gradient_and_second_moment(
                        model=model,
                        loader=few_diag_loader,
                        class_tokens=class_tokens,
                        logit_scale=logit_scale,
                        params=trainable_params,
                        device=args.device,
                        grad_eval_mode=args.diag_grad_eval_mode,
                        v_estimator=args.v_estimator,
                        adam_eps=args.adam_eps,
                    )
                    m_ref, v_ref, ref_loss = compute_dataset_gradient_and_second_moment(
                        model=model,
                        loader=ref_diag_loader,
                        class_tokens=class_tokens,
                        logit_scale=logit_scale,
                        params=trainable_params,
                        device=args.device,
                        grad_eval_mode=args.diag_grad_eval_mode,
                        v_estimator=args.v_estimator,
                        adam_eps=args.adam_eps,
                    )

                    if v_frozen is None:
                        v_frozen = clone_tensor_dict(v_few)

                    v_lag = None
                    if len(v_history) >= args.lag_steps:
                        v_lag = clone_tensor_dict(v_history[-args.lag_steps])

                    encoder_rows, block_rows = compute_counterfactual_rows(
                        m_few=m_few,
                        v_few=v_few,
                        m_ref=m_ref,
                        v_ref=v_ref,
                        v_frozen=v_frozen,
                        v_lag=v_lag,
                        param_to_block=param_to_block,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        few_size=len(few_diag_loader.dataset),
                        ref_size=len(ref_diag_loader.dataset),
                        few_loss=few_loss,
                        ref_loss=ref_loss,
                    )
                    for row in encoder_rows.values():
                        iter_writer.writerow(row)
                    for row in block_rows:
                        block_writer.writerow(row)
                    f_iter.flush()
                    f_block.flush()

                    # Build global directions for the strong-version one-step probe.
                    directions: Dict[str, Dict[torch.nn.Parameter, torch.Tensor]] = {
                        "u_ff": build_direction_dict(m_few, v_few, args.adam_eps),
                        "u_fr": build_direction_dict(m_few, v_ref, args.adam_eps),
                        "u_f0": build_direction_dict(m_few, v_frozen, args.adam_eps),
                        "u_rr": build_direction_dict(m_ref, v_ref, args.adam_eps),
                    }
                    if v_lag is not None:
                        directions["u_flag"] = build_direction_dict(m_few, v_lag, args.adam_eps)
                    else:
                        directions["u_flag"] = {p: torch.full_like(m_few[p], float("nan")) for p in m_few.keys()}

                    base_test_loss = None
                    if args.probe_eval_test and test_loader is not None:
                        base_test_loss = evaluate_loss_on_loader(
                            model=model,
                            loader=test_loader,
                            class_tokens=class_tokens,
                            logit_scale=logit_scale,
                            device=args.device,
                            grad_eval_mode=args.diag_grad_eval_mode,
                        )

                    probe_row = evaluate_probe_bundle(
                        model=model,
                        directions=directions,
                        order=trainable_params,
                        class_tokens=class_tokens,
                        logit_scale=logit_scale,
                        few_loader=few_diag_loader,
                        ref_loader=ref_diag_loader,
                        test_loader=test_loader,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        base_few_loss=few_loss,
                        base_ref_loss=ref_loss,
                        base_test_loss=base_test_loss,
                    )
                    probe_writer.writerow(probe_row)
                    f_probe.flush()

                    diag_count += 1
                    v_history.append(clone_tensor_dict(v_few))
                    if len(v_history) > max(args.lag_steps, 1) + 1:
                        v_history.popleft()

                optimizer.zero_grad(set_to_none=True)
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = model.encode_text(class_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()

                bs = int(target.size(0))
                epoch_loss += float(loss.item()) * bs
                epoch_samples += bs
                global_step += 1

            mean_epoch_loss = epoch_loss / max(epoch_samples, 1)
            print(f"[Epoch {epoch:02d}] training loss = {mean_epoch_loss:.4f}")

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

#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment I — Feature-drift landscape along an SGD trajectory.

Goal
----
Measure whether SGD stays in a low-drift region of CLIP feature space or moves
into a narrow support-fitting basin during few-shot adaptation.

Protocol implemented here
-------------------------
- CLIP ViT-B/16 by default (but backbone is configurable).
- 1-shot adaptation by default.
- Full-support training: one batch per epoch when --full_support_train is set.
- No augmentation by default (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Save checkpoints at t in {0,1,2,5,10,20,30} by default.
- Evaluate, at each checkpoint:
    * support / full-train / test loss and accuracy,
    * relative trainable-parameter drift from initialization,
    * per-encoder and per-block parameter drift,
    * image embedding drift on support / full-train / test,
    * text embedding drift,
    * class-margin drift on support / full-train / test.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/<run_name>_block_drift.csv
<out_dir>/<run_name>_summary.json
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt

The companion plotting script is:
  loss_landscape_experiment_files/plot_experimentI_feature_drift.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import torch
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader


def _import_torchvision_with_optional_nms_stub() -> None:
    """Import torchvision safely across environments.

    Some environments ship a torchvision build without the custom NMS operator,
    which causes imports inside OpenAI CLIP's preprocessing utilities to fail.
    We first try a normal import; only if the specific missing-NMS error occurs
    do we define a minimal schema stub and retry.
    """
    try:
        importlib.import_module("torchvision")
        importlib.import_module("torchvision.transforms")
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
        importlib.import_module("torchvision.transforms")


_import_torchvision_with_optional_nms_stub()

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
        help="Experiment I is defined for the SGD trajectory.",
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
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size used for full-train / test metric evaluation.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/experimentI_logs",
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
    return args


def load_clip_model(backbone: str, device: str):
    import getpass

    cache = os.getenv("CLIP_DOWNLOAD_ROOT")
    model, preprocess = clip.load(backbone, device=device, download_root=cache)
    return model, preprocess


def build_loader_from_source(
    data_source,
    *,
    batch_size: int,
    preprocess,
    is_train: bool,
    shuffle: bool,
    num_workers: int,
):
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


def maybe_full_support_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)


def compute_text_features(model, class_tokens: torch.Tensor, device: str) -> torch.Tensor:
    amp_enabled = str(device).startswith("cuda")
    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=amp_enabled):
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


def compute_loader_loss_acc(model, loader, text_features: torch.Tensor, logit_scale: float, device: str) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    amp_enabled = str(device).startswith("cuda")
    with torch.no_grad():
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


def _compute_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, target[:, None]).squeeze(1)
    masked = logits.clone()
    masked[torch.arange(logits.size(0), device=logits.device), target] = -float("inf")
    wrong_max = masked.max(dim=1).values
    return true_logits - wrong_max


def compute_feature_metrics(
    model,
    init_model,
    loader,
    text_features_cur: torch.Tensor,
    text_features_init: torch.Tensor,
    logit_scale: float,
    device: str,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    init_model.eval()
    amp_enabled = str(device).startswith("cuda")

    total_items = 0
    image_drift_sum = 0.0
    margin_drift_sum = 0.0
    margin_cur_sum = 0.0
    logit_mae_sum = 0.0

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            bs = int(target.size(0))
            with autocast("cuda", enabled=amp_enabled):
                img_cur = model.encode_image(images)
                img_cur = img_cur / img_cur.norm(dim=-1, keepdim=True)
                img_init = init_model.encode_image(images)
                img_init = img_init / img_init.norm(dim=-1, keepdim=True)

                logits_cur = logit_scale * (img_cur @ text_features_cur.t())
                logits_init = logit_scale * (img_init @ text_features_init.t())

            cos = (img_cur * img_init).sum(dim=-1).clamp(-1.0, 1.0)
            image_drift_sum += float((1.0 - cos).sum().item())

            margin_cur = _compute_margin(logits_cur, target)
            margin_init = _compute_margin(logits_init, target)
            margin_drift_sum += float((margin_cur - margin_init).abs().sum().item())
            margin_cur_sum += float(margin_cur.sum().item())
            logit_mae_sum += float((logits_cur - logits_init).abs().mean(dim=1).sum().item())
            total_items += bs

    model.train(was_training)
    if total_items == 0:
        return {
            "image_drift": float("nan"),
            "margin_drift": float("nan"),
            "margin_current": float("nan"),
            "logit_mae": float("nan"),
        }
    return {
        "image_drift": image_drift_sum / total_items,
        "margin_drift": margin_drift_sum / total_items,
        "margin_current": margin_cur_sum / total_items,
        "logit_mae": logit_mae_sum / total_items,
    }


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


def infer_encoder_and_block(name: str) -> Tuple[str, int]:
    if name.startswith("visual.transformer.resblocks."):
        parts = name.split(".")
        return "vision", int(parts[3])
    if name.startswith("transformer.resblocks."):
        parts = name.split(".")
        return "text", int(parts[2])
    return "other", -1


def _group_sqnorm(state: MutableMapping[str, torch.Tensor], names: Iterable[str]) -> float:
    total = 0.0
    for k in names:
        t = state[k].double()
        total += float((t * t).sum().item())
    return total


def _group_reldrift(
    current: MutableMapping[str, torch.Tensor],
    init: MutableMapping[str, torch.Tensor],
    names: Iterable[str],
) -> float:
    num = 0.0
    den = 0.0
    for k in names:
        c = current[k].double()
        i = init[k].double()
        d = c - i
        num += float((d * d).sum().item())
        den += float((i * i).sum().item())
    if den <= 0.0:
        return 0.0
    return math.sqrt(num / den)


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xv = torch.tensor(list(x), dtype=torch.float64)
    yv = torch.tensor(list(y), dtype=torch.float64)
    if float(xv.std().item()) == 0.0 or float(yv.std().item()) == 0.0:
        return float("nan")
    xm = xv - xv.mean()
    ym = yv - yv.mean()
    return float((xm * ym).mean().item() / max(xv.std(unbiased=False).item() * yv.std(unbiased=False).item(), 1e-32))


def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = load_clip_model(args.backbone, args.device)
    init_model, _ = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    if args.run_name is None:
        bb = args.backbone.replace("/", "-")
        run_name = (
            f"experimentI_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_sgd_"
            f"seed_{args.seed}_pos_{args.position}_enc_{args.encoder}_fullsupport_{str(args.full_support_train).lower()}"
        )
    else:
        run_name = args.run_name

    ckpt_dir = out_dir / "checkpoints" / run_name

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

    model = model.to(args.device)
    init_model = init_model.to(args.device)
    model.eval()
    init_model.eval()

    mark_as_trainable(model, args)
    # Freeze the reference pretrained model completely.
    for p in init_model.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    optimizer = SGD(trainable_params, lr=args.lr)
    scaler = GradScaler("cuda") if str(args.device).startswith("cuda") else GradScaler("cpu")

    init_state = get_trainable_state(model)
    group_names: Dict[str, List[str]] = {"all": list(init_state.keys()), "vision": [], "text": []}
    block_groups: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for name in init_state.keys():
        enc, blk = infer_encoder_and_block(name)
        if enc in ("vision", "text"):
            group_names[enc].append(name)
            block_groups[(enc, blk)].append(name)

    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)
    text_init = compute_text_features(init_model, class_tokens, args.device)

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
    block_rows: List[Dict[str, object]] = []

    def eval_and_save(epoch: int, state: MutableMapping[str, torch.Tensor]) -> None:
        set_trainable_state(model, state)
        text_cur = compute_text_features(model, class_tokens, args.device)
        text_cos = (text_cur * text_init).sum(dim=-1).clamp(-1.0, 1.0)
        text_drift = float((1.0 - text_cos.mean()).item())

        few_loss, few_acc = compute_loader_loss_acc(model, few_eval_loader, text_cur, logit_scale, args.device)
        full_loss, full_acc = compute_loader_loss_acc(model, full_eval_loader, text_cur, logit_scale, args.device)
        test_loss, test_acc = compute_loader_loss_acc(model, test_loader, text_cur, logit_scale, args.device)

        few_feat = compute_feature_metrics(model, init_model, few_eval_loader, text_cur, text_init, logit_scale, args.device)
        full_feat = compute_feature_metrics(model, init_model, full_eval_loader, text_cur, text_init, logit_scale, args.device)
        test_feat = compute_feature_metrics(model, init_model, test_loader, text_cur, text_init, logit_scale, args.device)

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
                "gap_full_minus_few_loss": float(full_loss - few_loss),
                "gap_test_minus_few_loss": float(test_loss - few_loss),
                "gap_few_minus_full_acc": float(few_acc - full_acc),
                "gap_few_minus_test_acc": float(few_acc - test_acc),
                "param_drift_all": _group_reldrift(state, init_state, group_names["all"]),
                "param_drift_vision": _group_reldrift(state, init_state, group_names["vision"]),
                "param_drift_text": _group_reldrift(state, init_state, group_names["text"]),
                "image_drift_few": float(few_feat["image_drift"]),
                "image_drift_full": float(full_feat["image_drift"]),
                "image_drift_test": float(test_feat["image_drift"]),
                "text_drift": float(text_drift),
                "margin_drift_few": float(few_feat["margin_drift"]),
                "margin_drift_full": float(full_feat["margin_drift"]),
                "margin_drift_test": float(test_feat["margin_drift"]),
                "margin_current_few": float(few_feat["margin_current"]),
                "margin_current_full": float(full_feat["margin_current"]),
                "margin_current_test": float(test_feat["margin_current"]),
                "logit_mae_few": float(few_feat["logit_mae"]),
                "logit_mae_full": float(full_feat["logit_mae"]),
                "logit_mae_test": float(test_feat["logit_mae"]),
                "checkpoint_path": str(ckpt_path),
            }
        )

        for (enc, blk), names in sorted(block_groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            block_rows.append(
                {
                    "epoch": int(epoch),
                    "encoder": enc,
                    "block": int(blk),
                    "numel": int(sum(state[n].numel() for n in names)),
                    "rel_param_drift": _group_reldrift(state, init_state, names),
                }
            )

        print(
            f"[ExperimentI] epoch={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | "
            f"test_loss={test_loss:.4f} | param_drift={trajectory_rows[-1]['param_drift_all']:.4f} | "
            f"img_full_drift={trajectory_rows[-1]['image_drift_full']:.4f} | text_drift={text_drift:.4f}"
        )

    # Save / evaluate t = 0.
    eval_and_save(0, init_state)

    start_time = time.perf_counter()
    amp_enabled = str(args.device).startswith("cuda")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for images, target in few_train_loader:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = model.encode_text(class_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * (image_features @ text_features.t())
                loss = F.cross_entropy(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        if epoch in args.checkpoint_epochs:
            eval_and_save(epoch, get_trainable_state(model))

    elapsed = time.perf_counter() - start_time
    print(f"[ExperimentI] Training finished in {elapsed:.1f}s")

    # Write CSV bundles.
    import csv

    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with open(out_dir / f"{run_name}_block_drift.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    # Summary metrics.
    epochs = [float(r["epoch"]) for r in trajectory_rows]
    param_all = [float(r["param_drift_all"]) for r in trajectory_rows]
    full_losses = [float(r["full_loss"]) for r in trajectory_rows]
    test_losses = [float(r["test_loss"]) for r in trajectory_rows]
    test_accs = [float(r["test_acc"]) for r in trajectory_rows]
    img_full = [float(r["image_drift_full"]) for r in trajectory_rows]
    text_drift = [float(r["text_drift"]) for r in trajectory_rows]

    idx_best_full = int(min(range(len(full_losses)), key=lambda i: full_losses[i]))
    idx_best_test = int(max(range(len(test_accs)), key=lambda i: test_accs[i]))

    summary = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "optimizer": args.optimizer,
        "checkpoint_epochs": args.checkpoint_epochs,
        "best_full_loss": {
            "epoch": int(trajectory_rows[idx_best_full]["epoch"]),
            "full_loss": float(trajectory_rows[idx_best_full]["full_loss"]),
            "test_acc": float(trajectory_rows[idx_best_full]["test_acc"]),
            "param_drift_all": float(trajectory_rows[idx_best_full]["param_drift_all"]),
            "image_drift_full": float(trajectory_rows[idx_best_full]["image_drift_full"]),
            "text_drift": float(trajectory_rows[idx_best_full]["text_drift"]),
        },
        "best_test_acc": {
            "epoch": int(trajectory_rows[idx_best_test]["epoch"]),
            "test_acc": float(trajectory_rows[idx_best_test]["test_acc"]),
            "full_loss": float(trajectory_rows[idx_best_test]["full_loss"]),
            "param_drift_all": float(trajectory_rows[idx_best_test]["param_drift_all"]),
            "image_drift_test": float(trajectory_rows[idx_best_test]["image_drift_test"]),
            "text_drift": float(trajectory_rows[idx_best_test]["text_drift"]),
        },
        "corr_param_drift_all_vs_full_loss": safe_corr(param_all, full_losses),
        "corr_param_drift_all_vs_test_loss": safe_corr(param_all, test_losses),
        "corr_param_drift_all_vs_test_acc": safe_corr(param_all, test_accs),
        "corr_image_drift_full_vs_full_loss": safe_corr(img_full, full_losses),
        "corr_text_drift_vs_full_loss": safe_corr(text_drift, full_losses),
        "corr_epoch_vs_param_drift_all": safe_corr(epochs, param_all),
        "final_checkpoint": trajectory_rows[-1],
    }

    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ExperimentI] Wrote logs to: {out_dir}")
    print(f"[ExperimentI] Run name: {run_name}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

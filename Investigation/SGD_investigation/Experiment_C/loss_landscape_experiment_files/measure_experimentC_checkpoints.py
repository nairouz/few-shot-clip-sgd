#!/usr/bin/env python3
"""Experiment C — training pass to save SGD checkpoints.

This script runs the exact SGD trajectory used for the 2D local-surface study and
saves trainable checkpoints at t in {0,1,2,5,10,20,30} by default.

Protocol
--------
- CLIP ViT-B/16 by default.
- 1-shot adaptation by default.
- Full-support training when --full_support_train is set.
- No augmentation (geometry experiment).
- Train both encoders, all selected blocks trainable.
- Plain SGD, one update per epoch under full-support training.

Outputs
-------
<out_dir>/<run_name>_config.json
<out_dir>/<run_name>_trajectory.csv
<out_dir>/checkpoints/<run_name>/epoch_XX_trainable.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import SGD

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_landscape_experiment_files.experimentC_utils import (  # noqa: E402
    DEFAULT_CHECKPOINT_EPOCHS,
    compute_loader_loss_acc,
    get_trainable_state,
    make_default_run_name,
    prepare_common_experiment,
    save_trainable_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root_path", type=str, default="~/datasets")
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--position", type=str, default="all", choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"])
    parser.add_argument("--encoder", type=str, default="both", choices=["vision", "text", "both"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64, help="Ignored when --full_support_train is set.")
    parser.add_argument("--full_support_train", action="store_true")
    parser.add_argument("--checkpoint_epochs", type=str, default=",".join(str(x) for x in DEFAULT_CHECKPOINT_EPOCHS))
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="./results_without_aug/experimentC_logs")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    args.checkpoint_epochs = sorted({int(x) for x in args.checkpoint_epochs.split(",") if x.strip()})
    if 0 not in args.checkpoint_epochs:
        args.checkpoint_epochs = [0] + args.checkpoint_epochs
    if args.max_epochs not in args.checkpoint_epochs:
        args.checkpoint_epochs.append(args.max_epochs)
    args.checkpoint_epochs = sorted(set(args.checkpoint_epochs))
    return args


def run_experiment(args: argparse.Namespace) -> None:
    bundle = prepare_common_experiment(args)
    model = bundle["clip_model"]
    class_tokens = bundle["class_tokens"]
    logit_scale = bundle["logit_scale"]
    few_train_loader = bundle["few_train_loader"]
    few_eval_loader = bundle["few_eval_loader"]
    full_eval_loader = bundle["full_eval_loader"]
    test_loader = bundle["test_loader"]
    few_dataset = bundle["few_dataset"]
    full_dataset = bundle["full_dataset"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or make_default_run_name(
        prefix="experimentC",
        dataset=args.dataset,
        shots=args.shots,
        backbone=args.backbone,
        seed=args.seed,
        position=args.position,
        encoder=args.encoder,
        full_support_train=args.full_support_train,
    )
    ckpt_dir = out_dir / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        "no_augmentation": True,
        "logit_scale": logit_scale,
    }
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(trainable_params, lr=args.lr)
    amp_enabled = str(args.device).startswith("cuda")
    scaler = GradScaler("cuda") if amp_enabled else GradScaler("cpu")

    rows = []

    def eval_and_save(epoch: int) -> None:
        state = get_trainable_state(model)
        few_loss, few_acc = compute_loader_loss_acc(model, few_eval_loader, class_tokens, logit_scale, args.device)
        full_loss, full_acc = compute_loader_loss_acc(model, full_eval_loader, class_tokens, logit_scale, args.device)
        test_loss, test_acc = compute_loader_loss_acc(model, test_loader, class_tokens, logit_scale, args.device)
        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_trainable.pt"
        save_trainable_checkpoint(
            ckpt_path,
            state,
            {
                "epoch": int(epoch),
                "dataset": args.dataset,
                "shots": args.shots,
                "backbone": args.backbone,
                "optimizer": "SGD",
                "position": args.position,
                "encoder": args.encoder,
            },
        )
        rows.append(
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
            f"[ExperimentC-ckpt] epoch={epoch:02d} | few_loss={few_loss:.4f} | full_loss={full_loss:.4f} | "
            f"test_loss={test_loss:.4f} | few_acc={few_acc:.2f} | test_acc={test_acc:.2f}"
        )

    eval_and_save(0)
    start_time = time.perf_counter()

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
            eval_and_save(epoch)

    elapsed = time.perf_counter() - start_time
    with open(out_dir / f"{run_name}_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ExperimentC-ckpt] Training finished in {elapsed:.1f}s")
    print(f"[ExperimentC-ckpt] Logs written to: {out_dir}")
    print(f"[ExperimentC-ckpt] Run name: {run_name}")


if __name__ == "__main__":
    run_experiment(parse_args())

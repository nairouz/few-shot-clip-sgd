"""Command-line and reproducibility utilities for NeurIPS 2026 P5."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
DATASETS = ("eurosat", "fgvc", "dtd", "oxford_pets")
SHOTS = (1, 4, 16)
SEEDS = (1, 2, 3)
EVAL_EPOCHS = (0, 1, 2, 5, 10, 20, 30)
DIAGNOSTIC_EPOCHS = (0, 10, 20, 30)
MECHANISM_DATASETS = ("eurosat", "dtd")
MECHANISM_SHOTS = (1, 16)


def variant_from_directory(
    path: Optional[Union[str, os.PathLike]] = None,
) -> str:
    resolved = Path(path or __file__).resolve()
    name = resolved.name if resolved.is_dir() else resolved.parent.name
    variants = {
        "P5_ZeroShot": "zero-shot",
        "P5_DenseSGD": "sgd",
        "P5_Adam": "adam",
        "P5_AdamM0": "adam-m0",
    }
    try:
        return variants[name]
    except KeyError as exc:
        raise RuntimeError(f"Unrecognized standalone P5 directory: {name}") from exc


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader_seed(dataset: str, shots: int, seed: int) -> int:
    """Stable, variant-independent seed for sample order and worker RNG."""
    payload = f"p5-loader-v1|{dataset}|{shots}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def diagnostic_seed(dataset: str, shots: int, seed: int) -> int:
    payload = f"p5-diagnostic-b16-v1|{dataset}|{shots}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def mechanism_cell(dataset: str, shots: int) -> bool:
    return dataset in MECHANISM_DATASETS and shots in MECHANISM_SHOTS


def get_arguments(argv=None):
    variant = variant_from_directory()
    zero_shot = variant == "zero-shot"
    parser = argparse.ArgumentParser(
        description=f"NeurIPS 2026 P5 SigLIP condition: {variant}"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="eurosat")
    parser.add_argument("--shots", type=int, choices=(0,) + SHOTS, default=0 if zero_shot else 1)
    parser.add_argument("--seed", type=int, default=0 if zero_shot else 1)
    parser.add_argument("--root_path", required=False, default=os.getenv("DATASETS_ROOT", "datasets"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--mechanism", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostic_batches", type=int, default=64)
    parser.add_argument("--diagnostic_batch_size", type=int, default=16)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source_commit", default=None)
    args = parser.parse_args(argv)

    if args.revision != MODEL_REVISION:
        parser.error(f"P5 is pinned to revision {MODEL_REVISION}")
    if args.batch_size != 64 and not args.smoke:
        parser.error("Production P5 requires nominal batch size 64")
    if args.epochs != 30 and not args.smoke and not zero_shot:
        parser.error("Production P5 requires exactly 30 epochs")
    if zero_shot:
        args.shots = 0
        args.seed = 0
        args.epochs = 0
        args.mechanism = False
    else:
        if args.shots not in SHOTS or args.seed not in SEEDS:
            parser.error("Trainable P5 conditions require shots in {1,4,16} and seed in {1,2,3}")
        if args.mechanism is None:
            args.mechanism = mechanism_cell(args.dataset, args.shots)
    if args.diagnostic_batch_size != 16 or args.diagnostic_batches != 64:
        if not args.smoke:
            parser.error("Production diagnostics require 64 fixed batches of size 16")
    if not args.offline:
        parser.error("P5 requires --offline and a verified local checkpoint path")
    if args.output_root is None:
        args.output_root = "results_zero_shot" if zero_shot else "results_with_aug"
    return args


def optimizer_config(variant: str) -> Dict[str, object]:
    if variant == "sgd":
        return {
            "class": "torch.optim.SGD",
            "lr": 1e-4,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False,
        }
    if variant in {"adam", "adam-m0"}:
        return {
            "class": "torch.optim.Adam",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "reference_sgd_lr": 1e-4 if variant == "adam-m0" else None,
        }
    return {"class": None}


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "DATASETS",
    "SHOTS",
    "SEEDS",
    "EVAL_EPOCHS",
    "DIAGNOSTIC_EPOCHS",
    "get_arguments",
    "variant_from_directory",
    "set_random_seed",
    "loader_seed",
    "diagnostic_seed",
    "seed_worker",
    "mechanism_cell",
    "optimizer_config",
]

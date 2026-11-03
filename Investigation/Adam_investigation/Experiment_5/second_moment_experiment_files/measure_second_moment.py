#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

r"""Experiment 3: few-shot vs full-data second moment in CLIP adaptation.

This script is the diagonal-preconditioner analogue of the ECCV gradient-noise
study. At selected training steps t (typically one *full-support* batch per
epoch), it freezes the current checkpoint \Theta_t and compares two estimators
computed on the same task:

  1) a *few-shot* estimator built from the K-shot support set used for training
  2) a *reference* estimator built from a larger support set from the same task

For a data source D, we define the full-support mean gradient

    m_t(D) = \nabla L_D(\Theta_t)

and a diagonal second-moment estimator v_t(D), chosen by ``--v_estimator``:

  - ``batch_mean_square``:
        v_t(D) = E_b[g_b \odot g_b]
    where g_b is the gradient of the batch-mean loss on diagnostic batch b.
    This is the most Adam-faithful option.

  - ``full_support_square``:
        v_t(D) = m_t(D) \odot m_t(D)
    This is the exact full-support analogue of the ECCV Fig. 1 gradient study.

From these quantities, the script forms three Adam-like directions:

    u_ff = m_few / sqrt(v_few + eps)
    u_fr = m_few / sqrt(v_ref + eps)
    u_rr = m_ref / sqrt(v_ref + eps)

The logged metrics answer two mechanistic questions:

  (i)  How different are the few-shot and reference second moments?
  (ii) How much does replacing only the denominator move the adaptive update
       direction, and how close does it move it toward the reference one?

Outputs
-------
The script writes a paper-ready log bundle:

  - ``<run_name>_config.json``
  - ``<run_name>_per_iteration.csv``  (encoder aggregates: vision / text)
  - ``<run_name>_per_block.csv``      (per transformer block)
  - ``<run_name>_hists_and_curves.npz``

The companion plotting script ``plot_second_moment.py`` uses the exact same
1x2, vision/text presentation style as the ECCV sign-correctness figures.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

def _import_torchvision_with_optional_nms_stub():
    """Import torchvision robustly across mixed build environments.

    Some environments ship a torchvision build where importing torchvision
    fails with ``RuntimeError: operator torchvision::nms does not exist``
    because the Python meta-registration code expects the NMS operator while
    the compiled extension does not provide it. In that specific case, we
    register a minimal placeholder schema *only after* the normal import has
    failed, then retry the import.

    This avoids the duplicate-registration crash seen on environments where
    torchvision already provides the real NMS operator.
    """
    try:
        import torchvision  # type: ignore
        import torchvision.transforms as transforms  # type: ignore
        return torchvision, transforms
    except RuntimeError as exc:
        if "operator torchvision::nms does not exist" not in str(exc):
            raise

        # Remove any partially imported torchvision modules before retrying.
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)

        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")

        import torchvision  # type: ignore
        import torchvision.transforms as transforms  # type: ignore
        return torchvision, transforms


torchvision, transforms = _import_torchvision_with_optional_nms_stub()
from torch.utils.data import DataLoader, Subset

import clip
from datasets import build_dataset
from datasets.imagenet import imagenet_classes, imagenet_templates
from datasets.utils import Datum, build_data_loader
from run_utils import set_random_seed
from train import evaluate_CLIP, mark_as_trainable
from utils import resolve_root
from second_moment_experiment_files.clip_block_utils import BlockKey, build_param_to_block_map


EPS = 1e-12


# ============================================================================
# Argument parsing
# ============================================================================

def _str2bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}



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
        "--v_estimator",
        type=str,
        default="batch_mean_square",
        choices=["batch_mean_square", "full_support_square"],
        help="How to estimate the diagonal second moment v_t(D).",
    )
    parser.add_argument(
        "--diag_every",
        type=int,
        default=1,
        help="Run the second-moment diagnostic every N optimizer steps.",
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
        help="Diagnostic mini-batch size used when --v_estimator=batch_mean_square. Set <=0 for full-support.",
    )
    parser.add_argument(
        "--diag_grad_eval_mode",
        type=_str2bool,
        default=True,
        metavar="BOOL",
        help="If True, compute diagnostics in model.eval() mode (dropout off).",
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used in Adam-like preconditioned directions.",
    )
    parser.add_argument(
        "--overlap_densities",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated densities rho for top-k overlap of v_few vs v_ref.",
    )
    parser.add_argument(
        "--logratio_min",
        type=float,
        default=-6.0,
        help="Minimum value for the histogram of log10(v_few / v_ref).",
    )
    parser.add_argument(
        "--logratio_max",
        type=float,
        default=6.0,
        help="Maximum value for the histogram of log10(v_few / v_ref).",
    )
    parser.add_argument(
        "--logratio_bins",
        type=int,
        default=120,
        help="Number of bins for the histogram of log10(v_few / v_ref).",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="second_moment_logs",
        help="Directory where the log bundle will be written.",
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
    args.overlap_densities = [float(x) for x in args.overlap_densities.split(",") if x.strip()]
    if args.ref_mode == "kshot" and args.ref_shots < args.shots:
        raise ValueError("--ref_shots must be >= --shots when --ref_mode=kshot")
    return args


# ============================================================================
# Dataset helpers
# ============================================================================


def load_clip_model(backbone: str, device: str):
    cache = os.getenv("CLIP_DOWNLOAD_ROOT")
    model, preprocess = clip.load(backbone, device=device, download_root=cache)
    return model, preprocess



def make_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )



def get_labels_from_data_source(data_source: Sequence[Datum]) -> List[int]:
    return [item.label for item in data_source]



def group_indices_by_label(labels: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(int(idx))
    return groups



def sample_nested_indices(
    labels: Sequence[int],
    *,
    few_shots: int,
    ref_mode: str,
    ref_shots: int,
    max_ref_per_class: int,
    seed: int,
    repeat_if_needed: bool,
) -> Tuple[List[int], List[int]]:
    """Sample nested few-shot and reference index lists from a full training split."""
    rng = random.Random(seed)
    by_label = group_indices_by_label(labels)

    few_indices: List[int] = []
    ref_indices: List[int] = []

    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)

        if ref_mode == "full_train":
            ref_count = len(idxs)
            if max_ref_per_class > 0:
                ref_count = min(ref_count, max_ref_per_class)
            ref_this = idxs[:ref_count]
        else:
            if len(idxs) >= ref_shots:
                ref_this = idxs[:ref_shots]
            elif repeat_if_needed:
                ref_this = idxs + rng.choices(idxs, k=ref_shots - len(idxs))
            else:
                ref_this = idxs[:]

        if len(ref_this) >= few_shots:
            few_this = ref_this[:few_shots]
        elif repeat_if_needed:
            base = ref_this if len(ref_this) > 0 else idxs
            if len(base) == 0:
                raise RuntimeError(f"No training samples found for label={label}")
            few_this = ref_this + rng.choices(base, k=few_shots - len(ref_this))
        else:
            few_this = ref_this[:]

        few_indices.extend(few_this)
        ref_indices.extend(ref_this)

    return few_indices, ref_indices


class NonImagenetBundle(SimpleNamespace):
    train_full: List[Datum]
    few_train: List[Datum]
    ref_train: List[Datum]
    test: Sequence
    val: Sequence
    classnames: List[str]
    template: List[str]


class ImagenetBundle(SimpleNamespace):
    train_aug: torchvision.datasets.ImageFolder
    train_eval: torchvision.datasets.ImageFolder
    few_indices: List[int]
    ref_indices: List[int]
    test: torchvision.datasets.ImageFolder
    classnames: List[str]
    template: List[str]



def build_non_imagenet_bundle(args: argparse.Namespace, preprocess) -> NonImagenetBundle:
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)
    full_train = list(full_dataset.train_x)
    labels = get_labels_from_data_source(full_train)
    few_idx, ref_idx = sample_nested_indices(
        labels,
        few_shots=args.shots,
        ref_mode=args.ref_mode,
        ref_shots=args.ref_shots,
        max_ref_per_class=args.max_ref_per_class,
        seed=args.seed,
        repeat_if_needed=args.repeat_if_needed,
    )
    few_train = [full_train[i] for i in few_idx]
    ref_train = [full_train[i] for i in ref_idx]
    return NonImagenetBundle(
        train_full=full_train,
        few_train=few_train,
        ref_train=ref_train,
        test=full_dataset.test,
        val=full_dataset.val,
        classnames=list(full_dataset.classnames),
        template=list(full_dataset.template),
        dataset_obj=full_dataset,
    )



def build_imagenet_bundle(args: argparse.Namespace, preprocess) -> ImagenetBundle:
    root = Path(args.root_path).expanduser().resolve() / "imagenet"
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageNet train folder not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val folder not found: {val_dir}")

    train_aug_tfm = make_train_transform() if args.use_aug else preprocess
    train_aug = torchvision.datasets.ImageFolder(str(train_dir), transform=train_aug_tfm)
    train_eval = torchvision.datasets.ImageFolder(str(train_dir), transform=preprocess)
    test = torchvision.datasets.ImageFolder(str(val_dir), transform=preprocess)

    labels = list(train_eval.targets)
    few_idx, ref_idx = sample_nested_indices(
        labels,
        few_shots=args.shots,
        ref_mode=args.ref_mode,
        ref_shots=args.ref_shots,
        max_ref_per_class=args.max_ref_per_class,
        seed=args.seed,
        repeat_if_needed=args.repeat_if_needed,
    )

    return ImagenetBundle(
        train_aug=train_aug,
        train_eval=train_eval,
        few_indices=few_idx,
        ref_indices=ref_idx,
        test=test,
        classnames=list(imagenet_classes),
        template=list(imagenet_templates),
        dataset_obj=SimpleNamespace(classnames=list(imagenet_classes), template=list(imagenet_templates)),
    )



def build_dataset_bundle(args: argparse.Namespace, preprocess):
    if args.dataset == "imagenet":
        return build_imagenet_bundle(args, preprocess)
    return build_non_imagenet_bundle(args, preprocess)



def _train_batch_size(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return int(args.batch_size)



def _diag_batch_size(num_items: int, args: argparse.Namespace) -> int:
    if args.diag_batch_size <= 0:
        return max(1, num_items)
    return int(args.diag_batch_size)



def build_non_imagenet_loaders(bundle: NonImagenetBundle, args: argparse.Namespace, preprocess):
    train_tfm = make_train_transform() if args.use_aug else preprocess
    train_bs = _train_batch_size(len(bundle.few_train), args)
    diag_bs_few = _diag_batch_size(len(bundle.few_train), args)
    diag_bs_ref = _diag_batch_size(len(bundle.ref_train), args)

    train_loader = build_data_loader(
        data_source=bundle.few_train,
        batch_size=train_bs,
        tfm=train_tfm,
        is_train=True,
        shuffle=not (args.full_support_train or args.batch_size <= 0),
        num_workers=args.num_workers,
    )
    few_diag_loader = build_data_loader(
        data_source=bundle.few_train,
        batch_size=diag_bs_few,
        tfm=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    ref_diag_loader = build_data_loader(
        data_source=bundle.ref_train,
        batch_size=diag_bs_ref,
        tfm=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_data_loader(
        data_source=bundle.test,
        batch_size=256,
        tfm=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, few_diag_loader, ref_diag_loader, test_loader



def build_imagenet_loaders(bundle: ImagenetBundle, args: argparse.Namespace):
    train_bs = _train_batch_size(len(bundle.few_indices), args)
    diag_bs_few = _diag_batch_size(len(bundle.few_indices), args)
    diag_bs_ref = _diag_batch_size(len(bundle.ref_indices), args)

    train_loader = DataLoader(
        Subset(bundle.train_aug, bundle.few_indices),
        batch_size=train_bs,
        shuffle=not (args.full_support_train or args.batch_size <= 0),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    few_diag_loader = DataLoader(
        Subset(bundle.train_eval, bundle.few_indices),
        batch_size=diag_bs_few,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    ref_diag_loader = DataLoader(
        Subset(bundle.train_eval, bundle.ref_indices),
        batch_size=diag_bs_ref,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        bundle.test,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, few_diag_loader, ref_diag_loader, test_loader



def build_loaders(bundle, args: argparse.Namespace, preprocess):
    if args.dataset == "imagenet":
        return build_imagenet_loaders(bundle, args)
    return build_non_imagenet_loaders(bundle, args, preprocess)


# ============================================================================
# Diagnostics / metrics
# ============================================================================


def tokenize_class_texts(classnames: Sequence[str], template: Sequence[str], device: str) -> torch.Tensor:
    prompt_template = template[0]
    texts = [prompt_template.format(name.replace("_", " ")) for name in classnames]
    return clip.tokenize(texts).to(device)



def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> float:
    x_norm = torch.norm(x, p=2)
    y_norm = torch.norm(y, p=2)
    denom = float(x_norm.item() * y_norm.item())
    if denom <= eps:
        return float("nan")
    return float(torch.dot(x, y).item() / (denom + eps))



def sign_agreement(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = (x != 0) | (y != 0)
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    same = (torch.sign(x[mask]) == torch.sign(y[mask])).sum().item()
    return float(same) / float(denom)


def safe_percentile_cpu(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    arr = x.detach().contiguous().numpy()
    return float(np.percentile(arr, q * 100.0))



def safe_log10_ratio(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log10((a + eps) / (b + eps))



def density_to_col(density: float) -> str:
    s = f"{density:g}".replace(".", "p").replace("-", "m")
    return f"v_topk_overlap_rho_{s}"



def topk_overlap(x: torch.Tensor, y: torch.Tensor, density: float) -> float:
    n = int(x.numel())
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * density)))
    if k >= n:
        return 1.0
    idx_x = torch.topk(x, k=k, largest=True, sorted=False).indices.cpu().numpy()
    idx_y = torch.topk(y, k=k, largest=True, sorted=False).indices.cpu().numpy()
    inter = np.intersect1d(idx_x, idx_y, assume_unique=False)
    return float(inter.size) / float(k)



def build_scope_tensors(
    *,
    m_few: Dict[torch.nn.Parameter, torch.Tensor],
    v_few: Dict[torch.nn.Parameter, torch.Tensor],
    m_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_ref: Dict[torch.nn.Parameter, torch.Tensor],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
) -> Tuple[Dict[BlockKey, Dict[str, List[torch.Tensor]]], Dict[str, Dict[str, List[torch.Tensor]]]]:
    by_block: Dict[BlockKey, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    by_encoder: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    for p, gf in m_few.items():
        bk = param_to_block.get(p)
        if bk is None:
            continue
        enc, _ = bk
        tensors = {
            "m_few": gf.reshape(-1),
            "v_few": v_few[p].reshape(-1),
            "m_ref": m_ref[p].reshape(-1),
            "v_ref": v_ref[p].reshape(-1),
        }
        for key, value in tensors.items():
            by_block[bk][key].append(value)
            by_encoder[enc][key].append(value)

    return by_block, by_encoder



def compute_metric_row(
    *,
    m_few_vec: torch.Tensor,
    v_few_vec: torch.Tensor,
    m_ref_vec: torch.Tensor,
    v_ref_vec: torch.Tensor,
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

    numel = int(m_few_vec.numel())
    log_ratio = safe_log10_ratio(v_few_vec, v_ref_vec, args.adam_eps).abs()

    u_ff = m_few_vec / torch.sqrt(v_few_vec + args.adam_eps)
    u_fr = m_few_vec / torch.sqrt(v_ref_vec + args.adam_eps)
    u_rr = m_ref_vec / torch.sqrt(v_ref_vec + args.adam_eps)

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
        "grad_cosine": cosine_similarity(m_few_vec, m_ref_vec),
        "grad_sign_agreement": sign_agreement(m_few_vec, m_ref_vec),
        "few_grad_l2": float(torch.norm(m_few_vec, p=2).item()),
        "ref_grad_l2": float(torch.norm(m_ref_vec, p=2).item()),
        "few_v_l2": float(torch.norm(v_few_vec, p=2).item()),
        "ref_v_l2": float(torch.norm(v_ref_vec, p=2).item()),
        "v_mean_abs_log10_ratio": float(torch.mean(log_ratio).item()),
        "v_median_abs_log10_ratio": float(torch.median(log_ratio).item()),
        "v_p90_abs_log10_ratio": safe_percentile_cpu(log_ratio, 0.9),
        "adam_cos_fewfew_vs_fewref": cosine_similarity(u_ff, u_fr),
        "adam_cos_fewref_vs_refref": cosine_similarity(u_fr, u_rr),
        "adam_l2_fewfew": float(torch.norm(u_ff, p=2).item()),
        "adam_l2_fewref": float(torch.norm(u_fr, p=2).item()),
        "adam_l2_refref": float(torch.norm(u_rr, p=2).item()),
    }
    for density in args.overlap_densities:
        row[density_to_col(density)] = topk_overlap(v_few_vec, v_ref_vec, density)
    return row


class SecondMomentLogger:
    """Write per-step CSVs and an NPZ bundle for paper-style plots."""

    def __init__(
        self,
        out_dir: Path,
        run_name: str,
        *,
        topk_fracs: Sequence[float],
        logratio_min: float,
        logratio_max: float,
        logratio_bins: int,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.topk_fracs = np.array(list(topk_fracs), dtype=np.float64)
        self.logratio_min = float(logratio_min)
        self.logratio_max = float(logratio_max)
        self.logratio_bins = int(logratio_bins)
        self.logratio_edges = np.linspace(self.logratio_min, self.logratio_max, self.logratio_bins + 1, dtype=np.float64)

        self._encoders = ["vision", "text"]
        self.logratio_hist: Dict[str, np.ndarray] = {e: np.zeros(self.logratio_bins, dtype=np.int64) for e in self._encoders}
        self.topk_overlap_curve: Dict[str, List[np.ndarray]] = {e: [] for e in self._encoders}

        self.iter_csv_path = self.out_dir / f"{self.run_name}_per_iteration.csv"
        self.block_csv_path = self.out_dir / f"{self.run_name}_per_block.csv"
        self.npz_path = self.out_dir / f"{self.run_name}_hists_and_curves.npz"

        fieldnames = [
            "global_step",
            "epoch",
            "step_in_epoch",
            "scope",
            "encoder",
            "block",
            "numel",
            "few_size",
            "ref_size",
            "few_loss",
            "ref_loss",
            "traj_optimizer",
            "dataset",
            "shots",
            "ref_mode",
            "ref_shots",
            "v_estimator",
            "grad_cosine",
            "grad_sign_agreement",
            "few_grad_l2",
            "ref_grad_l2",
            "few_v_l2",
            "ref_v_l2",
            "v_mean_abs_log10_ratio",
            "v_median_abs_log10_ratio",
            "v_p90_abs_log10_ratio",
            "adam_cos_fewfew_vs_fewref",
            "adam_cos_fewref_vs_refref",
            "adam_l2_fewfew",
            "adam_l2_fewref",
            "adam_l2_refref",
        ] + [density_to_col(d) for d in self.topk_fracs]

        self._iter_f = open(self.iter_csv_path, "w", newline="")
        self._block_f = open(self.block_csv_path, "w", newline="")
        self._iter_w = csv.DictWriter(self._iter_f, fieldnames=fieldnames)
        self._block_w = csv.DictWriter(self._block_f, fieldnames=fieldnames)
        self._iter_w.writeheader()
        self._block_w.writeheader()

    def _accum_hist_(self, out_counts: np.ndarray, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        h = torch.histc(
            values.to(dtype=torch.float32),
            bins=self.logratio_bins,
            min=float(self.logratio_min),
            max=float(self.logratio_max),
        )
        out_counts += h.detach().cpu().to(dtype=torch.int64).numpy()

    def log_step(
        self,
        *,
        encoder_rows: Dict[str, Dict[str, object]],
        block_rows: Sequence[Dict[str, object]],
        encoder_tensors: Dict[str, Dict[str, torch.Tensor]],
        adam_eps: float,
    ) -> None:
        for enc in self._encoders:
            row = encoder_rows.get(enc)
            tensors = encoder_tensors.get(enc)
            if row is None or tensors is None:
                continue

            self._iter_w.writerow(row)

            v_few_vec = tensors["v_few"].float().cpu()
            v_ref_vec = tensors["v_ref"].float().cpu()
            valid = (v_few_vec.abs() > adam_eps) | (v_ref_vec.abs() > adam_eps)
            if valid.any():
                logratio = safe_log10_ratio(v_few_vec[valid], v_ref_vec[valid], adam_eps)
                self._accum_hist_(self.logratio_hist[enc], logratio)

            overlaps = np.array(
                [topk_overlap(v_few_vec, v_ref_vec, d) for d in self.topk_fracs],
                dtype=np.float64,
            )
            self.topk_overlap_curve[enc].append(overlaps)

        for row in block_rows:
            self._block_w.writerow(row)

        self._iter_f.flush()
        self._block_f.flush()

    def close(self) -> None:
        self._iter_f.close()
        self._block_f.close()
        np.savez_compressed(
            self.npz_path,
            logratio_edges=self.logratio_edges,
            topk_fracs=self.topk_fracs,
            logratio_hist_vision=self.logratio_hist["vision"],
            logratio_hist_text=self.logratio_hist["text"],
            topk_overlap_curve_vision=(
                np.stack(self.topk_overlap_curve["vision"], axis=0)
                if self.topk_overlap_curve["vision"]
                else np.zeros((0, len(self.topk_fracs)), dtype=np.float64)
            ),
            topk_overlap_curve_text=(
                np.stack(self.topk_overlap_curve["text"], axis=0)
                if self.topk_overlap_curve["text"]
                else np.zeros((0, len(self.topk_fracs)), dtype=np.float64)
            ),
        )


# ============================================================================
# Core gradient / second-moment computations
# ============================================================================


def compute_dataset_gradient_and_second_moment(
    *,
    model,
    loader: DataLoader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    params: Sequence[torch.nn.Parameter],
    device: str,
    grad_eval_mode: bool,
    v_estimator: str,
    adam_eps: float,
) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], Dict[torch.nn.Parameter, torch.Tensor], float]:
    """Estimate m_t(D) and v_t(D) on a data source D.

    ``m_t(D)`` is always the dataset-level mean gradient. ``v_t(D)`` follows
    ``--v_estimator``.
    """
    was_training = model.training
    if grad_eval_mode:
        model.eval()
    else:
        model.train()

    total_size = len(loader.dataset)
    if total_size <= 0:
        raise RuntimeError("Diagnostic loader is empty; cannot estimate second moments.")

    m_est = {p: torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params}
    v_est = {p: torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params}
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
        loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()

        total_loss += float(loss.item()) * batch_size

        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            m_est[p].add_(g, alpha=batch_weight)
            if v_estimator == "batch_mean_square":
                v_est[p].add_(g * g, alpha=batch_weight)

    if v_estimator == "full_support_square":
        for p in params:
            v_est[p] = m_est[p] * m_est[p]

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return m_est, v_est, total_loss / float(total_size)



def compute_rows_and_encoder_tensors(
    *,
    m_few: Dict[torch.nn.Parameter, torch.Tensor],
    v_few: Dict[torch.nn.Parameter, torch.Tensor],
    m_ref: Dict[torch.nn.Parameter, torch.Tensor],
    v_ref: Dict[torch.nn.Parameter, torch.Tensor],
    param_to_block: Dict[torch.nn.Parameter, BlockKey],
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    few_size: int,
    ref_size: int,
    few_loss: float,
    ref_loss: float,
) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]], Dict[str, Dict[str, torch.Tensor]]]:
    by_block, by_encoder = build_scope_tensors(
        m_few=m_few,
        v_few=v_few,
        m_ref=m_ref,
        v_ref=v_ref,
        param_to_block=param_to_block,
    )

    encoder_rows: Dict[str, Dict[str, object]] = {}
    encoder_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    block_rows: List[Dict[str, object]] = []

    for enc, tensors in sorted(by_encoder.items()):
        m_few_vec = torch.cat(tensors["m_few"], dim=0)
        v_few_vec = torch.cat(tensors["v_few"], dim=0)
        m_ref_vec = torch.cat(tensors["m_ref"], dim=0)
        v_ref_vec = torch.cat(tensors["v_ref"], dim=0)
        encoder_tensors[enc] = {
            "m_few": m_few_vec,
            "v_few": v_few_vec,
            "m_ref": m_ref_vec,
            "v_ref": v_ref_vec,
        }
        encoder_rows[enc] = compute_metric_row(
            m_few_vec=m_few_vec,
            v_few_vec=v_few_vec,
            m_ref_vec=m_ref_vec,
            v_ref_vec=v_ref_vec,
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

    for (enc, blk), tensors in sorted(by_block.items(), key=lambda x: (x[0][0], x[0][1])):
        block_rows.append(
            compute_metric_row(
                m_few_vec=torch.cat(tensors["m_few"], dim=0),
                v_few_vec=torch.cat(tensors["v_few"], dim=0),
                m_ref_vec=torch.cat(tensors["m_ref"], dim=0),
                v_ref_vec=torch.cat(tensors["v_ref"], dim=0),
                args=args,
                scope="block",
                encoder=enc,
                block=blk,
                global_step=global_step,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                few_size=few_size,
                ref_size=ref_size,
                few_loss=few_loss,
                ref_loss=ref_loss,
            )
        )

    return encoder_rows, block_rows, encoder_tensors



def make_optimizer(name: str, params: Sequence[torch.nn.Parameter], lr: float, eps: float):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=eps)
    raise ValueError(f"Unsupported trajectory optimizer: {name}")


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
    vest = args.v_estimator.replace("_", "-")
    return (
        f"second_moment_dataset_{args.dataset}_noisyshots_{args.shots}_ref_{ref_tag}"
        f"_backbone_{bb}_trajopt_{args.traj_optimizer.lower()}_seed_{args.seed}"
        f"_pos_{args.position}_enc_{args.encoder}_vest_{vest}"
    )



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
    config_path = out_dir / f"{run_name}_config.json"

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger = SecondMomentLogger(
        out_dir=out_dir,
        run_name=run_name,
        topk_fracs=args.overlap_densities,
        logratio_min=args.logratio_min,
        logratio_max=args.logratio_max,
        logratio_bins=args.logratio_bins,
    )

    print(f"[SecondMoment] Logging bundle to: {out_dir}")
    print(f"[SecondMoment] Run name: {run_name}")
    print(
        f"[SecondMoment] trajectory={args.traj_optimizer} | full_support_train={args.full_support_train or args.batch_size <= 0} "
        f"| v_estimator={args.v_estimator} | ref_mode={args.ref_mode}"
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
                    f"[SecondMoment] step={global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"estimating few-shot/reference moments ..."
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
                encoder_rows, block_rows, encoder_tensors = compute_rows_and_encoder_tensors(
                    m_few=m_few,
                    v_few=v_few,
                    m_ref=m_ref,
                    v_ref=v_ref,
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
                logger.log_step(
                    encoder_rows=encoder_rows,
                    block_rows=block_rows,
                    encoder_tensors=encoder_tensors,
                    adam_eps=args.adam_eps,
                )
                diag_count += 1

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
        eval_path = out_dir / f"{run_name}_final_eval.json"
        with open(eval_path, "w") as f:
            json.dump({"test_acc": acc}, f, indent=2)



def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Experiment 3: EMA staleness audit for Adam in few-shot CLIP adaptation.

This experiment answers the question:

    Is the problem specifically Adam's *running* second moment, or would even
    the instantaneous few-shot second moment already be bad?

At each logged checkpoint \Theta_t along an **Adam trajectory**, the script
computes three few-shot denominator candidates and compares all of them to the
same full-data reference second moment:

  1) instant:
       v_inst = m_few \odot m_few
     This is the full-support / instantaneous squared gradient on the few-shot
     support at the current checkpoint.

  2) batch:
       v_batch = E_b[g_b \odot g_b]
     This is the batch-based few-shot second moment, estimated over diagnostic
     mini-batches of the few-shot support. To make it meaningfully different
     from ``v_inst`` for one-shot problems, this script lets the few-shot and
     reference diagnostic batch sizes differ.

  3) ema:
       \hat v_ema
     This is the *actual* bias-corrected Adam running second moment extracted
     from the optimizer state at the current checkpoint.

All three are compared against the same reference moment

       v_ref = E_b[g_b \odot g_b]

computed on the larger reference dataset (typically the full training split of
that target task).

For each source ``s in {instant, batch, ema}``, we form

    u_s   = m_few / sqrt(v_s   + eps)
    u_or  = m_few / sqrt(v_ref + eps)
    u_rr  = m_ref / sqrt(v_ref + eps)

and log:

  - how different v_s is from v_ref,
  - how different u_s is from the oracle-denominator direction u_or,
  - how close u_s is to the reference adaptive direction u_rr.

This implements the exact core protocol recommended for the EMA-staleness
experiment. The beta2 sweep is handled by running this script multiple times
with different values of ``--adam_beta2``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

# Import the Experiment 2 measurement module first so its robust torchvision
# import path is executed before importing train.py / clip.py.
from second_moment_experiment_files.measure_second_moment import (
    _str2bool,
    build_dataset_bundle,
    compute_dataset_gradient_and_second_moment,
    build_scope_tensors,
    cosine_similarity,
    density_to_col,
    load_clip_model,
    make_train_transform,
    safe_log10_ratio,
    safe_percentile_cpu,
    sign_agreement,
    topk_overlap,
    tokenize_class_texts,
)
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from second_moment_experiment_files.clip_block_utils import BlockKey, build_param_to_block_map
from train import mark_as_trainable

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

EPS = 1e-12


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

    # Experiment 3 is an Adam-only audit because it explicitly inspects the
    # running Adam second moment stored in the optimizer state.
    parser.add_argument(
        "--traj_optimizer",
        type=str,
        default="Adam",
        choices=["Adam"],
        help="Experiment 3 inspects Adam's running second moment, so the trajectory is Adam-only.",
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
        help="Run the EMA-staleness diagnostic every N optimizer steps.",
    )
    parser.add_argument(
        "--max_diag_steps",
        type=int,
        default=-1,
        help="Stop logging after this many checkpoints (-1 = log whenever scheduled).",
    )
    parser.add_argument(
        "--few_diag_batch_size",
        type=int,
        default=0,
        help=(
            "Diagnostic batch size used to estimate the *few-shot batch* second moment. "
            "Use a value smaller than the support size to make the batch estimator distinct from the instantaneous one. "
            "Set <=0 to use the full few-shot support in one batch."
        ),
    )
    parser.add_argument(
        "--ref_diag_batch_size",
        type=int,
        default=256,
        help="Diagnostic batch size used to estimate the full-data reference second moment.",
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 for the training trajectory.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 for the training trajectory and the extracted EMA state.")

    parser.add_argument(
        "--overlap_densities",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated densities rho for top-k overlap of v_source vs v_ref.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results_without_aug/ema_staleness_logs",
        help="Directory where the log bundle will be written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, the script creates one automatically.",
    )

    args = parser.parse_args()
    args.root_path = str(Path(args.root_path).expanduser())
    args.overlap_densities = [float(x) for x in args.overlap_densities.split(",") if x.strip()]
    return args


# ============================================================================
# Loader helpers with separate few/ref diagnostic batch sizes
# ============================================================================

def _train_batch_size(num_items: int, args: argparse.Namespace) -> int:
    if args.full_support_train or args.batch_size <= 0:
        return max(1, num_items)
    return min(int(args.batch_size), max(1, num_items))


def _few_diag_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.few_diag_batch_size <= 0:
        return max(1, num_items)
    return min(int(args.few_diag_batch_size), max(1, num_items))


def _ref_diag_bs(num_items: int, args: argparse.Namespace) -> int:
    if args.ref_diag_batch_size <= 0:
        return max(1, num_items)
    return min(int(args.ref_diag_batch_size), max(1, num_items))


def build_loaders_separate_diag(bundle, args: argparse.Namespace, preprocess):
    train_tfm = make_train_transform() if args.use_aug else preprocess

    if args.dataset == "imagenet":
        train_loader = DataLoader(
            Subset(bundle.train_aug, bundle.few_indices),
            batch_size=_train_batch_size(len(bundle.few_indices), args),
            shuffle=not (args.full_support_train or args.batch_size <= 0),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        few_diag_loader = DataLoader(
            Subset(bundle.train_eval, bundle.few_indices),
            batch_size=_few_diag_bs(len(bundle.few_indices), args),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        ref_diag_loader = DataLoader(
            Subset(bundle.train_eval, bundle.ref_indices),
            batch_size=_ref_diag_bs(len(bundle.ref_indices), args),
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
    else:
        train_loader = build_data_loader(
            data_source=bundle.few_train,
            batch_size=_train_batch_size(len(bundle.few_train), args),
            tfm=train_tfm,
            is_train=True,
            shuffle=not (args.full_support_train or args.batch_size <= 0),
            num_workers=args.num_workers,
        )
        few_diag_loader = build_data_loader(
            data_source=bundle.few_train,
            batch_size=_few_diag_bs(len(bundle.few_train), args),
            tfm=preprocess,
            is_train=False,
            shuffle=False,
            num_workers=args.num_workers,
        )
        ref_diag_loader = build_data_loader(
            data_source=bundle.ref_train,
            batch_size=_ref_diag_bs(len(bundle.ref_train), args),
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


# ============================================================================
# Adam-state extraction
# ============================================================================

def _state_step_to_int(step_obj) -> int:
    if step_obj is None:
        return 0
    if torch.is_tensor(step_obj):
        return int(step_obj.item())
    return int(step_obj)



def extract_bias_corrected_exp_avg_sq(
    optimizer: torch.optim.Optimizer,
    params: Sequence[torch.nn.Parameter],
    beta2: float,
) -> Dict[torch.nn.Parameter, torch.Tensor]:
    out: Dict[torch.nn.Parameter, torch.Tensor] = {}
    for p in params:
        st = optimizer.state.get(p, None)
        if st is None or "exp_avg_sq" not in st:
            out[p] = torch.zeros_like(p, dtype=torch.float32, device="cpu")
            continue
        step = _state_step_to_int(st.get("step", 0))
        exp_avg_sq = st["exp_avg_sq"].detach().float().cpu()
        if step <= 0:
            out[p] = torch.zeros_like(exp_avg_sq, dtype=torch.float32)
            continue
        bias_correction2 = 1.0 - (beta2 ** step)
        if bias_correction2 <= EPS:
            out[p] = exp_avg_sq.clone()
        else:
            out[p] = exp_avg_sq / bias_correction2
    return out


# ============================================================================
# Row computation
# ============================================================================

def compute_ema_audit_row(
    *,
    m_few_vec: torch.Tensor,
    v_source_vec: torch.Tensor,
    m_ref_vec: torch.Tensor,
    v_ref_vec: torch.Tensor,
    args: argparse.Namespace,
    scope: str,
    moment_source: str,
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
    v_source_vec = v_source_vec.float().cpu()
    m_ref_vec = m_ref_vec.float().cpu()
    v_ref_vec = v_ref_vec.float().cpu()

    numel = int(m_few_vec.numel())
    log_ratio = safe_log10_ratio(v_source_vec, v_ref_vec, args.adam_eps).abs()

    u_src = m_few_vec / torch.sqrt(v_source_vec + args.adam_eps)
    u_or = m_few_vec / torch.sqrt(v_ref_vec + args.adam_eps)
    u_rr = m_ref_vec / torch.sqrt(v_ref_vec + args.adam_eps)

    src_l2 = float(torch.norm(u_src, p=2).item())
    or_l2 = float(torch.norm(u_or, p=2).item())
    rr_l2 = float(torch.norm(u_rr, p=2).item())

    row: Dict[str, object] = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "scope": scope,
        "moment_source": moment_source,
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
        "few_diag_batch_size": int(_few_diag_bs(few_size, args)),
        "ref_diag_batch_size": int(_ref_diag_bs(ref_size, args)),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "grad_cosine": cosine_similarity(m_few_vec, m_ref_vec),
        "grad_sign_agreement": sign_agreement(m_few_vec, m_ref_vec),
        "few_grad_l2": float(torch.norm(m_few_vec, p=2).item()),
        "ref_grad_l2": float(torch.norm(m_ref_vec, p=2).item()),
        "source_v_l2": float(torch.norm(v_source_vec, p=2).item()),
        "ref_v_l2": float(torch.norm(v_ref_vec, p=2).item()),
        "v_mean_abs_log10_ratio": float(torch.mean(log_ratio).item()),
        "v_median_abs_log10_ratio": float(torch.median(log_ratio).item()),
        "v_p90_abs_log10_ratio": safe_percentile_cpu(log_ratio, 0.9),
        "source_cos_vs_oracle": cosine_similarity(u_src, u_or),
        "source_cos_vs_ref": cosine_similarity(u_src, u_rr),
        "oracle_cos_vs_ref": cosine_similarity(u_or, u_rr),
        "source_l2": src_l2,
        "oracle_l2": or_l2,
        "ref_l2": rr_l2,
        "source_over_ref_l2": float(src_l2 / max(rr_l2, EPS)),
        "oracle_over_ref_l2": float(or_l2 / max(rr_l2, EPS)),
    }
    for density in args.overlap_densities:
        row[density_to_col(density)] = topk_overlap(v_source_vec, v_ref_vec, density)
    return row



def compute_rows_for_source(
    *,
    moment_source: str,
    m_few: Dict[torch.nn.Parameter, torch.Tensor],
    v_source: Dict[torch.nn.Parameter, torch.Tensor],
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
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_block, by_encoder = build_scope_tensors(
        m_few=m_few,
        v_few=v_source,
        m_ref=m_ref,
        v_ref=v_ref,
        param_to_block=param_to_block,
    )

    iter_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []

    for enc, tensors in sorted(by_encoder.items()):
        iter_rows.append(
            compute_ema_audit_row(
                m_few_vec=torch.cat(tensors["m_few"], dim=0),
                v_source_vec=torch.cat(tensors["v_few"], dim=0),
                m_ref_vec=torch.cat(tensors["m_ref"], dim=0),
                v_ref_vec=torch.cat(tensors["v_ref"], dim=0),
                args=args,
                scope="encoder",
                moment_source=moment_source,
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
        )

    for (enc, blk), tensors in sorted(by_block.items(), key=lambda x: (x[0][0], x[0][1])):
        block_rows.append(
            compute_ema_audit_row(
                m_few_vec=torch.cat(tensors["m_few"], dim=0),
                v_source_vec=torch.cat(tensors["v_few"], dim=0),
                m_ref_vec=torch.cat(tensors["m_ref"], dim=0),
                v_ref_vec=torch.cat(tensors["v_ref"], dim=0),
                args=args,
                scope="block",
                moment_source=moment_source,
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

    return iter_rows, block_rows


# ============================================================================
# Logger
# ============================================================================

class EMAStalenessLogger:
    def __init__(self, out_dir: Path, run_name: str, overlap_densities: Sequence[float]):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.iter_csv_path = self.out_dir / f"{self.run_name}_per_iteration.csv"
        self.block_csv_path = self.out_dir / f"{self.run_name}_per_block.csv"

        fieldnames = [
            "global_step",
            "epoch",
            "step_in_epoch",
            "scope",
            "moment_source",
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
            "few_diag_batch_size",
            "ref_diag_batch_size",
            "adam_beta1",
            "adam_beta2",
            "grad_cosine",
            "grad_sign_agreement",
            "few_grad_l2",
            "ref_grad_l2",
            "source_v_l2",
            "ref_v_l2",
            "v_mean_abs_log10_ratio",
            "v_median_abs_log10_ratio",
            "v_p90_abs_log10_ratio",
            "source_cos_vs_oracle",
            "source_cos_vs_ref",
            "oracle_cos_vs_ref",
            "source_l2",
            "oracle_l2",
            "ref_l2",
            "source_over_ref_l2",
            "oracle_over_ref_l2",
        ] + [density_to_col(d) for d in overlap_densities]

        self._iter_f = open(self.iter_csv_path, "w", newline="")
        self._block_f = open(self.block_csv_path, "w", newline="")
        self._iter_w = csv.DictWriter(self._iter_f, fieldnames=fieldnames)
        self._block_w = csv.DictWriter(self._block_f, fieldnames=fieldnames)
        self._iter_w.writeheader()
        self._block_w.writeheader()

    def log_rows(self, iter_rows: List[Dict[str, object]], block_rows: List[Dict[str, object]]) -> None:
        for row in iter_rows:
            self._iter_w.writerow(row)
        for row in block_rows:
            self._block_w.writerow(row)
        self._iter_f.flush()
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
    b2 = f"{args.adam_beta2:g}".replace(".", "p")
    return (
        f"ema_staleness_dataset_{args.dataset}_noisyshots_{args.shots}_ref_{ref_tag}"
        f"_backbone_{bb}_trajopt_adam_b2_{b2}_seed_{args.seed}"
        f"_pos_{args.position}_enc_{args.encoder}"
    )



def run_experiment(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0
    bundle = build_dataset_bundle(args, preprocess)
    train_loader, few_diag_loader, ref_diag_loader, _ = build_loaders_separate_diag(bundle, args, preprocess)

    mark_as_trainable(model, args)
    model = model.to(args.device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected. Check --encoder and --position.")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    param_to_block = build_param_to_block_map(model, only_trainable=True)
    class_tokens = tokenize_class_texts(bundle.classnames, bundle.template, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)

    cfg = vars(args).copy()
    cfg["experiment"] = "ema_staleness_audit"
    with open(out_dir / f"{run_name}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger = EMAStalenessLogger(out_dir=out_dir, run_name=run_name, overlap_densities=args.overlap_densities)

    print(f"[EMAStaleness] Logging bundle to: {out_dir}")
    print(f"[EMAStaleness] Run name: {run_name}")
    print(
        f"[EMAStaleness] trajectory=Adam | beta2={args.adam_beta2} | "
        f"few_diag_batch_size={_few_diag_bs(len(few_diag_loader.dataset), args)} | "
        f"ref_diag_batch_size={_ref_diag_bs(len(ref_diag_loader.dataset), args)} | ref_mode={args.ref_mode}"
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
                    f"[EMAStaleness] step={global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"estimating instant / batch / ema denominators ..."
                )

                m_few, v_batch, few_loss = compute_dataset_gradient_and_second_moment(
                    model=model,
                    loader=few_diag_loader,
                    class_tokens=class_tokens,
                    logit_scale=logit_scale,
                    params=trainable_params,
                    device=args.device,
                    grad_eval_mode=args.diag_grad_eval_mode,
                    v_estimator="batch_mean_square",
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
                    v_estimator="batch_mean_square",
                    adam_eps=args.adam_eps,
                )

                v_inst = {p: (m_few[p] * m_few[p]).clone() for p in trainable_params}
                v_ema = extract_bias_corrected_exp_avg_sq(optimizer, trainable_params, args.adam_beta2)

                for moment_source, v_source in [("instant", v_inst), ("batch", v_batch), ("ema", v_ema)]:
                    iter_rows, block_rows = compute_rows_for_source(
                        moment_source=moment_source,
                        m_few=m_few,
                        v_source=v_source,
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
                    logger.log_rows(iter_rows, block_rows)

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



def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

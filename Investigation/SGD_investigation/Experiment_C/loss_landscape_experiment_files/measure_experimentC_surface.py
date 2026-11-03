#!/usr/bin/env python3
"""Experiment C — 2D local surfaces around SGD checkpoints.

For a saved SGD checkpoint Theta_t, evaluate the local 2D loss surface

    L(alpha, beta) = L(Theta_t + scale * (alpha d_t + beta e_t))

where
- d_t is the actual SGD descent direction at Theta_t (for plain SGD: -few-shot gradient),
- e_t is an orthogonal probe direction.

Supported probes
----------------
- hessian : top few-shot Hessian eigenvector, orthogonalized to d_t
- random  : random direction, orthogonalized to d_t
- fullgrad: full-train gradient direction, orthogonalized to d_t
- adam    : virtual Adam-like direction m_few / sqrt(v_few + eps), orthogonalized to d_t

The surface is measured for three losses:
- support / few-shot loss
- full-train loss
- test loss

Outputs
-------
<out_dir>/<surface_run_name>_surface.npz
<out_dir>/<surface_run_name>_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_landscape_experiment_files.experimentC_utils import (  # noqa: E402
    compute_dataset_gradient_state,
    compute_dataset_second_moment_state,
    compute_loader_loss,
    copy_state,
    cuda_memory_cleanup,
    cosine_state,
    load_trainable_checkpoint,
    make_default_run_name,
    normalize_state,
    orthogonalize_state,
    power_iteration_hessian,
    prepare_common_experiment,
    set_trainable_state,
    set_trainable_state_linear_combo,
    state_mul_invsqrt,
    state_norm,
    state_scale,
    state_dot,
    random_state_like,
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Used only to scale the SGD-step unit.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--full_support_train", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for loss-only surface evaluation.")
    parser.add_argument("--grad_eval_batch_size", type=int, default=32, help="Batch size for full-gradient probe computations. Lower this if the fullgrad probe OOMs.")
    parser.add_argument("--surface_flush_cache_every", type=int, default=1, help="Clear CUDA cache every N grid points during surface measurement; <=0 disables it.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint_epoch", type=int, required=True)
    parser.add_argument("--probe", type=str, default="hessian", choices=["hessian", "random", "fullgrad", "adam"])
    parser.add_argument("--grid_size", type=int, default=25, help="Number of alpha/beta coordinates per axis.")
    parser.add_argument("--grid_radius", type=float, default=2.0, help="Extent in multiples of the actual SGD step norm.")
    parser.add_argument("--step_scale_mode", type=str, default="relative_to_sgd_step", choices=["relative_to_sgd_step", "absolute"])
    parser.add_argument("--absolute_scale", type=float, default=1.0, help="Used only when --step_scale_mode=absolute.")
    parser.add_argument("--num_power_iters", type=int, default=8)
    parser.add_argument("--hvp_mode", type=str, default="auto", choices=["auto", "autograd", "finite_difference"], help="How to compute Hessian-vector products for the hessian probe.")
    parser.add_argument("--hvp_fd_eps", type=float, default=1e-3, help="Finite-difference epsilon used when Hessian-vector products fall back to finite differences.")
    parser.add_argument("--adam_probe_eps", type=float, default=1e-8)
    parser.add_argument("--checkpoints_logdir", type=str, default="./results_without_aug/experimentC_logs")
    parser.add_argument("--out_dir", type=str, default="./results_without_aug/experimentC_surfaces")
    parser.add_argument("--run_name", type=str, default=None, help="Checkpoint run name. If omitted, the default ExperimentC name is inferred.")
    parser.add_argument("--surface_run_name", type=str, default=None, help="Optional explicit output run name for the measured surface.")
    return parser.parse_args()


def build_probe_direction(args, model, base_state, few_loader, full_loader, class_tokens, logit_scale, device, d_state):
    if args.probe == "random":
        raw = random_state_like(base_state, seed=args.seed + args.checkpoint_epoch + 123)
        raw_name = "random"
        extra = {}
    elif args.probe == "fullgrad":
        g_full, full_loss = compute_dataset_gradient_state(model, full_loader, class_tokens, logit_scale, device)
        raw = state_scale(g_full, -1.0)
        raw_name = "-fullgrad"
        extra = {"fullgrad_loss": full_loss}
    elif args.probe == "adam":
        g_few, few_loss = compute_dataset_gradient_state(model, few_loader, class_tokens, logit_scale, device)
        v_few = compute_dataset_second_moment_state(model, few_loader, class_tokens, logit_scale, device)
        raw = state_scale(state_mul_invsqrt(v_few, g_few, eps=args.adam_probe_eps), -1.0)
        raw_name = "-adam_like"
        extra = {"fewgrad_loss": few_loss}
    elif args.probe == "hessian":
        raw, eigval, hess_info = power_iteration_hessian(
            model=model,
            few_loader=few_loader,
            class_tokens=class_tokens,
            logit_scale=logit_scale,
            device=device,
            ref_state=base_state,
            num_iters=args.num_power_iters,
            seed=args.seed + args.checkpoint_epoch,
            hvp_mode=args.hvp_mode,
            fd_eps=args.hvp_fd_eps,
        )
        raw_name = "top_hessian_eigvec"
        extra = {"hessian_top_eigval": eigval, **hess_info}
    else:
        raise ValueError(f"Unsupported probe: {args.probe}")

    raw_cos = cosine_state(raw, d_state)
    e_state = normalize_state(orthogonalize_state(raw, d_state))
    ortho_cos = cosine_state(e_state, d_state)
    extra.update({
        "probe_raw_name": raw_name,
        "probe_raw_cos_to_sgd": raw_cos,
        "probe_orth_cos_to_sgd": ortho_cos,
        "probe_raw_norm": state_norm(raw),
        "probe_orth_norm": state_norm(e_state),
    })
    return e_state, extra


def measure_surface(args: argparse.Namespace) -> None:
    bundle = prepare_common_experiment(args)
    model = bundle["clip_model"]
    class_tokens = bundle["class_tokens"]
    logit_scale = bundle["logit_scale"]
    few_eval_loader = bundle["few_eval_loader"]
    full_eval_loader = bundle["full_eval_loader"]
    full_grad_loader = bundle.get("full_grad_loader", full_eval_loader)
    test_loader = bundle["test_loader"]

    checkpoints_logdir = Path(args.checkpoints_logdir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_run_name = args.run_name or make_default_run_name(
        prefix="experimentC",
        dataset=args.dataset,
        shots=args.shots,
        backbone=args.backbone,
        seed=args.seed,
        position=args.position,
        encoder=args.encoder,
        full_support_train=args.full_support_train,
    )
    ckpt_path = checkpoints_logdir / "checkpoints" / base_run_name / f"epoch_{args.checkpoint_epoch:02d}_trainable.pt"
    base_state, ckpt_meta = load_trainable_checkpoint(ckpt_path)
    set_trainable_state(model, base_state)

    # Actual local SGD direction around Theta_t: plain SGD => negative few-shot gradient.
    g_few, few_loss_center = compute_dataset_gradient_state(model, few_eval_loader, class_tokens, logit_scale, args.device)
    d_state = normalize_state(state_scale(g_few, -1.0))
    sgd_step_norm = args.lr * state_norm(g_few)

    e_state, probe_extra = build_probe_direction(
        args=args,
        model=model,
        base_state=base_state,
        few_loader=few_eval_loader,
        full_loader=full_grad_loader,
        class_tokens=class_tokens,
        logit_scale=logit_scale,
        device=args.device,
        d_state=d_state,
    )

    cuda_memory_cleanup(args.device)

    step_scale = sgd_step_norm if args.step_scale_mode == "relative_to_sgd_step" else args.absolute_scale
    alphas = np.linspace(-args.grid_radius, args.grid_radius, args.grid_size, dtype=np.float64)
    betas = np.linspace(-args.grid_radius, args.grid_radius, args.grid_size, dtype=np.float64)

    few_surface = np.zeros((args.grid_size, args.grid_size), dtype=np.float64)
    full_surface = np.zeros((args.grid_size, args.grid_size), dtype=np.float64)
    test_surface = np.zeros((args.grid_size, args.grid_size), dtype=np.float64)

    n_grid_points = int(args.grid_size * args.grid_size)
    done_points = 0
    for i, beta in enumerate(betas):
        print(f"[ExperimentC-surface] row {i + 1}/{args.grid_size} beta={float(beta):.4f}", flush=True)
        for j, alpha in enumerate(alphas):
            set_trainable_state_linear_combo(model, base_state, d_state, e_state, float(alpha), float(beta), float(step_scale))
            few_surface[i, j] = compute_loader_loss(model, few_eval_loader, class_tokens, logit_scale, args.device)
            full_surface[i, j] = compute_loader_loss(model, full_eval_loader, class_tokens, logit_scale, args.device)
            test_surface[i, j] = compute_loader_loss(model, test_loader, class_tokens, logit_scale, args.device)
            done_points += 1
            if args.surface_flush_cache_every > 0 and (done_points % args.surface_flush_cache_every == 0):
                cuda_memory_cleanup(args.device)

    cuda_memory_cleanup(args.device)

    # Restore checkpoint state after measurement.
    set_trainable_state(model, base_state)

    surface_run_name = args.surface_run_name or (
        f"surface_dataset_{args.dataset}_step_{args.checkpoint_epoch}_probe_{args.probe}_backbone_{args.backbone.replace('/', '-')}"
        f"_pos_{args.position}_enc_{args.encoder}"
    )

    npz_path = out_dir / f"{surface_run_name}_surface.npz"
    np.savez_compressed(
        npz_path,
        alphas=alphas,
        betas=betas,
        few_surface=few_surface,
        full_surface=full_surface,
        test_surface=test_surface,
    )

    full_loss_center = float(full_surface[np.argmin(np.abs(betas)), np.argmin(np.abs(alphas))])
    test_loss_center = float(test_surface[np.argmin(np.abs(betas)), np.argmin(np.abs(alphas))])

    meta = {
        "dataset": args.dataset,
        "shots": args.shots,
        "backbone": args.backbone,
        "position": args.position,
        "encoder": args.encoder,
        "optimizer": "SGD",
        "checkpoint_epoch": args.checkpoint_epoch,
        "probe": args.probe,
        "hvp_mode": args.hvp_mode,
        "hvp_fd_eps": args.hvp_fd_eps,
        "grid_size": args.grid_size,
        "grid_radius": args.grid_radius,
        "eval_batch_size": args.eval_batch_size,
        "grad_eval_batch_size": args.grad_eval_batch_size,
        "surface_flush_cache_every": args.surface_flush_cache_every,
        "step_scale_mode": args.step_scale_mode,
        "step_scale": float(step_scale),
        "sgd_step_norm": float(sgd_step_norm),
        "few_loss_center": float(few_loss_center),
        "full_loss_center": float(full_loss_center),
        "test_loss_center": float(test_loss_center),
        "run_name": base_run_name,
        "checkpoint_path": str(ckpt_path),
        "surface_npz": str(npz_path),
        **probe_extra,
    }
    with open(out_dir / f"{surface_run_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ExperimentC-surface] Wrote surface to: {npz_path}")
    print(f"[ExperimentC-surface] Wrote meta to: {out_dir / f'{surface_run_name}_meta.json'}")


if __name__ == "__main__":
    measure_surface(parse_args())

"""Fixed-protocol P5 SigLIP training and zero-shot execution."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from completion import atomic_csv, atomic_csv_gz, atomic_json, atomic_text
from instrumentation import (
    assert_training_batch_source,
    compute_gradient_moments,
    native_optimizer_step,
    p5_adam_m0_first_step,
    snapshot_parameters,
)
from metrics import (
    HISTOGRAM_EDGES,
    bias_corrected_adam_v,
    drift_metric_rows,
    evaluate,
    summarize_moments,
    update_metric_rows,
)
from run_utils import DIAGNOSTIC_EPOCHS, EVAL_EPOCHS, optimizer_config
from support_manifest import diagnostic_batches


EPOCH_FIELDS = [
    "epoch",
    "augmented_train_loss",
    "augmented_train_accuracy",
    "deterministic_support_loss",
    "deterministic_support_accuracy",
    "validation_loss",
    "validation_accuracy",
    "test_loss",
    "test_accuracy",
    "learning_rate",
    "optimizer_updates",
    "examples_processed",
]
UPDATE_FIELDS = [
    "step",
    "epoch",
    "batch",
    "batch_size",
    "scope",
    "tower",
    "block",
    "parameter_l2",
    "update_l2",
    "relative_budget",
    "learning_rate",
]
DRIFT_FIELDS = [
    "epoch",
    "scope",
    "tower",
    "block",
    "initial_l2",
    "delta_l2",
    "relative_drift",
]


def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_optimizer(variant: str, parameters: Sequence[torch.nn.Parameter]):
    if variant == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=1e-4,
            momentum=0.0,
            weight_decay=0.0,
            dampening=0.0,
            nesterov=False,
        )
    if variant in {"adam", "adam-m0"}:
        return torch.optim.Adam(
            parameters,
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    raise ValueError(f"Unknown trainable P5 variant: {variant}")


def _batch_hash(images: torch.Tensor, labels: torch.Tensor, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(images.detach().float().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    for path in paths:
        digest.update(path.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _evaluation_row(
    epoch: int,
    train_loss,
    train_accuracy,
    support_loader,
    validation_loader,
    test_loader,
    adapter,
    text_inputs,
    device,
    learning_rate,
    updates,
    examples,
):
    support = evaluate(adapter, support_loader, text_inputs, device)
    validation = evaluate(adapter, validation_loader, text_inputs, device)
    test = evaluate(adapter, test_loader, text_inputs, device)
    return (
        {
            "epoch": epoch,
            "augmented_train_loss": train_loss,
            "augmented_train_accuracy": train_accuracy,
            "deterministic_support_loss": support["loss"],
            "deterministic_support_accuracy": support["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "learning_rate": learning_rate,
            "optimizer_updates": updates,
            "examples_processed": examples,
        },
        test,
    )


def _diagnose(
    *,
    epoch: int,
    adapter,
    optimizer,
    scope_map,
    diagnostic_records,
    deterministic_transform,
    text_inputs,
    device,
    dataset_root,
):
    support_batches = diagnostic_batches(
        diagnostic_records, "support", dataset_root=dataset_root
    )
    full_batches = diagnostic_batches(
        diagnostic_records, "full_reference", dataset_root=dataset_root
    )
    few_m, few_v = compute_gradient_moments(
        adapter,
        support_batches,
        deterministic_transform,
        text_inputs,
        scope_map,
        device,
    )
    full_m, full_v = compute_gradient_moments(
        adapter,
        full_batches,
        deterministic_transform,
        text_inputs,
        scope_map,
        device,
    )
    ema_v = bias_corrected_adam_v(optimizer, scope_map)
    rows = summarize_moments(
        few_m,
        few_v,
        full_m,
        full_v,
        scope_map,
        epoch=epoch,
        ema_v=ema_v,
    )
    del few_m, few_v, full_m, full_v, ema_v
    return rows


def _diagnostic_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields = ["epoch", "scope", "tower", "block"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def run_training(
    *,
    args,
    variant: str,
    adapter,
    dataset,
    train_loader,
    support_loader,
    validation_loader,
    test_loader,
    text_inputs,
    scope_map,
    trainable_manifest,
    run_dir: Path,
    diagnostic_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if args.epochs != 30 and not args.smoke:
        raise AssertionError("P5 production must run exactly 30 epochs")
    if args.batch_size != 64 and not args.smoke:
        raise AssertionError("P5 nominal batch size must remain 64")
    parameters = [parameter for _, parameter in scope_map.named_parameters]
    optimizer = build_optimizer(variant, parameters)
    initial = snapshot_parameters(scope_map)
    epoch_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    diagnostic_global: List[Dict[str, Any]] = []
    diagnostic_tower: List[Dict[str, Any]] = []
    diagnostic_block: List[Dict[str, Any]] = []
    diagnostic_histograms: Dict[str, np.ndarray] = {
        "z_edges": HISTOGRAM_EDGES
    }
    training_order_hashes = []
    augmented_batch_hashes = []
    actual_batch_sizes: List[int] = []
    optimizer_updates = 0
    examples_processed = 0
    first_step: Dict[str, Any] = {}
    both_towers_received_gradients = False
    oracle_test_accuracy = -math.inf
    deterministic_transform = adapter.deterministic_transform()

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()

    row, test = _evaluation_row(
        0,
        None,
        None,
        support_loader,
        validation_loader,
        test_loader,
        adapter,
        text_inputs,
        args.device,
        float(optimizer.param_groups[0]["lr"]),
        optimizer_updates,
        examples_processed,
    )
    epoch_rows.append(row)
    oracle_test_accuracy = max(oracle_test_accuracy, test["accuracy"])
    drift_rows.extend(drift_metric_rows(initial, scope_map, 0))

    if diagnostic_records is not None:
        global_rows, tower_rows, block_rows, histograms = _diagnose(
            epoch=0,
            adapter=adapter,
            optimizer=optimizer,
            scope_map=scope_map,
            diagnostic_records=diagnostic_records,
            deterministic_transform=deterministic_transform,
            text_inputs=text_inputs,
            device=args.device,
            dataset_root=args.root_path,
        )
        diagnostic_global.extend(global_rows)
        diagnostic_tower.extend(tower_rows)
        diagnostic_block.extend(block_rows)
        diagnostic_histograms.update(histograms)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order_digest = hashlib.sha256()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch_index, (images, labels, paths) in enumerate(train_loader):
            assert_training_batch_source("support")
            actual_batch_sizes.append(int(labels.numel()))
            for path in paths:
                order_digest.update(path.encode())
                order_digest.update(b"\0")
            if epoch == 1 and batch_index < 2:
                augmented_batch_hashes.append(
                    {
                        "batch": batch_index,
                        "sha256": _batch_hash(images, labels, paths),
                    }
                )
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(args.device):
                image_embeddings = adapter.encode_image(images)
                text_embeddings = adapter.encode_text(text_inputs)
                logits = adapter.classification_logits(image_embeddings, text_embeddings)
                loss = F.cross_entropy(logits.float(), labels)
            loss.backward()

            if optimizer_updates == 0:
                gradients_by_tower = {"vision": 0, "text": 0}
                for entry in scope_map.entries:
                    if entry.parameter.grad is not None:
                        gradients_by_tower[entry.encoder] += int(
                            torch.count_nonzero(entry.parameter.grad.detach()).item()
                        )
                if min(gradients_by_tower.values()) <= 0:
                    raise RuntimeError(
                        f"Both towers must receive nonzero gradients: {gradients_by_tower}"
                    )
                both_towers_received_gradients = True

            if variant == "adam-m0" and optimizer_updates == 0:
                m0 = p5_adam_m0_first_step(
                    optimizer,
                    scope_map,
                    reference_sgd_lr=1e-4,
                    matching_tolerance=1e-5,
                )
                measurement = m0.applied_adam_step
                first_step = {
                    "raw_adam_update_l2": m0.raw_adam_budget.update_l2,
                    "raw_adam_relative_budget": m0.raw_adam_budget.relative_budget,
                    "sgd_reference_update_l2": m0.reference_sgd_budget.update_l2,
                    "sgd_reference_relative_budget": m0.reference_sgd_budget.relative_budget,
                    "raw_adam_to_sgd_budget_ratio": m0.raw_adam_budget.relative_budget
                    / (m0.reference_sgd_budget.relative_budget + 1e-30),
                    "analytic_gamma_0": m0.analytic_gamma_0,
                    "gamma_0": m0.gamma_0,
                    "native_lr_correction_factor": m0.gamma_0
                    / (m0.analytic_gamma_0 + 1e-30),
                    "native_calibration_evaluations": m0.native_calibration_evaluations,
                    "matched_learning_rate": m0.matched_learning_rate,
                    "applied_relative_budget": m0.applied_adam_budget.relative_budget,
                    "relative_matching_error": m0.relative_matching_error,
                    "adam_state_advances": 1,
                }
                # Keep only the applied-step measurement needed by the common
                # logger; the two exact temporary proposals otherwise retain
                # several full-size GPU snapshots for the rest of training.
                del m0
            else:
                measurement = native_optimizer_step(optimizer, scope_map)

            learning_rate = float(optimizer.param_groups[0]["lr"])
            update_rows.extend(
                update_metric_rows(
                    measurement,
                    scope_map,
                    step=optimizer_updates,
                    epoch=epoch,
                    batch_index=batch_index,
                    batch_size=int(labels.numel()),
                    learning_rate=learning_rate,
                )
            )
            if optimizer_updates == 0 and variant != "adam-m0":
                global_row = next(
                    row for row in update_rows if row["step"] == 0 and row["scope"] == "global"
                )
                first_step = {
                    "applied_update_l2": global_row["update_l2"],
                    "applied_relative_budget": global_row["relative_budget"],
                    "learning_rate": learning_rate,
                }
            optimizer_updates += 1
            examples_processed += int(labels.numel())
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            count += int(labels.numel())
            del measurement

        training_order_hashes.append(
            {"epoch": epoch, "sha256": order_digest.hexdigest()}
        )
        train_loss = loss_sum / count
        train_accuracy = 100.0 * correct / count
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if epoch in EVAL_EPOCHS or epoch == args.epochs:
            row, test = _evaluation_row(
                epoch,
                train_loss,
                train_accuracy,
                support_loader,
                validation_loader,
                test_loader,
                adapter,
                text_inputs,
                args.device,
                learning_rate,
                optimizer_updates,
                examples_processed,
            )
            oracle_test_accuracy = max(oracle_test_accuracy, test["accuracy"])
            drift_rows.extend(drift_metric_rows(initial, scope_map, epoch))
        else:
            row = {
                "epoch": epoch,
                "augmented_train_loss": train_loss,
                "augmented_train_accuracy": train_accuracy,
                "deterministic_support_loss": None,
                "deterministic_support_accuracy": None,
                "validation_loss": None,
                "validation_accuracy": None,
                "test_loss": None,
                "test_accuracy": None,
                "learning_rate": learning_rate,
                "optimizer_updates": optimizer_updates,
                "examples_processed": examples_processed,
            }
        epoch_rows.append(row)

        if diagnostic_records is not None and epoch in DIAGNOSTIC_EPOCHS:
            global_rows, tower_rows, block_rows, histograms = _diagnose(
                epoch=epoch,
                adapter=adapter,
                optimizer=optimizer,
                scope_map=scope_map,
                diagnostic_records=diagnostic_records,
                deterministic_transform=deterministic_transform,
                text_inputs=text_inputs,
                device=args.device,
                dataset_root=args.root_path,
            )
            diagnostic_global.extend(global_rows)
            diagnostic_tower.extend(tower_rows)
            diagnostic_block.extend(block_rows)
            diagnostic_histograms.update(histograms)

        atomic_csv(run_dir / "epoch_metrics.csv", EPOCH_FIELDS, epoch_rows)
        atomic_csv_gz(run_dir / "update_metrics.csv.gz", UPDATE_FIELDS, update_rows)
        atomic_csv(run_dir / "parameter_drift.csv", DRIFT_FIELDS, drift_rows)
        atomic_json(
            run_dir / "training_order_hashes.json",
            {
                "epochs": training_order_hashes,
                "first_two_augmented_batches": augmented_batch_hashes,
            },
        )

    final_row = next(row for row in epoch_rows if row["epoch"] == args.epochs)
    if final_row["test_accuracy"] is None:
        raise RuntimeError("Final fixed-epoch test accuracy was not evaluated")
    elapsed = time.monotonic() - start
    peak_memory = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0
    )
    summary = {
        "variant": variant,
        "final_epoch": args.epochs,
        "final_test_accuracy": final_row["test_accuracy"],
        "final_test_loss": final_row["test_loss"],
        "analysis_only_test_oracle_accuracy": oracle_test_accuracy,
        "total_optimizer_updates": optimizer_updates,
        "total_examples_processed": examples_processed,
        "support_size": len(dataset.train_x),
        "nominal_batch_size": args.batch_size,
        "actual_batch_sizes": sorted(set(actual_batch_sizes)),
        "number_of_batches_per_epoch": len(train_loader),
        "current_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "wall_clock_seconds": elapsed,
        "peak_allocated_gpu_memory_bytes": int(peak_memory),
        "trainable_parameter_count": trainable_manifest["trainable_parameter_count"],
        "both_towers_received_gradients": both_towers_received_gradients,
        "first_step": first_step,
        "weights_saved": bool(args.save_weights),
        "fixed_epoch_result": True,
        "early_stopping": False,
    }
    atomic_json(run_dir / "summary.json", summary)

    optimizer_name = variant
    compatibility_name = (
        f"final_results_dataset_{args.dataset}_shots_{args.shots}"
        f"_backbone_siglip-base-patch16-224_opt_{optimizer_name}_seed_{args.seed}.txt"
    )
    atomic_text(
        run_dir / compatibility_name,
        "Epoch,Dataset,Shots,Seed,Train Acc,Test Acc,Loss\n"
        f"{args.epochs},{args.dataset},{args.shots},{args.seed},"
        f"{final_row['augmented_train_accuracy']},{final_row['test_accuracy']},"
        f"{final_row['augmented_train_loss']}\n",
    )

    if diagnostic_records is not None:
        atomic_csv(
            run_dir / "diagnostic_global.csv",
            _diagnostic_fields(diagnostic_global),
            diagnostic_global,
        )
        atomic_csv(
            run_dir / "diagnostic_tower.csv",
            _diagnostic_fields(diagnostic_tower),
            diagnostic_tower,
        )
        atomic_csv(
            run_dir / "diagnostic_block.csv",
            _diagnostic_fields(diagnostic_block),
            diagnostic_block,
        )
        _write_npz_atomic(
            run_dir / "diagnostic_histograms.npz", diagnostic_histograms
        )
    if args.save_weights:
        torch.save(adapter.model.state_dict(), run_dir / "weights.pt")
    return summary


def run_zero_shot(
    *,
    args,
    adapter,
    test_loader,
    text_inputs,
    run_dir: Path,
) -> Dict[str, Any]:
    before = {
        name: parameter.detach().cpu().contiguous().numpy().tobytes()
        for name, parameter in adapter.model.named_parameters()
    }
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise AssertionError("Zero-shot model contains trainable parameters")
    start = time.monotonic()
    metrics = evaluate(adapter, test_loader, text_inputs, args.device)
    elapsed = time.monotonic() - start
    for name, parameter in adapter.model.named_parameters():
        after = parameter.detach().cpu().contiguous().numpy().tobytes()
        if after != before[name]:
            raise AssertionError(f"Zero-shot parameter changed: {name}")
    result = {
        "variant": "zero-shot",
        "dataset": args.dataset,
        "shots": 0,
        "seed": 0,
        "test_accuracy": metrics["accuracy"],
        "test_loss": metrics["loss"],
        "test_examples": metrics["examples"],
        "wall_clock_seconds": elapsed,
        "optimizer": None,
        "backward_calls": 0,
        "trainable_parameter_count": 0,
        "parameters_byte_identical": True,
        "native_siglip_logits": True,
    }
    atomic_json(run_dir / "zero_shot_metrics.json", result)
    filename = (
        f"final_results_dataset_{args.dataset}_shots_0"
        "_backbone_siglip-base-patch16-224_opt_zero-shot_seed_0.txt"
    )
    atomic_text(
        run_dir / filename,
        "Epoch,Dataset,Shots,Seed,Test Acc,Loss\n"
        f"0,{args.dataset},0,0,{metrics['accuracy']},{metrics['loss']}\n",
    )
    return result


__all__ = ["build_optimizer", "run_training", "run_zero_shot"]

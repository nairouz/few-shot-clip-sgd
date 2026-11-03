"""Entry point for one standalone NeurIPS 2026 P5 SigLIP condition."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from completion import (
    CompletionError,
    atomic_json,
    run_lock,
    sha256_file,
    validate_done,
    write_done,
)
from datasets import build_dataset
from environment import capture_environment
from model_adapter import (
    SigLIPAdapter,
    enable_dense_transformer_blocks,
    freeze_all,
    trainable_parameter_manifest,
)
from run_utils import (
    MODEL_ID,
    MODEL_REVISION,
    get_arguments,
    loader_seed,
    optimizer_config,
    seed_worker,
    set_random_seed,
    variant_from_directory,
)
from support_manifest import (
    P5ImageDataset,
    build_diagnostic_manifest,
    build_full_reference,
    canonical_records,
    write_manifest,
)
from train import run_training, run_zero_shot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoint" / "checkpoint_manifest.json"


def _verify_checkpoint(model_dir: Path, manifest):
    if manifest["hub_repository_id"] != MODEL_ID:
        raise RuntimeError("Checkpoint manifest repository mismatch")
    if manifest["immutable_revision"] != MODEL_REVISION:
        raise RuntimeError("Checkpoint manifest revision mismatch")
    for record in manifest["files"]:
        path = model_dir / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Pinned checkpoint file is missing: {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch for {path.name}")


def _run_id(args, variant):
    if args.run_id:
        return args.run_id
    if variant == "zero-shot":
        return f"{args.dataset}_shots_0_seed_0_zero-shot"
    return f"{args.dataset}_shots_{args.shots}_seed_{args.seed}_{variant}"


def _loader(items, transform, *, batch_size, shuffle, workers, generator=None):
    return DataLoader(
        P5ImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def execute(args, variant: str, run_dir: Path):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    root_path = str(Path(args.root_path).expanduser().resolve())
    if not Path(root_path).is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    # CLIP/FT/Plain parity invariant: seed, then immediately construct the
    # dataset. No Python-random-consuming operation is permitted in between.
    set_random_seed(args.seed)
    dataset = build_dataset(
        args.dataset,
        root_path,
        -1 if variant == "zero-shot" else args.shots,
        None,
    )

    model_dir = Path(args.model_id).expanduser().resolve()
    checkpoint_manifest = json.loads(_checkpoint_manifest_path().read_text())
    _verify_checkpoint(model_dir, checkpoint_manifest)
    runtime_checkpoint_manifest = dict(checkpoint_manifest)
    runtime_checkpoint_manifest["runtime_snapshot_path"] = str(model_dir)
    atomic_json(run_dir / "checkpoint_manifest.json", runtime_checkpoint_manifest)

    adapter = SigLIPAdapter.from_pretrained(
        str(model_dir), revision=args.revision, local_files_only=True
    )
    adapter.to(args.device)
    deterministic_transform = adapter.deterministic_transform()
    prompts = [
        dataset.template[0].format(classname.replace("_", " "))
        for classname in dataset.classnames
    ]
    text_inputs = adapter.tokenize_prompts(prompts, args.device)
    atomic_json(run_dir / "rendered_prompts.json", prompts)

    source_commit = args.source_commit or _git_commit()
    config = {
        "run_id": run_dir.name,
        "variant": variant,
        "dataset": args.dataset,
        "shots": args.shots,
        "seed": args.seed,
        "hub_model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_model_dir": str(model_dir),
        "source_commit": source_commit,
        "epochs": args.epochs,
        "nominal_batch_size": args.batch_size,
        "drop_last": False,
        "augmentation": variant != "zero-shot",
        "train_transform": {
            "RandomResizedCrop": {
                "size": 224,
                "scale": [0.08, 1.0],
                "interpolation": "BICUBIC",
            },
            "RandomHorizontalFlip": {"p": 0.5},
            "Normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
        "optimizer": optimizer_config(variant),
        "rendered_prompts": prompts,
        "prompt_template": dataset.template[0],
        "text_maximum_length": int(adapter.tokenizer.model_max_length),
        "native_scale_and_bias_frozen": True,
        "mixed_precision": {
            "autocast": "bfloat16",
            "grad_scaler": False,
            "model_parameters": "float32",
            "gradients": "float32",
            "norm_accumulation": "float64",
        },
        "evaluation_epochs": [0, 1, 2, 5, 10, 20, 30],
        "diagnostic_epochs": [0, 10, 20, 30] if args.mechanism else [],
        "mechanism": bool(args.mechanism),
        "save_weights": bool(args.save_weights),
        "learning_rate_schedule": None,
        "warmup": None,
        "gradient_clipping": None,
        "gradient_accumulation": None,
        "early_stopping": False,
        "final_result_policy": "fixed epoch 30 test accuracy",
    }
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "environment.json", capture_environment())

    test_loader = _loader(
        dataset.test,
        deterministic_transform,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.num_workers,
    )
    if variant == "zero-shot":
        freeze_all(adapter)
        result = run_zero_shot(
            args=args,
            adapter=adapter,
            test_loader=test_loader,
            text_inputs=text_inputs,
            run_dir=run_dir,
        )
        required = [
            "config.json",
            "environment.json",
            "checkpoint_manifest.json",
            "rendered_prompts.json",
            "zero_shot_metrics.json",
            (
                f"final_results_dataset_{args.dataset}_shots_0"
                "_backbone_siglip-base-patch16-224_opt_zero-shot_seed_0.txt"
            ),
        ]
        return result, required

    scope_map = enable_dense_transformer_blocks(adapter)
    parameter_manifest = trainable_parameter_manifest(adapter)
    atomic_json(run_dir / "trainable_parameters.json", parameter_manifest)
    support_records = canonical_records(
        dataset.train_x, "support", args.dataset, args.shots, args.seed
    )
    validation_records = canonical_records(
        dataset.val, "validation", args.dataset, args.shots, args.seed
    )
    write_manifest(run_dir, "support_manifest", support_records)
    write_manifest(run_dir, "validation_manifest", validation_records)

    generator = torch.Generator()
    generator.manual_seed(loader_seed(args.dataset, args.shots, args.seed))
    train_loader = _loader(
        dataset.train_x,
        adapter.training_transform(),
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        generator=generator,
    )
    support_loader = _loader(
        dataset.train_x,
        deterministic_transform,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.num_workers,
    )
    validation_loader = _loader(
        dataset.val,
        deterministic_transform,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.num_workers,
    )

    diagnostic_records = None
    if args.mechanism:
        full_dataset = build_full_reference(
            build_dataset,
            args.dataset,
            root_path,
            deterministic_transform,
            support_records,
        )
        full_records = canonical_records(
            full_dataset.train_x,
            "full_reference",
            args.dataset,
            args.shots,
            args.seed,
        )
        write_manifest(run_dir, "full_reference_manifest", full_records)
        diagnostic_records = build_diagnostic_manifest(
            dataset.train_x,
            full_dataset.train_x,
            args.dataset,
            args.shots,
            args.seed,
            batch_size=args.diagnostic_batch_size,
            batches=args.diagnostic_batches,
        )
        write_manifest(run_dir, "diagnostic_manifest_b16", diagnostic_records)

    result = run_training(
        args=args,
        variant=variant,
        adapter=adapter,
        dataset=dataset,
        train_loader=train_loader,
        support_loader=support_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        text_inputs=text_inputs,
        scope_map=scope_map,
        trainable_manifest=parameter_manifest,
        run_dir=run_dir,
        diagnostic_records=diagnostic_records,
    )
    compatibility = (
        f"final_results_dataset_{args.dataset}_shots_{args.shots}"
        f"_backbone_siglip-base-patch16-224_opt_{variant}_seed_{args.seed}.txt"
    )
    required = [
        "config.json",
        "environment.json",
        "checkpoint_manifest.json",
        "trainable_parameters.json",
        "rendered_prompts.json",
        "support_manifest.json",
        "support_manifest.sha256",
        "validation_manifest.json",
        "validation_manifest.sha256",
        "training_order_hashes.json",
        "epoch_metrics.csv",
        "update_metrics.csv.gz",
        "parameter_drift.csv",
        "summary.json",
        compatibility,
    ]
    if args.mechanism:
        required.extend(
            [
                "full_reference_manifest.json",
                "full_reference_manifest.sha256",
                "diagnostic_manifest_b16.json",
                "diagnostic_manifest_b16.sha256",
                "diagnostic_global.csv",
                "diagnostic_tower.csv",
                "diagnostic_block.csv",
                "diagnostic_histograms.npz",
            ]
        )
    return result, required


def main(argv=None):
    args = get_arguments(argv)
    variant = variant_from_directory()
    run_dir = Path(args.output_root).expanduser().resolve() / _run_id(args, variant)
    zero_shot = variant == "zero-shot"
    expected_epoch = 0 if zero_shot else args.epochs
    if args.resume:
        try:
            validate_done(run_dir, expected_epoch, zero_shot=zero_shot)
            print(f"Skipping valid completed run: {run_dir}")
            return 0
        except CompletionError:
            pass
    try:
        with run_lock(run_dir):
            if args.resume:
                try:
                    validate_done(run_dir, expected_epoch, zero_shot=zero_shot)
                    print(f"Skipping valid completed run: {run_dir}")
                    return 0
                except CompletionError:
                    pass
            # A retry must never expose a stale terminal marker while work is
            # in progress, nor leave FAILED.json beside a later valid DONE.
            with contextlib.suppress(FileNotFoundError):
                (run_dir / "DONE.json").unlink()
            with contextlib.suppress(FileNotFoundError):
                (run_dir / "FAILED.json").unlink()
            with (run_dir / "stdout.log").open("a", buffering=1) as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(
                    Tee(sys.stderr, log)
                ):
                    _, required = execute(args, variant, run_dir)
            required.append("stdout.log")
            write_done(
                run_dir,
                expected_epoch,
                required,
                zero_shot=zero_shot,
            )
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

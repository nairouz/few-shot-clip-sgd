# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Capture the immutable runtime and hardware provenance of one P5 run."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict

import torch


def _version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args):
    try:
        return subprocess.check_output(
            ["git"] + list(args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "huggingface-hub",
                "sentencepiece",
                "numpy",
                "Pillow",
            )
        },
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_CLUSTER_NAME",
                "SLURM_JOB_PARTITION",
            )
        },
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "mixed_precision": {
            "autocast_dtype": "bfloat16",
            "grad_scaler": False,
            "model_parameter_dtype": "float32",
            "gradient_dtype": "float32",
            "norm_accumulation_dtype": "float64",
        },
        "offline_environment": {
            key: os.environ.get(key)
            for key in (
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
            )
        },
    }


__all__ = ["capture_environment"]

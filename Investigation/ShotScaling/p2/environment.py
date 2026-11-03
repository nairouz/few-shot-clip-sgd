# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import os, platform, socket
from datetime import datetime, timezone
import numpy as np
import torch

def snapshot(pool: str, physical_gpu_id: str) -> dict:
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception as exc:
        tv = f"unavailable: {exc}"
    return {
        "hostname": socket.gethostname(), "pool": pool,
        "physical_gpu_id": physical_gpu_id,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
        "pytorch": torch.__version__, "torchvision": tv, "numpy": np.__version__,
        "clip_implementation": "vendored OpenAI CLIP snapshot",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
    }

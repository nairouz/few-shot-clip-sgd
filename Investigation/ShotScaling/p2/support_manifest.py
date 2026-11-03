from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .logging_utils import canonical_json_bytes, sha256_bytes


def _record(item, *, split: str, dataset: str, shots: int, seed: int) -> dict:
    path = Path(item.impath).expanduser().resolve()
    return {
        "class_name": str(item.classname),
        "dataset": dataset,
        "image_path": os.path.normpath(str(path)),
        "label": int(item.label),
        "seed": int(seed),
        "shots": int(shots),
        "split": split,
    }


def manifest(data: Iterable, *, split: str, dataset: str, shots: int, seed: int) -> list[dict]:
    records = [_record(x, split=split, dataset=dataset, shots=shots, seed=seed) for x in data]
    return sorted(records, key=lambda x: (x["image_path"], x["label"], x["class_name"]))


def manifest_digest(records: list[dict]) -> str:
    return sha256_bytes(canonical_json_bytes(records))


def image_id(item) -> str:
    return sha256_bytes(os.path.normpath(str(Path(item.impath).expanduser().resolve())).encode())


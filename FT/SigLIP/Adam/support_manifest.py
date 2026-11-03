# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Canonical support/reference manifests and P5 image datasets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from completion import atomic_json, atomic_text
from run_utils import diagnostic_seed


def normalized_path(path: str) -> str:
    """Return a staging-root-independent dataset-relative image identity."""
    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    dataset_roots = [index for index, part in enumerate(parts) if part == "datasets"]
    if dataset_roots:
        suffix = parts[dataset_roots[-1] + 1 :]
        if suffix:
            return Path(*suffix).as_posix()
    return resolved.as_posix()


def physical_path(path: str, dataset_root: Optional[str] = None) -> str:
    """Resolve a canonical manifest identity to a readable local path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if dataset_root is None:
        raise ValueError("dataset_root is required for a dataset-relative path")
    return str((Path(dataset_root).expanduser().resolve() / candidate).resolve())


def _record(item, split: str, dataset: str, shots: int, seed: int) -> Dict[str, Any]:
    return {
        "image_path": normalized_path(item.impath),
        "label": int(item.label),
        "class_name": str(item.classname),
        "split": split,
        "dataset": dataset,
        "shots": int(shots),
        "seed": int(seed),
    }


def canonical_records(
    items: Iterable[Any], split: str, dataset: str, shots: int, seed: int
) -> List[Dict[str, Any]]:
    records = [_record(item, split, dataset, shots, seed) for item in items]
    return sorted(
        records,
        key=lambda row: (
            row["image_path"],
            row["label"],
            row["class_name"],
            row["split"],
        ),
    )


def canonical_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        json.dumps(list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def manifest_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def write_manifest(run_dir: Path, stem: str, records: Sequence[Mapping[str, Any]]) -> str:
    run_dir = Path(run_dir)
    records = list(records)
    atomic_json(run_dir / f"{stem}.json", records)
    digest = manifest_digest(records)
    atomic_text(run_dir / f"{stem}.sha256", f"{digest}  {stem}.json\n")
    return digest


def assert_manifest_file(run_dir: Path, stem: str) -> str:
    run_dir = Path(run_dir)
    records = json.loads((run_dir / f"{stem}.json").read_text())
    digest = manifest_digest(records)
    expected = (run_dir / f"{stem}.sha256").read_text().split()[0]
    if digest != expected:
        raise RuntimeError(f"{stem} checksum mismatch: {digest} != {expected}")
    return digest


class P5ImageDataset(Dataset):
    """Dataset returning the source path so training order is auditable."""

    def __init__(self, items: Sequence[Any], transform) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item.impath) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(item.label), normalized_path(item.impath)


def build_full_reference(
    build_dataset,
    dataset_name: str,
    root_path: str,
    preprocess,
    support_records: Sequence[Mapping[str, Any]],
):
    """Build the complete training split without perturbing any global RNG."""
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        full_dataset = build_dataset(dataset_name, root_path, -1, preprocess)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    full_paths = {normalized_path(item.impath) for item in full_dataset.train_x}
    support_paths = {str(row["image_path"]) for row in support_records}
    missing = sorted(support_paths - full_paths)
    if missing:
        raise AssertionError(
            "Support paths absent from complete training split: " + ", ".join(missing[:5])
        )
    return full_dataset


def build_diagnostic_manifest(
    support_items: Sequence[Any],
    full_items: Sequence[Any],
    dataset: str,
    shots: int,
    seed: int,
    batch_size: int = 16,
    batches: int = 64,
) -> List[Dict[str, Any]]:
    support_by_label: Dict[int, List[Any]] = defaultdict(list)
    full_by_label: Dict[int, List[Any]] = defaultdict(list)
    for item in support_items:
        support_by_label[int(item.label)].append(item)
    for item in full_items:
        full_by_label[int(item.label)].append(item)
    labels = sorted(support_by_label)
    if labels != sorted(full_by_label):
        raise AssertionError("Support/full diagnostic label sets differ")
    rng = random.Random(diagnostic_seed(dataset, shots, seed))
    records: List[Dict[str, Any]] = []
    for batch_index in range(batches):
        for position in range(batch_size):
            label = rng.choice(labels)
            support = rng.choice(support_by_label[label])
            full = rng.choice(full_by_label[label])
            if int(support.label) != int(full.label):
                raise AssertionError("Paired diagnostic labels differ")
            records.append(
                {
                    "batch": batch_index,
                    "position": position,
                    "label": label,
                    "class_name": str(support.classname),
                    "support_path": normalized_path(support.impath),
                    "full_reference_path": normalized_path(full.impath),
                    "dataset": dataset,
                    "shots": int(shots),
                    "seed": int(seed),
                    "sampling": "class-balanced-with-replacement",
                    "preprocessing": "deterministic-pinned-siglip",
                }
            )
    return records


def diagnostic_batches(
    records: Sequence[Mapping[str, Any]], source: str, dataset_root: Optional[str] = None
) -> List[List[Tuple[str, int]]]:
    key = "support_path" if source == "support" else "full_reference_path"
    grouped: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for record in records:
        grouped[int(record["batch"])].append(
            (physical_path(str(record[key]), dataset_root), int(record["label"]))
        )
    return [grouped[index] for index in sorted(grouped)]


__all__ = [
    "normalized_path",
    "physical_path",
    "canonical_records",
    "canonical_bytes",
    "manifest_digest",
    "write_manifest",
    "assert_manifest_file",
    "P5ImageDataset",
    "build_full_reference",
    "build_diagnostic_manifest",
    "diagnostic_batches",
]

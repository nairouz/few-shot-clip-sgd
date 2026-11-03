# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import random
from collections import defaultdict
from .support_manifest import image_id

def _group(data):
    groups = defaultdict(list)
    for item in data:
        groups[int(item.label)].append(item)
    return groups

def diagnostic_seed(dataset: str, shots: int, seed: int, batch_size: int) -> int:
    text = f"p2:{dataset}:{shots}:{seed}:b{batch_size}"
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2**31)

def build_diagnostic_manifest(support, reference, *, batch_size: int, batches: int = 64, seed: int):
    if batch_size <= 0 or batches != 64:
        raise ValueError("P2 requires a positive diagnostic batch size and exactly 64 batches")
    few, full = _group(support), _group(reference)
    if not few or set(few) != set(full):
        raise ValueError("support and full reference must have identical class sets")
    classes, rng = sorted(few), random.Random(seed)
    records = []
    for index in range(batches):
        sequence = [rng.choice(classes) for _ in range(batch_size)]
        records.append({
            "batch_index": index, "class_sequence": sequence,
            "support_image_ids": [image_id(rng.choice(few[c])) for c in sequence],
            "full_reference_image_ids": [image_id(rng.choice(full[c])) for c in sequence],
        })
    return {"diagnostic_rng_seed": seed, "batch_size": batch_size,
            "number_of_batches": batches, "batches": records}

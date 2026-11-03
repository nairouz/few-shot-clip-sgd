#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Check the exact 112-cell completion contract."""

from __future__ import annotations

import argparse
import json

from analysis_common import VARIANT_ORDER, accuracy_records, atomic_json


def expected_keys():
    datasets = ("eurosat", "fgvc", "dtd", "oxford_pets")
    expected = {(dataset, 0, 0, "zero-shot") for dataset in datasets}
    expected |= {
        (dataset, shots, seed, variant)
        for dataset in datasets
        for shots in (1, 4, 16)
        for seed in (1, 2, 3)
        for variant in ("sgd", "adam", "adam-m0")
    }
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="tables/completeness.json")
    args = parser.parse_args()
    records = accuracy_records(args.input)
    seen = {(row["dataset"], row["shots"], row["seed"], row["variant"]) for row in records}
    expected = expected_keys()
    output = {
        "expected": 112,
        "complete": len(expected & seen),
        "missing": sorted(expected - seen),
        "unexpected": sorted(seen - expected),
        "duplicate_records": len(records) - len(seen),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

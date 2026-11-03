# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paired cell bootstrap utilities (seeds remain paired within each cell)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis_common import accuracy_records


def paired_cell_bootstrap(values, statistic, resamples=10000, seed=2026):
    cells = sorted(values)
    if not cells:
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan")}
    estimate = float(statistic([values[cell] for cell in cells]))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(cells), size=len(cells))
        samples[index] = statistic([values[cells[position]] for position in selected])
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"estimate": estimate, "lower": float(lower), "upper": float(upper), "resamples": resamples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    records = accuracy_records(args.input)
    grouped = defaultdict(dict)
    for row in records:
        if row["variant"] != "zero-shot":
            grouped[(row["dataset"], row["shots"], row["seed"])][row["variant"]] = row["accuracy"]
    values = defaultdict(list)
    for (dataset, shots, seed), variants in grouped.items():
        if {"sgd", "adam", "adam-m0"} <= set(variants):
            values[(dataset, shots)].append(
                (variants["sgd"], variants["adam"], variants["adam-m0"])
            )

    def closure(cells):
        outputs = []
        for cell in cells:
            for sgd, adam, m0 in cell:
                if abs(sgd - adam) > 1e-12:
                    outputs.append(100.0 * (m0 - adam) / (sgd - adam))
        return float(np.mean(outputs))

    print(json.dumps(paired_cell_bootstrap(values, closure, args.resamples), indent=2))


if __name__ == "__main__":
    main()

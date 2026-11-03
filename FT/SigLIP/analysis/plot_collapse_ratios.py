#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Figure 9: denominator/numerator collapse on the Dense SGD trajectory."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_common import finish, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    data = [row for row in rows(Path(args.tables) / "mechanism_primary.csv") if row["scope"] == "global"]
    per_seed = [
        row
        for row in rows(Path(args.tables) / "mechanism_primary_per_seed.csv")
        if row["scope"] == "global"
    ]
    labels = [f"{row['dataset']} K={int(row['shots'])}" for row in data]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    samples = [
        [
            seed_row["collapse_ratio_dense_sgd"]
            for seed_row in per_seed
            if seed_row["dataset"] == row["dataset"]
            and int(seed_row["shots"]) == int(row["shots"])
        ]
        for row in data
    ]
    errors = [
        4.302652729911275 * statistics.stdev(sample) / math.sqrt(len(sample))
        for sample in samples
    ]
    positions = list(range(len(labels)))
    axis.bar(
        positions,
        [row["collapse_ratio_dense_sgd"] for row in data],
        yerr=errors,
        capsize=3,
        color="#1f77b4",
    )
    for position, sample in zip(positions, samples):
        axis.scatter(
            [position] * len(sample),
            sample,
            color="black",
            alpha=0.38,
            s=11,
            zorder=3,
        )
    axis.set_xticks(positions, labels=labels)
    axis.axhline(1, color="black", linestyle="--")
    axis.set_ylabel("$C_v/C_m$ on Dense SGD")
    axis.tick_params(axis="x", rotation=25)
    finish(fig, Path(args.output_dir) / "collapse_ratio_dense_sgd.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Figure 3: SGD-Adam accuracy gap versus shots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_common import finish, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    cells = rows(Path(args.tables) / "accuracy_by_cell.csv")
    seeds = rows(Path(args.tables) / "accuracy_per_seed.csv")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for dataset in ("eurosat", "fgvc", "dtd", "oxford_pets"):
        selected = sorted(
            [row for row in cells if row["dataset"] == dataset],
            key=lambda row: row["shots"],
        )
        values = [row["sgd_minus_adam"] for row in selected]
        axis.errorbar(
            [row["shots"] for row in selected],
            values,
            yerr=[
                [
                    value - row["sgd_minus_adam_ci95_lower"]
                    for value, row in zip(values, selected)
                ],
                [
                    row["sgd_minus_adam_ci95_upper"] - value
                    for value, row in zip(values, selected)
                ],
            ],
            marker="o",
            capsize=3,
            label=dataset,
        )
        individual = [row for row in seeds if row["dataset"] == dataset]
        axis.scatter(
            [row["shots"] for row in individual],
            [row["sgd_minus_adam"] for row in individual],
            alpha=0.32,
            s=14,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 4, 16], labels=["1", "4", "16"])
    axis.set_xlabel("Shots per class")
    axis.set_ylabel("Dense SGD - Adam accuracy (pp)")
    axis.legend(ncol=2)
    finish(fig, Path(args.output_dir) / "sgd_adam_gap_vs_shots.png")


if __name__ == "__main__":
    main()

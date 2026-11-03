#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Figures 5-7 and 11: initialization and vision/text comparisons."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_common import COLORS, finish, rows


METRICS = (
    ("initial_r_v", "initial_r_v_by_tower.png", "Initialization $R_v$"),
    ("initial_coordinate_mismatch_m", "initial_coordinate_mismatch_by_tower.png", "Coordinate mismatch $M(0)$"),
    ("initial_r_u", "initial_adaptive_norm_inflation.png", "Initialization $R_u$"),
)


def half_width(values):
    return 4.302652729911275 * statistics.stdev(values) / math.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    data = [
        row
        for row in rows(Path(args.tables) / "mechanism_primary.csv")
        if row["scope"] == "tower"
    ]
    per_seed = [
        row
        for row in rows(Path(args.tables) / "mechanism_primary_per_seed.csv")
        if row["scope"] == "tower"
    ]
    labels = sorted({f"{row['dataset']} K={int(row['shots'])}" for row in data})
    for metric, filename, ylabel in METRICS:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        width = 0.36
        x = list(range(len(labels)))
        for offset, tower in ((-width / 2, "vision"), (width / 2, "text")):
            values = []
            errors = []
            for label in labels:
                dataset, shot = label.split(" K=")
                match = next(row for row in data if row["dataset"] == dataset and int(row["shots"]) == int(shot) and row["tower"] == tower)
                values.append(match[metric])
                seed_values = [
                    row[metric]
                    for row in per_seed
                    if row["dataset"] == dataset
                    and int(row["shots"]) == int(shot)
                    and row["tower"] == tower
                ]
                errors.append(half_width(seed_values))
            positions = [value + offset for value in x]
            axis.bar(
                positions,
                values,
                yerr=errors,
                capsize=3,
                width=width,
                color=COLORS[tower],
                label=tower,
            )
            for position, label in zip(positions, labels):
                dataset, shot = label.split(" K=")
                seed_values = [
                    row[metric]
                    for row in per_seed
                    if row["dataset"] == dataset
                    and int(row["shots"]) == int(shot)
                    and row["tower"] == tower
                ]
                axis.scatter(
                    [position] * len(seed_values),
                    seed_values,
                    color="black",
                    alpha=0.38,
                    s=11,
                    zorder=3,
                )
        axis.set_xticks(x, labels=labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend()
        finish(fig, Path(args.output_dir) / filename)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (metric, _, ylabel) in zip(axes, METRICS):
        vision = [row[metric] for row in data if row["tower"] == "vision"]
        text = [row[metric] for row in data if row["tower"] == "text"]
        axis.scatter(vision, text)
        lower = min(vision + text)
        upper = max(vision + text)
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
        axis.set_xlabel("Vision")
        axis.set_ylabel("Text")
        axis.set_title(ylabel)
    finish(fig, Path(args.output_dir) / "vision_vs_text_mechanism.png")


if __name__ == "__main__":
    main()

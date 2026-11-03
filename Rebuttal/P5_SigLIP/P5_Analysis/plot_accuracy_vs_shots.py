#!/usr/bin/env python3
"""Figures 1-2: dataset panels and four-dataset average accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_common import COLORS, MARKERS, finish, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    cells = rows(Path(args.tables) / "accuracy_by_cell.csv")
    seeds = rows(Path(args.tables) / "accuracy_per_seed.csv")
    datasets = ("eurosat", "fgvc", "dtd", "oxford_pets")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, dataset in zip(axes.flat, datasets):
        selected = sorted(
            [row for row in cells if row["dataset"] == dataset],
            key=lambda row: row["shots"],
        )
        for variant, field in (("sgd", "sgd_mean"), ("adam", "adam_mean"), ("adam-m0", "adam_m0_mean")):
            prefix = variant.replace("-", "_")
            x = [row["shots"] for row in selected]
            y = [row[field] for row in selected]
            lower = [row[f"{prefix}_ci95_lower"] for row in selected]
            upper = [row[f"{prefix}_ci95_upper"] for row in selected]
            axis.errorbar(
                x,
                y,
                yerr=[
                    [center - bound for center, bound in zip(y, lower)],
                    [bound - center for center, bound in zip(y, upper)],
                ],
                marker=MARKERS[variant],
                color=COLORS[variant],
                label=variant,
                capsize=3,
            )
            seed_field = f"{prefix}_accuracy"
            offset = {"sgd": 0.92, "adam": 1.0, "adam-m0": 1.08}[variant]
            individual = [row for row in seeds if row["dataset"] == dataset]
            axis.scatter(
                [row["shots"] * offset for row in individual],
                [row[seed_field] for row in individual],
                color=COLORS[variant],
                alpha=0.38,
                s=13,
                zorder=3,
            )
        if selected:
            axis.axhline(selected[0]["zero_shot_accuracy"], color=COLORS["zero-shot"], linestyle="--", label="zero-shot (one evaluation)")
        axis.set_title(dataset)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        axis.set_ylabel("Accuracy (%)")
    axes[-1, 0].set_xlabel("Shots per class")
    axes[-1, 1].set_xlabel("Shots per class")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    finish(fig, Path(args.output_dir) / "accuracy_vs_shots_by_dataset.png")

    averages = rows(Path(args.tables) / "accuracy_averages.csv")
    shot_rows = sorted(
        [
            row
            for row in averages
            if row["aggregation"] == "four_dataset_average"
        ],
        key=lambda row: int(str(row["key"]).split("=")[1]),
    )
    confidence = rows(Path(args.tables) / "accuracy_confidence_intervals.csv")
    fig, axis = plt.subplots(figsize=(6, 4))
    for variant, field in (("sgd", "sgd_mean"), ("adam", "adam_mean"), ("adam-m0", "adam_m0_mean")):
        metric = f"{variant.replace('-', '_')}_accuracy"
        intervals = [
            next(
                row
                for row in confidence
                if row["aggregation"] == "shot"
                and row["key"] == f"K={shots}"
                and row["metric"] == metric
            )
            for shots in (1, 4, 16)
        ]
        values = [row[field] for row in shot_rows]
        axis.errorbar(
            [1, 4, 16],
            values,
            yerr=[
                [value - row["lower"] for value, row in zip(values, intervals)],
                [row["upper"] - value for value, row in zip(values, intervals)],
            ],
            marker=MARKERS[variant],
            color=COLORS[variant],
            label=variant,
            capsize=3,
        )
        seed_field = f"{variant.replace('-', '_')}_accuracy"
        for seed in (1, 2, 3):
            seed_means = []
            for shots in (1, 4, 16):
                values_for_seed = [
                    row[seed_field]
                    for row in seeds
                    if int(row["seed"]) == seed and int(row["shots"]) == shots
                ]
                seed_means.append(sum(values_for_seed) / len(values_for_seed))
            axis.scatter(
                [1, 4, 16],
                seed_means,
                color=COLORS[variant],
                alpha=0.35,
                s=14,
            )
    zero_average = sum(
        row["zero_shot_accuracy"]
        for row in cells
        if int(row["shots"]) == 1
    ) / len(datasets)
    axis.axhline(
        zero_average,
        color=COLORS["zero-shot"],
        linestyle="--",
        label="zero-shot (one evaluation/dataset)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 4, 16], labels=["1", "4", "16"])
    axis.set_xlabel("Shots per class")
    axis.set_ylabel("Four-dataset mean accuracy (%)")
    axis.legend()
    finish(fig, Path(args.output_dir) / "average_accuracy_vs_shots.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Figures 8 and 10: first-step ratios and Adam/M0 budget trajectories."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from analysis_common import load_run, run_directories
from _plot_common import COLORS, finish, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--input")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    mechanism = [
        row
        for row in rows(Path(args.tables) / "mechanism_primary.csv")
        if row["scope"] == "global"
    ]
    per_seed = [
        row
        for row in rows(Path(args.tables) / "mechanism_primary_per_seed.csv")
        if row["scope"] == "global"
    ]
    labels = [f"{row['dataset']} K={int(row['shots'])}" for row in mechanism]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(labels)))
    width = 0.36
    for offset, field, color, label in (
        (-width / 2, "actual_first_step_adam_to_sgd", COLORS["adam"], "Adam/SGD"),
        (width / 2, "actual_first_step_adam_m0_to_sgd", COLORS["adam-m0"], "Adam-M0/SGD"),
    ):
        values = [row[field] for row in mechanism]
        samples = [
            [
                seed_row[field]
                for seed_row in per_seed
                if seed_row["dataset"] == row["dataset"]
                and int(seed_row["shots"]) == int(row["shots"])
            ]
            for row in mechanism
        ]
        errors = [
            4.302652729911275 * statistics.stdev(sample) / math.sqrt(len(sample))
            for sample in samples
        ]
        positions = [value + offset for value in x]
        axis.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=label,
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
    axis.axhline(1, color="black", linestyle="--")
    axis.set_xticks(x, labels=labels, rotation=25, ha="right")
    axis.set_ylabel("Actual first-step relative budget ratio")
    axis.legend()
    finish(fig, Path(args.output_dir) / "first_step_budget_ratios.png")

    if not args.input:
        return
    trajectory = defaultdict(lambda: defaultdict(list))
    for run_dir in run_directories(args.input):
        try:
            config, _ = load_run(run_dir)
        except Exception:
            continue
        if config["variant"] not in {"adam", "adam-m0"}:
            continue
        path = run_dir / "update_metrics.csv.gz"
        if not path.is_file():
            continue
        with gzip.open(path, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["scope"] == "global":
                    trajectory[config["variant"]][int(row["step"])].append(float(row["relative_budget"]))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for variant in ("adam", "adam-m0"):
        steps = sorted(trajectory[variant])
        values = [
            statistics.mean(trajectory[variant][step]) for step in steps
        ]
        errors = [
            (
                1.96
                * statistics.stdev(trajectory[variant][step])
                / math.sqrt(len(trajectory[variant][step]))
                if len(trajectory[variant][step]) > 1
                else 0.0
            )
            for step in steps
        ]
        axis.plot(steps, values, color=COLORS[variant], label=variant)
        axis.fill_between(
            steps,
            [value - error for value, error in zip(values, errors)],
            [value + error for value, error in zip(values, errors)],
            color=COLORS[variant],
            alpha=0.16,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Actual global update budget")
    axis.legend()
    finish(fig, Path(args.output_dir) / "adam_vs_adam_m0_budget_trajectories.png")


if __name__ == "__main__":
    main()

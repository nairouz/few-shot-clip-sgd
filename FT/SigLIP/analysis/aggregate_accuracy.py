#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Aggregate fixed epoch-30 P5 accuracies and signed Adam-M0 gap closure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis_common import accuracy_records, atomic_json, markdown_table, write_csv
from bootstrap import paired_cell_bootstrap


def mean(values):
    return float(statistics.mean(values))


def std(values):
    return float(statistics.stdev(values)) if len(values) > 1 else None


def ci95(values):
    """Small-sample two-sided t interval for the three fixed support seeds."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return (None, None)
    critical = 4.302652729911275 if len(finite) == 3 else 12.706204736432095
    half_width = critical * statistics.stdev(finite) / math.sqrt(len(finite))
    center = mean(finite)
    return center - half_width, center + half_width


def closure(m0, adam, sgd):
    return float("nan") if abs(sgd - adam) <= 1e-12 else 100.0 * (m0 - adam) / (sgd - adam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="tables")
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    records = accuracy_records(args.input)
    zero = {row["dataset"]: row["accuracy"] for row in records if row["variant"] == "zero-shot"}
    grouped = defaultdict(list)
    per_seed = defaultdict(dict)
    for row in records:
        if row["variant"] == "zero-shot":
            continue
        grouped[(row["dataset"], row["shots"], row["variant"])].append(row["accuracy"])
        per_seed[(row["dataset"], row["shots"], row["seed"])][row["variant"]] = row["accuracy"]
    rows = []
    closure_seed_rows = []
    for dataset in ("eurosat", "fgvc", "dtd", "oxford_pets"):
        for shots in (1, 4, 16):
            values = {
                variant: grouped[(dataset, shots, variant)]
                for variant in ("sgd", "adam", "adam-m0")
            }
            if any(len(items) != 3 for items in values.values()):
                continue
            means = {variant: mean(items) for variant, items in values.items()}
            intervals = {variant: ci95(items) for variant, items in values.items()}
            closures = []
            seed_gaps = {
                "sgd_minus_adam": [],
                "adam_m0_minus_adam": [],
                "sgd_minus_adam_m0": [],
            }
            for seed in (1, 2, 3):
                variants = per_seed[(dataset, shots, seed)]
                if {"sgd", "adam", "adam-m0"} <= set(variants):
                    value = closure(variants["adam-m0"], variants["adam"], variants["sgd"])
                    if math.isfinite(value):
                        closures.append(value)
                    seed_gaps["sgd_minus_adam"].append(
                        variants["sgd"] - variants["adam"]
                    )
                    seed_gaps["adam_m0_minus_adam"].append(
                        variants["adam-m0"] - variants["adam"]
                    )
                    seed_gaps["sgd_minus_adam_m0"].append(
                        variants["sgd"] - variants["adam-m0"]
                    )
                    closure_seed_rows.append(
                        {
                            "dataset": dataset,
                            "shots": shots,
                            "seed": seed,
                            "sgd_accuracy": variants["sgd"],
                            "adam_accuracy": variants["adam"],
                            "adam_m0_accuracy": variants["adam-m0"],
                            "sgd_minus_adam": variants["sgd"] - variants["adam"],
                            "adam_m0_minus_adam": variants["adam-m0"] - variants["adam"],
                            "sgd_minus_adam_m0": variants["sgd"] - variants["adam-m0"],
                            "gap_closure_percent": value,
                        }
                    )
            rows.append(
                {
                    "dataset": dataset,
                    "shots": shots,
                    "zero_shot_accuracy": zero.get(dataset),
                    "zero_shot_std": None,
                    "sgd_mean": means["sgd"],
                    "sgd_std": std(values["sgd"]),
                    "sgd_ci95_lower": intervals["sgd"][0],
                    "sgd_ci95_upper": intervals["sgd"][1],
                    "adam_mean": means["adam"],
                    "adam_std": std(values["adam"]),
                    "adam_ci95_lower": intervals["adam"][0],
                    "adam_ci95_upper": intervals["adam"][1],
                    "adam_m0_mean": means["adam-m0"],
                    "adam_m0_std": std(values["adam-m0"]),
                    "adam_m0_ci95_lower": intervals["adam-m0"][0],
                    "adam_m0_ci95_upper": intervals["adam-m0"][1],
                    "sgd_minus_adam": means["sgd"] - means["adam"],
                    "sgd_minus_adam_ci95_lower": ci95(seed_gaps["sgd_minus_adam"])[0],
                    "sgd_minus_adam_ci95_upper": ci95(seed_gaps["sgd_minus_adam"])[1],
                    "adam_m0_minus_adam": means["adam-m0"] - means["adam"],
                    "sgd_minus_adam_m0": means["sgd"] - means["adam-m0"],
                    "adam_m0_gap_closure_percent": mean(closures) if closures else float("nan"),
                    "adam_m0_gap_closure_ci95_lower": ci95(closures)[0],
                    "adam_m0_gap_closure_ci95_upper": ci95(closures)[1],
                }
            )

    average_rows = []
    for shots in (1, 4, 16):
        selected = [row for row in rows if row["shots"] == shots]
        if selected:
            average_rows.append(
                {
                    "aggregation": "four_dataset_average",
                    "key": f"K={shots}",
                    **{
                        key: mean([row[key] for row in selected])
                        for key in ("sgd_mean", "adam_mean", "adam_m0_mean", "sgd_minus_adam", "adam_m0_gap_closure_percent")
                    },
                }
            )
    for dataset in ("eurosat", "fgvc", "dtd", "oxford_pets"):
        selected = [row for row in rows if row["dataset"] == dataset]
        if selected:
            average_rows.append(
                {
                    "aggregation": "three_shot_average",
                    "key": dataset,
                    **{
                        key: mean([row[key] for row in selected])
                        for key in ("sgd_mean", "adam_mean", "adam_m0_mean", "sgd_minus_adam", "adam_m0_gap_closure_percent")
                    },
                }
            )
    if rows:
        average_rows.append(
            {
                "aggregation": "macro_12_cell_average",
                "key": "all",
                **{
                    key: mean([row[key] for row in rows])
                    for key in ("sgd_mean", "adam_mean", "adam_m0_mean", "sgd_minus_adam", "adam_m0_gap_closure_percent")
                },
            }
        )

    bootstrap_values = defaultdict(list)
    for row in closure_seed_rows:
        bootstrap_values[(row["dataset"], row["shots"])].append(
            (row["sgd_accuracy"], row["adam_accuracy"], row["adam_m0_accuracy"])
        )

    statistics_by_name = {
        "sgd_accuracy": lambda triples: float(np.mean([value[0] for value in triples])),
        "adam_accuracy": lambda triples: float(np.mean([value[1] for value in triples])),
        "adam_m0_accuracy": lambda triples: float(np.mean([value[2] for value in triples])),
        "sgd_minus_adam": lambda triples: float(
            np.mean([sgd - adam for sgd, adam, _ in triples])
        ),
        "adam_m0_minus_adam": lambda triples: float(
            np.mean([m0 - adam for _, adam, m0 in triples])
        ),
        "sgd_minus_adam_m0": lambda triples: float(
            np.mean([sgd - m0 for sgd, _, m0 in triples])
        ),
        "adam_m0_gap_closure_percent": lambda triples: float(
            np.mean(
                [
                    closure(m0, adam, sgd)
                    for sgd, adam, m0 in triples
                    if abs(sgd - adam) > 1e-12
                ]
            )
        ),
    }

    def statistic(function):
        return lambda cells: function([triple for cell in cells for triple in cell])

    ci_rows = []
    groupings = [("macro", "all", lambda _: True)]
    groupings.extend(
        ("shot", f"K={shots}", lambda key, shots=shots: key[1] == shots)
        for shots in (1, 4, 16)
    )
    groupings.extend(
        ("dataset", dataset, lambda key, dataset=dataset: key[0] == dataset)
        for dataset in ("eurosat", "fgvc", "dtd", "oxford_pets")
    )
    for aggregation, key, predicate in groupings:
        selected = {
            cell: triples
            for cell, triples in bootstrap_values.items()
            if predicate(cell)
        }
        for index, (name, function) in enumerate(statistics_by_name.items()):
            interval = paired_cell_bootstrap(
                selected,
                statistic(function),
                resamples=args.bootstrap_resamples,
                seed=2026 + index,
            )
            ci_rows.append(
                {
                    "aggregation": aggregation,
                    "key": key,
                    "metric": name,
                    **interval,
                }
            )
    closure_ci = next(
        {
            key: row[key]
            for key in ("estimate", "lower", "upper", "resamples")
        }
        for row in ci_rows
        if row["aggregation"] == "macro"
        and row["metric"] == "adam_m0_gap_closure_percent"
    )
    write_csv(output / "accuracy_by_cell.csv", rows)
    write_csv(output / "accuracy_per_seed.csv", closure_seed_rows)
    write_csv(output / "accuracy_averages.csv", average_rows)
    write_csv(output / "accuracy_confidence_intervals.csv", ci_rows)
    atomic_json(
        output / "accuracy_tables.json",
        {
            "cells": rows,
            "per_seed": closure_seed_rows,
            "averages": average_rows,
            "paired_confidence_intervals": ci_rows,
            "paired_gap_closure_95ci": closure_ci,
        },
    )
    (output / "accuracy_by_cell.md").write_text(
        markdown_table(
            rows,
            [
                ("dataset", "Dataset"),
                ("shots", "K"),
                ("zero_shot_accuracy", "Zero-shot"),
                ("sgd_mean", "SGD mean"),
                ("sgd_std", "SGD std"),
                ("adam_mean", "Adam mean"),
                ("adam_std", "Adam std"),
                ("adam_m0_mean", "Adam-M0 mean"),
                ("adam_m0_std", "Adam-M0 std"),
                ("sgd_minus_adam", "SGD-Adam"),
                ("adam_m0_gap_closure_percent", "M0 closure %"),
            ],
        )
    )
    print(json.dumps({"cells": len(rows), "gap_closure_ci": closure_ci}, indent=2))


if __name__ == "__main__":
    main()

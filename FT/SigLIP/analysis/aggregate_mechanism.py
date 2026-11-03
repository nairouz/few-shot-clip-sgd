#!/usr/bin/env python3
"""Aggregate P5 initialization, collapse, budget, and closure mechanisms."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analysis_common import (
    accuracy_records,
    load_run,
    markdown_table,
    run_directories,
    write_csv,
)


def number(value):
    if value in {"", None, "None"}:
        return None
    return float(value)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_gzip_csv(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else float("nan")


def scope_key(row):
    if row["scope"] == "global":
        return ("global", None)
    return ("tower", row["tower"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="tables")
    args = parser.parse_args()
    output = Path(args.output_dir)
    diagnostics = defaultdict(list)
    budgets = defaultdict(dict)
    block_rows = []
    for run_dir in run_directories(args.input):
        try:
            config, summary = load_run(run_dir)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if not config.get("mechanism"):
            continue
        identity = (
            config["dataset"],
            int(config["shots"]),
            int(config["seed"]),
            config["variant"],
        )
        for filename in ("diagnostic_global.csv", "diagnostic_tower.csv"):
            path = run_dir / filename
            if not path.is_file():
                continue
            for row in read_csv(path):
                parsed = {}
                for key, value in row.items():
                    if key == "tower":
                        parsed[key] = value or None
                    elif key in {"scope", "ema_state_available"}:
                        parsed[key] = value
                    else:
                        parsed[key] = number(value)
                diagnostics[identity].append(parsed)
        block_path = run_dir / "diagnostic_block.csv"
        if block_path.is_file():
            for row in read_csv(block_path):
                block_rows.append(
                    {
                        "dataset": config["dataset"],
                        "shots": int(config["shots"]),
                        "seed": int(config["seed"]),
                        "variant": config["variant"],
                        **row,
                    }
                )
        update_path = run_dir / "update_metrics.csv.gz"
        if update_path.is_file():
            for row in read_gzip_csv(update_path):
                if int(row["step"]) == 0 and row["scope"] in {"global", "tower"}:
                    budgets[(config["dataset"], int(config["shots"]), int(config["seed"]), row["scope"], row["tower"] or None)][
                        config["variant"]
                    ] = float(row["relative_budget"])

    accuracies = defaultdict(dict)
    for row in accuracy_records(args.input):
        if row["variant"] != "zero-shot":
            accuracies[(row["dataset"], row["shots"], row["seed"])][row["variant"]] = row["accuracy"]

    checkpoint_rows = []
    collapse_rows = []
    for identity, rows_for_run in diagnostics.items():
        dataset, shots, seed, variant = identity
        for row in rows_for_run:
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "shots": shots,
                    "seed": seed,
                    "variant": variant,
                    **row,
                }
            )
        scopes = sorted(
            {(row["scope"], row.get("tower")) for row in rows_for_run}
        )
        for level, tower in scopes:
            selected = [
                row
                for row in rows_for_run
                if row["scope"] == level and row.get("tower") == tower
            ]
            initial = next(
                (row for row in selected if int(row["epoch"]) == 0), None
            )
            final = next(
                (row for row in selected if int(row["epoch"]) == 30), None
            )
            if initial is None or final is None:
                continue
            c_m = initial["r_m"] / (final["r_m"] + 1e-30)
            c_v = initial["r_v"] / (final["r_v"] + 1e-30)
            collapse_rows.append(
                {
                    "dataset": dataset,
                    "shots": shots,
                    "seed": seed,
                    "variant": variant,
                    "scope": level,
                    "tower": tower,
                    "c_m": c_m,
                    "c_v": c_v,
                    "collapse_ratio": c_v / (c_m + 1e-30),
                }
            )

    primary_per_seed = []
    for dataset in ("eurosat", "dtd"):
        for shots in (1, 16):
            for level, tower in (("global", None), ("tower", "vision"), ("tower", "text")):
                for seed in (1, 2, 3):
                    rows = diagnostics[(dataset, shots, seed, "sgd")]
                    selected = [
                        row
                        for row in rows
                        if row["scope"] == level and row.get("tower") == tower
                    ]
                    initial = next(
                        (row for row in selected if int(row["epoch"]) == 0),
                        None,
                    )
                    final = next(
                        (row for row in selected if int(row["epoch"]) == 30),
                        None,
                    )
                    if initial is None or final is None:
                        continue
                    c_m = initial["r_m"] / (final["r_m"] + 1e-30)
                    c_v = initial["r_v"] / (final["r_v"] + 1e-30)
                    key = (dataset, shots, seed, level, tower)
                    variants = budgets[key]
                    adam_ratio = float("nan")
                    m0_ratio = float("nan")
                    if {"sgd", "adam", "adam-m0"} <= set(variants):
                        adam_ratio = variants["adam"] / (
                            variants["sgd"] + 1e-30
                        )
                        m0_ratio = variants["adam-m0"] / (
                            variants["sgd"] + 1e-30
                        )
                    accuracy = accuracies[(dataset, shots, seed)]
                    closure = float("nan")
                    if {"sgd", "adam", "adam-m0"} <= set(accuracy):
                        denominator = accuracy["sgd"] - accuracy["adam"]
                        if abs(denominator) > 1e-12:
                            closure = (
                                100.0
                                * (accuracy["adam-m0"] - accuracy["adam"])
                                / denominator
                            )
                    primary_per_seed.append(
                        {
                            "dataset": dataset,
                            "shots": shots,
                            "seed": seed,
                            "scope": level,
                            "tower": tower,
                            "initial_r_v": initial["r_v"],
                            "initial_coordinate_mismatch_m": initial[
                                "mismatch_mean_abs"
                            ],
                            "initial_fraction_v_few_lt_v_full": initial[
                                "fraction_z_lt_0"
                            ],
                            "initial_r_u": initial["r_u_adam"],
                            "initial_cosine_u_ff_u_rr": initial[
                                "cosine_u_ff_u_rr"
                            ],
                            "oracle_denominator_cosine": initial[
                                "cosine_u_fr_u_rr"
                            ],
                            "c_m_dense_sgd": c_m,
                            "c_v_dense_sgd": c_v,
                            "collapse_ratio_dense_sgd": c_v / (c_m + 1e-30),
                            "actual_first_step_adam_to_sgd": adam_ratio,
                            "actual_first_step_adam_m0_to_sgd": m0_ratio,
                            "adam_m0_gap_closure_percent": closure,
                        }
                    )

    primary = []
    metric_fields = (
        "initial_r_v",
        "initial_coordinate_mismatch_m",
        "initial_fraction_v_few_lt_v_full",
        "initial_r_u",
        "initial_cosine_u_ff_u_rr",
        "oracle_denominator_cosine",
        "c_m_dense_sgd",
        "c_v_dense_sgd",
        "collapse_ratio_dense_sgd",
        "actual_first_step_adam_to_sgd",
        "actual_first_step_adam_m0_to_sgd",
        "adam_m0_gap_closure_percent",
    )
    for dataset in ("eurosat", "dtd"):
        for shots in (1, 16):
            for level, tower in (
                ("global", None),
                ("tower", "vision"),
                ("tower", "text"),
            ):
                selected = [
                    row
                    for row in primary_per_seed
                    if row["dataset"] == dataset
                    and row["shots"] == shots
                    and row["scope"] == level
                    and row["tower"] == tower
                ]
                if not selected:
                    continue
                primary.append(
                    {
                        "dataset": dataset,
                        "shots": shots,
                        "scope": level,
                        "tower": tower,
                        **{
                            field: mean([row[field] for row in selected])
                            for field in metric_fields
                        },
                        "seeds": len(selected),
                    }
                )
    write_csv(output / "mechanism_primary.csv", primary)
    write_csv(output / "mechanism_primary_per_seed.csv", primary_per_seed)
    write_csv(output / "mechanism_checkpoints_per_seed.csv", checkpoint_rows)
    write_csv(output / "mechanism_collapse_by_variant.csv", collapse_rows)
    write_csv(output / "mechanism_blocks.csv", block_rows)
    (output / "mechanism_primary.md").write_text(
        markdown_table(
            primary,
            [
                ("dataset", "Dataset"),
                ("shots", "K"),
                ("scope", "Scope"),
                ("tower", "Tower"),
                ("initial_r_v", "R_v(0)"),
                ("initial_coordinate_mismatch_m", "M(0)"),
                ("initial_r_u", "R_u(0)"),
                ("collapse_ratio_dense_sgd", "C_v/C_m"),
                ("actual_first_step_adam_to_sgd", "Adam/SGD budget"),
                ("actual_first_step_adam_m0_to_sgd", "M0/SGD budget"),
                ("adam_m0_gap_closure_percent", "M0 closure %"),
            ],
        )
    )
    print(json.dumps({"primary_rows": len(primary), "block_rows": len(block_rows)}))


if __name__ == "__main__":
    main()

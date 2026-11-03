#!/usr/bin/env python3
"""Verify source SHA, model revision, pairing, and collected checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analysis_common import accuracy_records, atomic_json, load_run, run_directories


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--collection_manifest", required=True)
    parser.add_argument("--output", default="tables/cross_cluster_validation.json")
    args = parser.parse_args()
    records = accuracy_records(args.input)
    commits = sorted({row["source_commit"] for row in records}, key=str)
    revisions = sorted({row["model_revision"] for row in records}, key=str)
    cells = defaultdict(set)
    clusters = defaultdict(set)
    for row in records:
        if row["variant"] != "zero-shot":
            key = (row["dataset"], row["shots"], row["seed"])
            cells[key].add(row["variant"])
            clusters[key].add(row["cluster"])
    unpaired = sorted(key for key, variants in cells.items() if variants != {"sgd", "adam", "adam-m0"})
    cross_hardware = sorted(key for key, values in clusters.items() if len(values) != 1)
    checksum_failures = []
    collection_conflicts = []
    with Path(args.collection_manifest).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source_sha256"] != row["destination_sha256"]:
                checksum_failures.append(row["destination_path"])
            if str(row.get("conflict", "")).lower() == "true":
                collection_conflicts.append(row["destination_path"])
    support = defaultdict(dict)
    validation = defaultdict(dict)
    training_randomness = defaultdict(dict)
    diagnostics = defaultdict(dict)
    trainable_manifest_digests = set()
    for run_dir in run_directories(args.input):
        config, _ = load_run(run_dir)
        if config["variant"] == "zero-shot":
            continue
        cell = (config["dataset"], int(config["shots"]), int(config["seed"]))
        variant = config["variant"]
        support[cell][variant] = (run_dir / "support_manifest.sha256").read_text().strip()
        validation[cell][variant] = (
            run_dir / "validation_manifest.sha256"
        ).read_text().strip()
        randomness = json.loads((run_dir / "training_order_hashes.json").read_text())
        training_randomness[cell][variant] = json.dumps(
            randomness, sort_keys=True, separators=(",", ":")
        )
        trainable_manifest_digests.add(digest(run_dir / "trainable_parameters.json"))
        diagnostic_path = run_dir / "diagnostic_manifest_b16.sha256"
        if diagnostic_path.is_file():
            diagnostics[cell][variant] = diagnostic_path.read_text().strip()

    def parity_failures(grouped):
        failures = []
        for cell, variants in grouped.items():
            if set(variants) != {"sgd", "adam", "adam-m0"}:
                failures.append((*cell, "missing-variant"))
            elif len(set(variants.values())) != 1:
                failures.append((*cell, "hash-mismatch"))
        return sorted(failures)

    support_failures = parity_failures(support)
    validation_failures = parity_failures(validation)
    randomness_failures = parity_failures(training_randomness)
    diagnostic_failures = parity_failures(diagnostics)
    output = {
        "run_count": len(records),
        "source_commits": commits,
        "model_revisions": revisions,
        "unpaired_cells": unpaired,
        "paired_cells_split_across_clusters": cross_hardware,
        "support_parity_failures": support_failures,
        "validation_parity_failures": validation_failures,
        "training_randomness_parity_failures": randomness_failures,
        "diagnostic_manifest_parity_failures": diagnostic_failures,
        "trainable_parameter_manifest_digests": sorted(trainable_manifest_digests),
        "collection_checksum_failures": checksum_failures,
        "collection_conflicts": collection_conflicts,
        "valid": (
            len(records) == 112
            and len(commits) == 1
            and None not in commits
            and len(revisions) == 1
            and None not in revisions
            and not unpaired
            and not cross_hardware
            and not support_failures
            and not validation_failures
            and not randomness_failures
            and not diagnostic_failures
            and len(trainable_manifest_digests) == 1
            and not checksum_failures
            and not collection_conflicts
        ),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

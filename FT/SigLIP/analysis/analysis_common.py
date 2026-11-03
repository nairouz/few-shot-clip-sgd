"""Uniquely named shared readers and writers for collected P5 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


VARIANT_ORDER = ("zero-shot", "sgd", "adam", "adam-m0")
DISPLAY = {
    "zero-shot": "Zero-shot",
    "sgd": "Dense SGD",
    "adam": "Adam",
    "adam-m0": "Adam-M0",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join(" --- " for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = "NaN" if not math.isfinite(value) else f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def run_directories(root):
    root = Path(root)
    return sorted(
        done.parent
        for done in root.rglob("DONE.json")
        if "conflicts" not in done.relative_to(root).parts
    )


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_run(run_dir):
    run_dir = Path(run_dir)
    done = json.loads((run_dir / "DONE.json").read_text())
    if done.get("status") != "complete" or not done.get("checksums"):
        raise RuntimeError(f"Invalid completion marker: {run_dir}")
    for relative, expected in done["checksums"].items():
        path = run_dir / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Completion checksum mismatch: {path}")
    config = json.loads((run_dir / "config.json").read_text())
    summary_name = "zero_shot_metrics.json" if config["variant"] == "zero-shot" else "summary.json"
    summary = json.loads((run_dir / summary_name).read_text())
    return config, summary


def accuracy_records(root):
    records = []
    for run_dir in run_directories(root):
        config, summary = load_run(run_dir)
        records.append(
            {
                "run_dir": str(run_dir),
                "cluster": run_dir.parent.name,
                "variant": config["variant"],
                "dataset": config["dataset"],
                "shots": int(config["shots"]),
                "seed": int(config["seed"]),
                "accuracy": float(
                    summary["test_accuracy"]
                    if config["variant"] == "zero-shot"
                    else summary["final_test_accuracy"]
                ),
                "source_commit": config.get("source_commit"),
                "model_revision": config.get("model_revision"),
            }
        )
    return records

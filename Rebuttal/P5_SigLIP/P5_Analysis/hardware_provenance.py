#!/usr/bin/env python3
"""Summarize per-run software, hardware, SLURM, and immutable provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_common import (
    atomic_json,
    load_run,
    markdown_table,
    run_directories,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="tables")
    args = parser.parse_args()
    rows = []
    for run_dir in run_directories(args.input):
        config, _ = load_run(run_dir)
        environment = json.loads((run_dir / "environment.json").read_text())
        packages = environment["packages"]
        slurm = environment["slurm"]
        rows.append(
            {
                "cluster": run_dir.parent.name,
                "run_id": config["run_id"],
                "variant": config["variant"],
                "dataset": config["dataset"],
                "shots": int(config["shots"]),
                "seed": int(config["seed"]),
                "hostname": environment["hostname"],
                "gpu": environment["cuda_device_name"],
                "cuda_capability": environment["cuda_capability"],
                "torch_cuda_version": environment["torch_cuda_version"],
                "python_executable": environment["executable"],
                "torch_version": packages["torch"],
                "torchvision_version": packages["torchvision"],
                "transformers_version": packages["transformers"],
                "sentencepiece_version": packages["sentencepiece"],
                "slurm_job_id": slurm["SLURM_JOB_ID"],
                "slurm_cluster": slurm["SLURM_CLUSTER_NAME"],
                "slurm_partition": slurm["SLURM_JOB_PARTITION"],
                "source_commit": config["source_commit"],
                "environment_git_commit": environment["git_commit"],
                "model_revision": config["model_revision"],
                "offline_transformers": environment["offline_environment"][
                    "TRANSFORMERS_OFFLINE"
                ],
                "offline_hub": environment["offline_environment"]["HF_HUB_OFFLINE"],
            }
        )
    output = Path(args.output_dir)
    write_csv(output / "hardware_provenance.csv", rows)
    atomic_json(output / "hardware_provenance.json", rows)
    (output / "hardware_provenance.md").write_text(
        markdown_table(
            rows,
            [
                ("cluster", "Cluster"),
                ("run_id", "Run"),
                ("gpu", "GPU"),
                ("torch_version", "Torch"),
                ("transformers_version", "Transformers"),
                ("slurm_job_id", "Job ID"),
                ("source_commit", "Source SHA"),
            ],
        )
    )
    print(json.dumps({"runs": len(rows), "clusters": sorted({row["cluster"] for row in rows})}))


if __name__ == "__main__":
    main()

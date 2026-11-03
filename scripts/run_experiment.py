#!/usr/bin/env python3
"""Run an unchanged experiment entry point in its own output directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "SGD": "FT/SGD", "Adam": "FT/Adam",
    "AdamW-0.001": "FT/AdamW_0.001", "AdamW-0.01": "FT/AdamW_0.01",
    "AdamW-0.1": "FT/AdamW_0.1", "Shampoo": "FT/Shampoo", "SOAP": "FT/SOAP",
    "ZS": "ZS",
    **{name: f"PEFT/{name}" for name in (
        "Adapter", "BitFit", "CoOp", "GaLoRE", "KgCoOp", "LayerNorm",
        "LoRA", "LP++", "PiSSA", "ProGrad", "SAFT", "SO", "TIP-Adapter"
    )},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Separate working directory for this run and learning-rate trial.")
    parser.add_argument("--dry-run", action="store_true")
    args, experiment_args = parser.parse_known_args()
    if experiment_args and experiment_args[0] == "--":
        experiment_args = experiment_args[1:]
    script = ROOT / METHODS[args.method] / "main.py"
    output = args.output_dir.expanduser().resolve()
    # Root/split/weight paths are interpreted before changing the working directory.
    for flag in ("--root_path", "--split_dir", "--weight_file"):
        if flag in experiment_args:
            idx = experiment_args.index(flag) + 1
            if idx >= len(experiment_args):
                parser.error(f"{flag} requires a path")
            experiment_args[idx] = str(Path(experiment_args[idx]).expanduser().resolve())
        for idx, arg in enumerate(experiment_args):
            if arg.startswith(flag + "="):
                experiment_args[idx] = flag + "=" + str(Path(arg.split("=", 1)[1]).expanduser().resolve())
    command = [sys.executable, str(script), *experiment_args]
    print(f"Working directory: {output}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. A distinct directory preserves run provenance.")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("DATASETS_ROOT", "CLIP_DOWNLOAD_ROOT"):
        if env.get(key):
            env[key] = str(Path(env[key]).expanduser().resolve())
    (output / "run_command.json").write_text(json.dumps({
        "method": args.method,
        "command": command,
        "working_directory": str(output),
    }, indent=2) + "\n")
    with (output / "console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=output, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())

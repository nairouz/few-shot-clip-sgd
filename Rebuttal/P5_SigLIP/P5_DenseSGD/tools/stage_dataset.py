#!/usr/bin/env python3
"""Stage exactly one P5 dataset archive into ``$SLURM_TMPDIR/datasets``."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target = destination / source.name
        if target.exists():
            raise SystemExit(f"Refusing to overwrite staged dataset: {target}")
        shutil.copytree(source, target)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            # Alliance dataset archives are trusted project inputs. Avoid the
            # Python 3.12-only ``filter=`` argument because the validated
            # SigLIP environments currently use Python 3.9/3.10.
            archive.extractall(destination)
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            archive.extractall(destination)
    else:
        raise SystemExit(f"Unsupported dataset archive: {source}")


if __name__ == "__main__":
    main()

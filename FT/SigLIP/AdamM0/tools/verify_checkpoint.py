#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Verify the pinned P5 checkpoint without contacting the network."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[2]
            / "checkpoint"
            / "checkpoint_manifest.json"
        ),
    )
    args = parser.parse_args()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_dir = Path(args.model_dir).expanduser().resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    for record in manifest["files"]:
        path = model_dir / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise SystemExit(f"FAILED: {path}")
    from transformers import AutoProcessor, AutoTokenizer, SiglipModel

    AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    SiglipModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
    print(f"verified {manifest['immutable_revision']} at {model_dir}")


if __name__ == "__main__":
    main()

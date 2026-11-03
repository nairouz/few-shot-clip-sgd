#!/usr/bin/env python3
"""Generate byte-identical audited trainable manifests for P5 variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from completion import atomic_json
from model_adapter import (
    SigLIPAdapter,
    enable_dense_transformer_blocks,
    trainable_parameter_manifest,
)
from run_utils import MODEL_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--siglip_root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    from transformers import SiglipConfig, SiglipModel

    config = SiglipConfig.from_pretrained(args.model_dir, local_files_only=True)
    with torch.device("meta"):
        model = SiglipModel(config)
    adapter = SigLIPAdapter(model, processor=None, tokenizer=None)
    enable_dense_transformer_blocks(adapter)
    manifest = trainable_parameter_manifest(adapter)
    root = Path(args.siglip_root)
    for folder in ("P5_DenseSGD", "P5_Adam", "P5_AdamM0"):
        atomic_json(root / folder / "trainable_parameters.json", manifest)
    print(
        json.dumps(
            {
                "count": manifest["trainable_parameter_count"],
                "outputs": 3,
            }
        )
    )


if __name__ == "__main__":
    main()

# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Model-neutral adapter for the pinned Hugging Face SigLIP checkpoint."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from instrumentation import ScopeMap, build_siglip_scope_map
from run_utils import MODEL_REVISION


_BLOCK = re.compile(r"^(vision_model|text_model)\.encoder\.layers\.(\d+)\.")


class ProcessorTransform:
    """A single-image DataLoader transform delegated to the pinned processor."""

    def __init__(self, image_processor) -> None:
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        output = self.image_processor(images=image, return_tensors="pt")
        return output["pixel_values"][0]


class SigLIPAdapter(nn.Module):
    def __init__(self, model: nn.Module, processor, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        revision: str = MODEL_REVISION,
        local_files_only: bool = True,
    ) -> "SigLIPAdapter":
        if revision != MODEL_REVISION:
            raise ValueError(f"P5 is pinned to {MODEL_REVISION}")
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                "P5 requires a verified local snapshot directory; missing " + str(path)
            )
        from transformers import AutoProcessor, AutoTokenizer, SiglipModel

        processor = AutoProcessor.from_pretrained(
            path, local_files_only=local_files_only, use_fast=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=local_files_only, use_fast=False
        )
        model = SiglipModel.from_pretrained(
            path, local_files_only=local_files_only, use_safetensors=True
        )
        return cls(model, processor, tokenizer)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: value
            for key, value in text_inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def classification_logits(
        self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        scale, bias = self.get_native_scale_and_bias()
        return scale * image_embeddings @ text_embeddings.transpose(-1, -2) + bias

    def get_native_scale_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is the exact Transformers 4.52.4 SiglipModel calculation.
        return self.model.logit_scale.exp(), self.model.logit_bias

    def calibration_trainability(self) -> Dict[str, bool]:
        return {
            "logit_scale": bool(self.model.logit_scale.requires_grad),
            "logit_bias": bool(self.model.logit_bias.requires_grad),
        }

    def deterministic_transform(self) -> ProcessorTransform:
        return ProcessorTransform(self.processor.image_processor)

    def training_transform(self):
        image_processor = self.processor.image_processor
        size = image_processor.size
        image_size = int(size.get("height", size.get("shortest_edge")))
        mean = tuple(float(value) for value in image_processor.image_mean)
        std = tuple(float(value) for value in image_processor.image_std)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.08, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def tokenize_prompts(self, prompts: List[str], device: str):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=int(self.tokenizer.model_max_length),
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in tokens.items()}


def freeze_all(adapter: SigLIPAdapter) -> None:
    for parameter in adapter.model.parameters():
        parameter.requires_grad = False


def enable_dense_transformer_blocks(adapter: SigLIPAdapter) -> ScopeMap:
    """Freeze everything, then unfreeze every parameter in both 12-block towers."""
    freeze_all(adapter)
    towers = {
        "vision": adapter.model.vision_model.encoder.layers,
        "text": adapter.model.text_model.encoder.layers,
    }
    for name, blocks in towers.items():
        if len(blocks) != 12:
            raise AssertionError(f"Expected 12 {name} blocks, found {len(blocks)}")
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True
    scope_map = build_siglip_scope_map(adapter.model, blocks_per_encoder=12)
    expected = {f"block/{tower}/{index}" for tower in ("vision", "text") for index in range(12)}
    if set(scope_map.groups("block")) != expected:
        raise AssertionError("Dense SigLIP block map is incomplete")
    if adapter.model.logit_scale.requires_grad or adapter.model.logit_bias.requires_grad:
        raise AssertionError("Native scale and bias must remain frozen")
    return scope_map


def trainable_parameter_manifest(adapter: SigLIPAdapter) -> Dict[str, Any]:
    entries = []
    trainable_count = 0
    for name, parameter in adapter.model.named_parameters():
        match = _BLOCK.match(name)
        tower: Optional[str] = None
        block_index: Optional[int] = None
        if match:
            tower = "vision" if match.group(1) == "vision_model" else "text"
            block_index = int(match.group(2))
        elif name.startswith("vision_model."):
            tower = "vision"
        elif name.startswith("text_model."):
            tower = "text"
        trainable = bool(parameter.requires_grad)
        if trainable:
            if match is None:
                raise AssertionError(f"Unexpected trainable parameter: {name}")
            trainable_count += parameter.numel()
        entries.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "encoder": tower,
                "block_index": block_index,
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "trainable": trainable,
            }
        )
    by_tower = {
        tower: sum(
            row["numel"] for row in entries if row["trainable"] and row["encoder"] == tower
        )
        for tower in ("vision", "text")
    }
    if min(by_tower.values()) <= 0:
        raise AssertionError("Both towers must have trainable parameters")
    return {
        "model_revision": MODEL_REVISION,
        "scope": "all parameters inside every Transformer block in both towers",
        "trainable_parameter_count": int(trainable_count),
        "trainable_parameter_count_by_tower": by_tower,
        "parameters": entries,
    }


def assert_native_logit_parity(
    adapter: SigLIPAdapter,
    pixel_values: torch.Tensor,
    text_inputs: Mapping[str, torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    adapter.eval()
    with torch.no_grad():
        native = adapter.model(pixel_values=pixel_values, **text_inputs).logits_per_image
        images = adapter.encode_image(pixel_values)
        texts = adapter.encode_text(text_inputs)
        adapted = adapter.classification_logits(images, texts)
    torch.testing.assert_close(adapted, native, atol=atol, rtol=rtol)
    scale, _ = adapter.get_native_scale_and_bias()
    if math.isclose(float(scale.detach()), 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("A fixed CLIP scale of 100 must never be used")


__all__ = [
    "SigLIPAdapter",
    "ProcessorTransform",
    "freeze_all",
    "enable_dense_transformer_blocks",
    "trainable_parameter_manifest",
    "assert_native_logit_parity",
]

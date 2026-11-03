#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Utilities for mapping CLIP parameters to encoder/block scopes.

This helper is shared by the second-moment diagnostic code. It mirrors the
parameter-name parsing used in the ECCV sign-correctness analysis so that the
vision and text encoders can be analysed separately and at the transformer-block
level.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import torch


BlockKey = Tuple[str, int]  # (encoder, block_idx)

_VISION_RE = re.compile(r"^visual\.transformer\.resblocks\.(\d+)\.")
_TEXT_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.")


def infer_encoder_and_block(param_name: str) -> BlockKey:
    """Infer (encoder, block_idx) from a CLIP parameter name.

    Returns:
        ("vision", i) for visual transformer block parameters,
        ("text", i) for text transformer block parameters,
        ("other", -1) otherwise.
    """
    m = _VISION_RE.match(param_name)
    if m:
        return "vision", int(m.group(1))
    m = _TEXT_RE.match(param_name)
    if m:
        return "text", int(m.group(1))
    return "other", -1



def build_param_to_block_map(
    model: torch.nn.Module,
    *,
    only_trainable: bool = True,
) -> Dict[torch.nn.Parameter, BlockKey]:
    """Map CLIP parameters to (encoder, block_idx).

    Args:
        model: CLIP model.
        only_trainable: If True, only include parameters with
            ``requires_grad=True``.
    """
    mapping: Dict[torch.nn.Parameter, BlockKey] = {}
    for name, param in model.named_parameters():
        if only_trainable and not getattr(param, "requires_grad", False):
            continue
        enc, blk = infer_encoder_and_block(name)
        if blk >= 0:
            mapping[param] = (enc, blk)
    return mapping

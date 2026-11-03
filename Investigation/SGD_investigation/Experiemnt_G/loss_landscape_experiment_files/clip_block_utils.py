from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, MutableMapping, Tuple

import torch

BlockKey = Tuple[str, int]

_VISION_RE = re.compile(r"^visual\.transformer\.resblocks\.(\d+)\.")
_TEXT_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.")


def infer_encoder_and_block(param_name: str) -> BlockKey:
    m = _VISION_RE.match(param_name)
    if m:
        return "vision", int(m.group(1))
    m = _TEXT_RE.match(param_name)
    if m:
        return "text", int(m.group(1))
    return "other", -1


def build_trainable_meta(model: torch.nn.Module) -> List[Dict[str, object]]:
    meta: List[Dict[str, object]] = []
    for name, p in model.named_parameters():
        if not getattr(p, "requires_grad", False):
            continue
        enc, blk = infer_encoder_and_block(name)
        meta.append({
            "name": name,
            "encoder": enc,
            "block": blk,
            "numel": int(p.numel()),
            "shape": tuple(p.shape),
        })
    return meta


def group_names_by_encoder(meta: Iterable[Dict[str, object]]) -> Dict[str, List[str]]:
    out = defaultdict(list)
    for m in meta:
        out[str(m["encoder"])].append(str(m["name"]))
    return dict(out)


def group_names_by_block(meta: Iterable[Dict[str, object]]) -> Dict[Tuple[str, int], List[str]]:
    out = defaultdict(list)
    for m in meta:
        out[(str(m["encoder"]), int(m["block"]))].append(str(m["name"]))
    return dict(out)


def subset_state(state: MutableMapping[str, torch.Tensor], names: Iterable[str]) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((n, state[n]) for n in names)

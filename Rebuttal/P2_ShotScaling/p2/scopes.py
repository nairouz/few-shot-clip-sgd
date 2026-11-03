from __future__ import annotations
import re
VISION = re.compile(r"^visual\.transformer\.resblocks\.(\d+)\.")
TEXT = re.compile(r"^transformer\.resblocks\.(\d+)\.")

def build_scope_map(model):
    mapping = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = VISION.match(name) or TEXT.match(name)
        if match is None:
            raise ValueError(f"unmapped trainable parameter: {name}")
        encoder = "vision" if VISION.match(name) else "text"
        block = int(match.group(1))
        mapping[parameter] = {"global": "global", "encoder": encoder, "block": f"{encoder}_{block}"}
    return mapping

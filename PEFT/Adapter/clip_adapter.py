# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import torch
import torch.nn as nn
import clip

class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.block = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class CLIPAdapter(nn.Module):
    def __init__(self, clip_model: "clip.CLIP", *, alpha: float = 0.2, reduction: int = 4, learn_alpha: bool = False, logit_scale: float = 1.0):
        super().__init__()
        self.clip = clip_model
        self.logit_scale = logit_scale
        dim = clip_model.visual.output_dim        
        base_dtype = getattr(clip_model, "dtype", next(clip_model.parameters()).dtype)
        self.adapter = BottleneckAdapter(dim, reduction).to(base_dtype)
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32), persistent=False)
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def encode_image(self, images):
        with torch.no_grad():
            feat = self.clip.encode_image(images)
        feat = feat.to(self.adapter.block[0].weight.dtype)
        delta = self.adapter(feat)
        return (1. - self.alpha) * feat + self.alpha * delta

    def encode_text(self, tokens):
        with torch.no_grad():
            return self.clip.encode_text(tokens)

    def forward(self, images, text_features):
        img = self.encode_image(images)
        img = img / img.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale * img @ text_features.t()
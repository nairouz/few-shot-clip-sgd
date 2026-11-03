# prograd_utils.py
import torch
from typing import Sequence

def project_and_step(model, g_ce, g_kl, lam=1.0, eps=1e-6):
    with torch.no_grad():
        for p, g1, g2 in zip(model.parameters(), g_ce, g_kl):
            if p.requires_grad is False or g1 is None:
                continue
            if g2 is None:
                p.grad = g1
                continue
            denom = torch.dot(g2.flatten(), g2.flatten()) + eps
            proj  = torch.dot(g1.flatten(), g2.flatten()) / denom
            p.grad = g1 - lam * proj * g2          # ← projected gradient
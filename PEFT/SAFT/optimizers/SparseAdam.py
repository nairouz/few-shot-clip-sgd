# SPDX-License-Identifier: Apache-2.0
# Author : 2025 — feel free to reuse / adapt.

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Iterable, Optional, Dict, Any, Tuple


class SparseAdam(Optimizer):
    r"""
    **Sparse Adam** - Adam with optional top-k / random masking.

    *If ``static_mask=True`` the first/second-moment buffers are stored only
    for the *k* selected weights, saving   k·dtype   instead of |p|·dtype.*

    Update
    ------
        Dense γ=0 : ``p ← p - lr · m̂``  
        Dense γ=1 : ``p ← p - lr · m̂ / (√v̂ + eps)``  
        Sparse    : same formula, but applied only on the mask *S* of size
        ``k = ceil(density_ratio · |p|)``.

    All memory-reduction tricks from your last SparseSignAdam are preserved
    (no scratch buffer, gradient re-used in dense γ=1).

    Example
    -------
    >>> opt = SparseAdam([
    ...     {"params": model.head.parameters(), "dense": True},        # classic Adam
    ...     {"params": model.backbone.parameters(),
    ...      "density_ratio": 5e-4, "static_mask": False,              # dynamic sparse
    ...      "sparse_mode": "topk", "use_second_moment": True}
    ... ], lr=1e-3, betas=(0.9, 0.999))
    """

    # ------------------------------------------------------------ init
    def __init__(
        self,
        params: Iterable,
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_second_moment: bool = True,
        density_ratio: float = 1e-3,
        static_mask: bool = True,
        sparse_mode: str = "topk",
        mask_refresh: int = 1,
        dense: bool = False,
    ):
        if mask_refresh < 1:
            raise ValueError("mask_refresh must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0 < density_ratio <= 1):
            raise ValueError("density_ratio must be in (0,1]")
        if weight_decay < 0:
            raise ValueError("weight_decay must be ≥ 0")
        if sparse_mode not in {"topk", "random"}:
            raise ValueError("sparse_mode must be 'topk' or 'random'")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                use_second_moment=use_second_moment,
                density_ratio=density_ratio,
                static_mask=static_mask,
                sparse_mode=sparse_mode,
                mask_refresh=mask_refresh,
                dense=dense,
            )
        )

    # -------------------------------------------------------- helpers
    @staticmethod
    def _select_mask(v: Tensor, k: int, mode: str) -> Tensor:
        if mode == "topk":
            buf = v.clone().abs_()
            _, idx = torch.topk(buf, k)
            del buf
            return idx
        return torch.randperm(v.numel(), device=v.device)[:k]

    # -------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for g in self.param_groups:
            lr, wd        = g["lr"], g["weight_decay"]
            beta1, beta2  = g["betas"]
            eps, use_v    = g["eps"], g["use_second_moment"]
            dr, static    = g["density_ratio"], g["static_mask"]
            mode, dense   = g["sparse_mode"], g["dense"]
            refresh       = g["mask_refresh"] 

            for p in g["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SparseAdam expects dense gradients")
                
                if wd:
                    grad = grad.add(p, alpha=wd) 
                else:
                    grad = p.grad
                st: Dict[str, Any] = self.state[p]

                # ---------------- state init -----------------------
                if not st:
                    st["step"] = 0
                    dev, dt = p.device, p.dtype
                    full    = p.numel()

                    if dense or not static:
                        st["exp_avg"] = torch.zeros(full, dtype=dt, device=dev)
                        if use_v:
                            st["exp_avg_sq"] = torch.zeros(full, dtype=dt, device=dev)
                    else:
                        k = max(1, int(full * dr))
                        st["mask"] = self._select_mask(grad.view(-1), k, mode)
                        st["exp_avg"] = torch.zeros(k, dtype=dt, device=dev)
                        if use_v:
                            st["exp_avg_sq"] = torch.zeros(k, dtype=dt, device=dev)

                # ---------------- scalars -------------------------
                st["step"] += 1
                t   = st["step"]
                bc1 = 1 - beta1 ** t
                if use_v:
                    bc2 = 1 - beta2 ** t

                # ==================================================
                # DENSE BRANCH
                # ==================================================
                if dense:
                    m = st["exp_avg"].view_as(p)
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    if use_v:                   
                        v = st["exp_avg_sq"].view_as(p)
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        grad.copy_(v).div_(bc2).sqrt_().add_(eps).reciprocal_()
                        grad.mul_(m)                    
                        p.add_(grad, alpha=-lr / bc1)
                    else:                               
                        p.add_(m, alpha=-lr / bc1)
                    p.grad = None
                    continue

                # ==================================================
                # STATIC SPARSE BRANCH
                # ==================================================
                if static:
                    idx = st["mask"]
                    m   = st["exp_avg"]
                    g_slice = grad.view(-1).index_select(0, idx)  # size k
                    m.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                    if use_v:
                        v = st["exp_avg_sq"]
                        v.mul_(beta2).add_(g_slice * g_slice, alpha=1 - beta2)
                        g_slice.copy_(v).div_(bc2).sqrt_().add_(eps).reciprocal_()
                        g_slice.mul_(m)           
                    else:
                        g_slice.copy_(m)                       
                    p.data.view(-1).index_add_(0, idx, g_slice, alpha=-lr / bc1)
                    p.grad = None
                    continue

                # ==================================================
                # DYNAMIC SPARSE BRANCH
                # ==================================================
                k   = max(1, int(p.numel() * dr))
                need_new = ("mask" not in st) or (t % refresh == 0)
                if need_new:
                    st["mask"] = self._select_mask(grad.view(-1), k, mode)
                idx = st["mask"]    
                m = st["exp_avg"].view(-1)
                m.mul_(beta1)
                g_slice = grad.view(-1).index_select(0, idx)       
                m.index_add_(0, idx, g_slice, alpha=1 - beta1)
                if use_v:
                    v = st["exp_avg_sq"].view(-1)
                    v.mul_(beta2)
                    v.index_add_(0, idx, g_slice * g_slice, alpha=1 - beta2)
                    g_slice.copy_(v.index_select(0, idx)).div_(bc2).sqrt_().add_(eps).reciprocal_()
                    g_slice.mul_(m.index_select(0, idx))           
                else:
                    g_slice.copy_(m.index_select(0, idx))          
                p.data.view(-1).index_add_(0, idx, g_slice, alpha=-lr / bc1)
                p.grad = None

        return None

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Iterable, Optional, Dict, Any


class SparseSGD(Optimizer):
    r"""
    SGD where each step touches **all** weights (dense) *or* only a sparse
    subset *S* (sparse), selected either by |gradient| **top-k** importance
    or by uniform **random** sampling.

        Dense  :  p ← p - lr · (g + wd·p)
        Sparse :  p_S ← p_S - lr · (g_S + wd·p_S)

    `|S| = ceil(density_ratio · |p|)` with `density_ratio ∈ (0,1]`.

    Parameters
    ----------
    params        : iterable of Tensors or param-group dicts
    lr            : float               - learning rate (default 1e-3)
    density_ratio : float in (0,1]      - fraction of weights updated when sparse mode is active (default 1e-3)
    weight_decay  : float ≥ 0           - L2 weight decay (default 0.0)
    static_mask   : bool                - cache mask once per parameter (default True)
    sparse_mode   : "topk" | "random"   - mask selection strategy (default "topk")
    dense         : bool                - force dense updates for that param-group
    
    Example
    -------
    >>> opt = SparseSGD([
    ...     {"params": model.head.parameters(), "dense": True},          # dense
    ...     {"params": model.backbone.parameters(),
    ...      "density_ratio": 5e-4, "static_mask": False}               # sparse
    ... ])
    """

    def __init__(
            self, 
            params: Iterable, 
            *, 
            lr: float = 1e-3, 
            density_ratio: float = 1e-3, 
            weight_decay: float = 0.0, 
            static_mask: bool = True, 
            sparse_mode: str = "topk", 
            mask_refresh: int = 1,
            dense: bool = False
        ):
        if mask_refresh < 1:
            raise ValueError("mask_refresh must be a positive integer")
        if not (lr > 0):
            raise ValueError("lr must be positive")
        if not (0.0 < density_ratio <= 1.0):
            raise ValueError("density_ratio must be in (0, 1]")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be ≥ 0")
        if sparse_mode not in {"topk", "random"}:
            raise ValueError("sparse_mode must be 'topk' or 'random'")

        defaults = dict(
            lr=lr, 
            density_ratio=density_ratio, 
            weight_decay=weight_decay, 
            static_mask=static_mask, 
            sparse_mode=sparse_mode,
            mask_refresh=mask_refresh,   
            dense=dense
        )
        super().__init__(params, defaults)

    @staticmethod
    @torch.no_grad()
    def _dense_step(p: Tensor, g: Tensor, lr: float, wd: float):
        if wd:
            g.add_(p, alpha=wd)            
        p.add_(g, alpha=-lr)

    @staticmethod
    def _select_mask(g_flat: Tensor, k: int, mode: str):
        if mode == "topk":
            buf = g_flat.clone().abs_()    
            _, idx = torch.topk(buf, k)
            del buf
            return idx
        return torch.randperm(g_flat.numel(), device=g_flat.device)[:k]

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            dr, static = group["density_ratio"], group["static_mask"]
            mode, dense_flag = group["sparse_mode"], group["dense"]
            refresh = group["mask_refresh"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SparseSGD expects dense gradients")
                g = p.grad              

                # ==================================================
                # DENSE BRANCH
                # ==================================================
                if dense_flag:
                    self._dense_step(p.data, g, lr, wd)
                    p.grad = None
                    continue

                # ==================================================
                # SPARSE BRANCH
                # ==================================================
                k = max(1, int(p.numel() * dr))
                state: Dict[str, Any] = self.state[p]
                step_cnt = state.get("step_cnt", 0) + 1
                state["step_cnt"] = step_cnt
                need_new_mask = (("mask" not in state) or (not static and step_cnt % refresh == 0))
                if need_new_mask:
                    state["mask"] = self._select_mask(g.view(-1), k, mode)
                mask = state["mask"]       
                g_vals = g.view(-1).index_select(0, mask)  
                if wd:
                    g_vals.add_(p.data.view(-1).index_select(0, mask), alpha=wd)
                p.data.view(-1).index_add_(0, mask, g_vals, alpha=-lr)
                p.grad = None      
                del g_vals           

        return loss

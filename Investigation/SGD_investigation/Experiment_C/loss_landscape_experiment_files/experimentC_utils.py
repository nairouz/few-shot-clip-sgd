#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import os
from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import importlib
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _import_clip_with_optional_torchvision_nms_stub():
    """Import the local CLIP package robustly across torchvision builds.

    Some environments ship torchvision without a registered `torchvision::nms`
    operator. Importing torchvision then fails inside `_meta_registrations`.
    We first try the normal import. Only if the specific missing-NMS error is
    raised do we clear the partial torchvision import state, define a dummy
    operator schema, and retry.
    """
    try:
        import clip as _clip  # local package from the repository root
        return _clip
    except RuntimeError as exc:
        msg = str(exc)
        if "operator torchvision::nms does not exist" not in msg:
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        return importlib.import_module("clip")


clip = _import_clip_with_optional_torchvision_nms_stub()
from datasets import build_dataset
from datasets.utils import build_data_loader
from run_utils import set_random_seed
from train import mark_as_trainable
from utils import resolve_root


StateDict = OrderedDict[str, torch.Tensor]

_HVP_FALLBACK_WARNED = False

DEFAULT_CHECKPOINT_EPOCHS = [0, 1, 2, 5, 10, 20, 30]


def cuda_memory_cleanup(device: Optional[str] = None) -> None:
    """Release Python refs and clear the CUDA caching allocator.

    Experiment C repeatedly alternates between gradient computations and
    no-grad surface evaluation. On large CLIP ViT-B/16 runs this can fragment
    the CUDA allocator, especially for the full-gradient probe. Calling this
    helper after heavy mini-batches makes long resume runs much more reliable
    on 48GB cards such as the RTX A6000.
    """
    gc.collect()
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _sdpa_double_backward_context():
    """Force PyTorch to use math SDPA when available.

    Double backward through flash / memory-efficient attention is not available
    in some PyTorch builds. Experiment C only needs the Hessian probe around a
    checkpoint, so using the math kernel is acceptable. If the build does not
    expose the SDPA kernel context manager, we simply return a nullcontext and
    let the finite-difference fallback handle unsupported cases.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return nullcontext()


def _looks_like_double_backward_attention_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    needles = [
        'scaled_dot_product',
        '_scaled_dot_product_efficient_attention_backward',
        'double backward',
        'derivative for aten::_scaled_dot_product',
        'not implemented',
    ]
    return any(n in msg for n in needles)


def load_clip_model(backbone: str, device: str):
    import getpass

    cache = os.getenv("CLIP_DOWNLOAD_ROOT")
    model, preprocess = clip.load(backbone, device=device, download_root=cache)
    return model, preprocess


def build_loader_from_source(data_source, *, batch_size: int, preprocess, is_train: bool, shuffle: bool, num_workers: int):
    if hasattr(data_source, "samples") and hasattr(data_source, "__getitem__"):
        return DataLoader(
            data_source,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return build_data_loader(
        data_source=data_source,
        batch_size=batch_size,
        tfm=preprocess,
        is_train=is_train,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def maybe_full_support_bs(n_items: int, full_support_train: bool, batch_size: int) -> int:
    if full_support_train:
        return max(1, n_items)
    return batch_size


def tokenize_class_texts(classnames: Sequence[str], template: Sequence[str], device: str) -> torch.Tensor:
    prompt_template = template[0]
    texts = [prompt_template.format(name.replace("_", " ")) for name in classnames]
    return clip.tokenize(texts).to(device)


def prepare_common_experiment(args):
    args.root_path = resolve_root(args.root_path)
    set_random_seed(args.seed)
    clip_model, preprocess = load_clip_model(args.backbone, args.device)
    logit_scale = 100.0

    few_dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)

    clip_model = clip_model.to(args.device)
    clip_model.eval()
    mark_as_trainable(clip_model, args)

    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    few_train_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=maybe_full_support_bs(len(few_dataset.train_x), args.full_support_train, args.batch_size),
        preprocess=preprocess,
        is_train=True,
        shuffle=not args.full_support_train,
        num_workers=args.num_workers,
    )
    # Full-support few-shot loaders for geometry / gradient / Hessian.
    few_eval_loader = build_loader_from_source(
        few_dataset.train_x,
        batch_size=max(1, len(few_dataset.train_x)),
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    full_eval_loader = build_loader_from_source(
        full_dataset.train_x,
        batch_size=args.eval_batch_size,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # Memory-safe loader for gradient-bearing full-train computations.
    # Loss-only surface evaluation can use a larger eval batch, but computing
    # full gradients with all CLIP blocks trainable can OOM at batch 256.
    grad_eval_batch_size = int(getattr(args, "grad_eval_batch_size", args.eval_batch_size))
    full_grad_loader = build_loader_from_source(
        full_dataset.train_x,
        batch_size=max(1, grad_eval_batch_size),
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader_from_source(
        full_dataset.test,
        batch_size=args.eval_batch_size,
        preprocess=preprocess,
        is_train=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_tokens = tokenize_class_texts(full_dataset.classnames, full_dataset.template, args.device)

    return {
        "clip_model": clip_model,
        "preprocess": preprocess,
        "logit_scale": logit_scale,
        "few_dataset": few_dataset,
        "full_dataset": full_dataset,
        "few_train_loader": few_train_loader,
        "few_eval_loader": few_eval_loader,
        "full_eval_loader": full_eval_loader,
        "full_grad_loader": full_grad_loader,
        "test_loader": test_loader,
        "class_tokens": class_tokens,
    }


def get_trainable_state(model: torch.nn.Module) -> StateDict:
    state: StateDict = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().float().clone()
    return state


def zero_like_state(ref: MutableMapping[str, torch.Tensor]) -> StateDict:
    out: StateDict = OrderedDict()
    for k, v in ref.items():
        out[k] = torch.zeros_like(v, dtype=torch.float32, device="cpu")
    return out


def copy_state(ref: MutableMapping[str, torch.Tensor]) -> StateDict:
    out: StateDict = OrderedDict()
    for k, v in ref.items():
        out[k] = v.clone()
    return out


def set_trainable_state(model: torch.nn.Module, state: MutableMapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def set_trainable_state_linear_combo(
    model: torch.nn.Module,
    base_state: MutableMapping[str, torch.Tensor],
    d_state: MutableMapping[str, torch.Tensor],
    e_state: MutableMapping[str, torch.Tensor],
    alpha: float,
    beta: float,
    scale: float,
) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                base = base_state[name].to(device=p.device, dtype=p.dtype)
                d = d_state[name].to(device=p.device, dtype=p.dtype)
                e = e_state[name].to(device=p.device, dtype=p.dtype)
                p.copy_(base + scale * (alpha * d + beta * e))


def save_trainable_checkpoint(path: Path, state: MutableMapping[str, torch.Tensor], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": dict(state), "meta": meta}, path)


def load_trainable_checkpoint(path: Path) -> Tuple[StateDict, Dict[str, object]]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = OrderedDict((k, v.float()) for k, v in obj["trainable_state"].items())
    meta = dict(obj.get("meta", {}))
    return state, meta


def state_dot(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> float:
    s = 0.0
    for k in a.keys():
        s += float((a[k].double() * b[k].double()).sum().item())
    return s


def state_sqnorm(a: MutableMapping[str, torch.Tensor]) -> float:
    return state_dot(a, a)


def state_norm(a: MutableMapping[str, torch.Tensor]) -> float:
    return math.sqrt(max(state_sqnorm(a), 0.0))


def state_scale(a: MutableMapping[str, torch.Tensor], scalar: float) -> StateDict:
    out: StateDict = OrderedDict()
    for k, v in a.items():
        out[k] = v * float(scalar)
    return out


def state_add(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor], alpha: float = 1.0) -> StateDict:
    out: StateDict = OrderedDict()
    for k in a.keys():
        out[k] = a[k] + float(alpha) * b[k]
    return out


def state_sub(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor]) -> StateDict:
    out: StateDict = OrderedDict()
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out


def normalize_state(a: MutableMapping[str, torch.Tensor], eps: float = 1e-12) -> StateDict:
    n = state_norm(a)
    if n <= eps:
        return copy_state(a)
    return state_scale(a, 1.0 / n)


def orthogonalize_state(e: MutableMapping[str, torch.Tensor], d: MutableMapping[str, torch.Tensor], eps: float = 1e-12) -> StateDict:
    denom = state_sqnorm(d)
    if denom <= eps:
        return copy_state(e)
    proj = state_dot(e, d) / denom
    out: StateDict = OrderedDict()
    for k in e.keys():
        out[k] = e[k] - proj * d[k]
    return out


def cosine_state(a: MutableMapping[str, torch.Tensor], b: MutableMapping[str, torch.Tensor], eps: float = 1e-12) -> float:
    denom = state_norm(a) * state_norm(b)
    if denom <= eps:
        return float("nan")
    return state_dot(a, b) / denom


def random_state_like(ref: MutableMapping[str, torch.Tensor], seed: int | None = None) -> StateDict:
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
    out: StateDict = OrderedDict()
    for k, v in ref.items():
        out[k] = torch.randn(v.shape, generator=gen, dtype=torch.float32)
    return out


def state_mul_invsqrt(v_state: MutableMapping[str, torch.Tensor], m_state: MutableMapping[str, torch.Tensor], eps: float = 1e-8) -> StateDict:
    out: StateDict = OrderedDict()
    for k in m_state.keys():
        out[k] = m_state[k] / torch.sqrt(v_state[k] + eps)
    return out


def compute_loader_loss(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target, reduction="mean")
            total_loss += float(loss.item()) * int(target.size(0))
            total_items += int(target.size(0))
            del images, target, image_features, logits, loss
    del text_features
    cuda_memory_cleanup(device)
    model.train(was_training)
    if total_items == 0:
        return float("nan")
    return total_loss / total_items


def compute_loader_loss_acc(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target, reduction="mean")
            total_loss += float(loss.item()) * int(target.size(0))
            total_correct += int((logits.argmax(dim=1) == target).sum().item())
            total_items += int(target.size(0))
            del images, target, image_features, logits, loss
    del text_features
    cuda_memory_cleanup(device)
    model.train(was_training)
    if total_items == 0:
        return float("nan"), float("nan")
    return total_loss / total_items, 100.0 * total_correct / total_items


def _trainable_named_params(model: torch.nn.Module):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def compute_dataset_gradient_state(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> Tuple[StateDict, float]:
    was_training = model.training
    model.eval()
    named_params = _trainable_named_params(model)
    ref = OrderedDict((name, p.detach().cpu().float().clone()) for name, p in named_params)
    grad_state = zero_like_state(ref)
    total_items = len(loader.dataset)
    total_loss = 0.0

    for images, target in loader:
        batch_n = int(target.size(0))
        weight = float(batch_n) / float(total_items)
        model.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features.t())
        loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()
        total_loss += float(loss.item()) * batch_n
        for name, p in named_params:
            if p.grad is not None:
                grad_state[name].add_(p.grad.detach().cpu().float(), alpha=weight)
        # Free gradient tensors immediately. Otherwise a long full-gradient pass
        # keeps the previous batch's gradient storage alive until the next batch.
        model.zero_grad(set_to_none=True)
        del images, target, text_features, image_features, logits, loss
        cuda_memory_cleanup(device)

    model.zero_grad(set_to_none=True)
    cuda_memory_cleanup(device)
    model.train(was_training)
    return grad_state, total_loss / max(total_items, 1)


def compute_dataset_second_moment_state(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> StateDict:
    was_training = model.training
    model.eval()
    named_params = _trainable_named_params(model)
    ref = OrderedDict((name, p.detach().cpu().float().clone()) for name, p in named_params)
    v_state = zero_like_state(ref)
    total_items = len(loader.dataset)

    for images, target in loader:
        batch_n = int(target.size(0))
        weight = float(batch_n) / float(total_items)
        model.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_features = model.encode_text(class_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * (image_features @ text_features.t())
        loss = F.cross_entropy(logits, target, reduction="mean")
        loss.backward()
        for name, p in named_params:
            if p.grad is not None:
                g = p.grad.detach().cpu().float()
                v_state[name].add_(g * g, alpha=weight)
        model.zero_grad(set_to_none=True)
        del images, target, text_features, image_features, logits, loss
        cuda_memory_cleanup(device)

    model.zero_grad(set_to_none=True)
    cuda_memory_cleanup(device)
    model.train(was_training)
    return v_state


def state_to_device_list(state: MutableMapping[str, torch.Tensor], params: Sequence[torch.nn.Parameter], device: str) -> List[torch.Tensor]:
    return [state[name].to(device=device, dtype=p.dtype) for name, p in ((n, p) for n, p in _trainable_named_params_from_params(params))]


def _trainable_named_params_from_params(params: Sequence[torch.nn.Parameter]):
    # Placeholder helper for typing symmetry; not used directly.
    raise NotImplementedError


def compute_fewshot_loss_single_batch(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
):
    images, target = next(iter(loader))
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    text_features = model.encode_text(class_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = logit_scale * (image_features @ text_features.t())
    loss = F.cross_entropy(logits, target, reduction="mean")
    return loss


def compute_single_batch_gradient_state(
    model: torch.nn.Module,
    loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
) -> StateDict:
    """Gradient of the first batch of ``loader`` at the model's current state."""
    named_params = _trainable_named_params(model)
    ref = OrderedDict((name, p.detach().cpu().float().clone()) for name, p in named_params)
    grad_state = zero_like_state(ref)
    model.zero_grad(set_to_none=True)
    loss = compute_fewshot_loss_single_batch(model, loader, class_tokens, logit_scale, device)
    loss.backward()
    for name, p in named_params:
        if p.grad is not None:
            grad_state[name].copy_(p.grad.detach().cpu().float())
    model.zero_grad(set_to_none=True)
    cuda_memory_cleanup(device)
    return grad_state


def _state_add_scaled(base: MutableMapping[str, torch.Tensor], vec: MutableMapping[str, torch.Tensor], alpha: float) -> StateDict:
    out: StateDict = OrderedDict()
    for k in base.keys():
        out[k] = base[k] + float(alpha) * vec[k]
    return out


def compute_hvp_state_finite_difference(
    model: torch.nn.Module,
    few_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    vec_state: MutableMapping[str, torch.Tensor],
    fd_eps: float = 1e-3,
) -> StateDict:
    """Finite-difference Hessian-vector product.

    This avoids double backward through attention kernels that do not implement
    second derivatives. ``vec_state`` is normalized internally to make the step
    size stable across checkpoints.
    """
    base_state = get_trainable_state(model)
    direction = normalize_state(vec_state)
    plus_state = _state_add_scaled(base_state, direction, fd_eps)
    minus_state = _state_add_scaled(base_state, direction, -fd_eps)

    set_trainable_state(model, plus_state)
    g_plus = compute_single_batch_gradient_state(model, few_loader, class_tokens, logit_scale, device)

    set_trainable_state(model, minus_state)
    g_minus = compute_single_batch_gradient_state(model, few_loader, class_tokens, logit_scale, device)

    set_trainable_state(model, base_state)

    out: StateDict = OrderedDict()
    scale = 1.0 / (2.0 * float(fd_eps))
    for k in base_state.keys():
        out[k] = (g_plus[k] - g_minus[k]) * scale
    return out


def compute_hvp_state_autograd(
    model: torch.nn.Module,
    few_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    vec_state: MutableMapping[str, torch.Tensor],
) -> StateDict:
    named_params = _trainable_named_params(model)
    params = [p for _, p in named_params]
    model.zero_grad(set_to_none=True)
    with _sdpa_double_backward_context():
        loss = compute_fewshot_loss_single_batch(model, few_loader, class_tokens, logit_scale, device)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        vec_list = [vec_state[name].to(device=device, dtype=p.dtype) for name, p in named_params]
        hvp = torch.autograd.grad(grads, params, grad_outputs=vec_list, retain_graph=False, create_graph=False)
    out: StateDict = OrderedDict()
    for (name, _), hv in zip(named_params, hvp):
        out[name] = hv.detach().cpu().float()
    model.zero_grad(set_to_none=True)
    return out


def compute_hvp_state(
    model: torch.nn.Module,
    few_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    vec_state: MutableMapping[str, torch.Tensor],
    *,
    mode: str = 'auto',
    fd_eps: float = 1e-3,
) -> Tuple[StateDict, str]:
    """Compute Hessian-vector product with an automatic finite-difference fallback."""
    if mode not in {'auto', 'autograd', 'finite_difference'}:
        raise ValueError(f'Unsupported HVP mode: {mode}')

    if mode in {'auto', 'autograd'}:
        try:
            hv = compute_hvp_state_autograd(model, few_loader, class_tokens, logit_scale, device, vec_state)
            return hv, 'autograd'
        except RuntimeError as exc:
            if mode == 'autograd' or not _looks_like_double_backward_attention_error(exc):
                raise
            global _HVP_FALLBACK_WARNED
            if not _HVP_FALLBACK_WARNED:
                print('[Experiment C] Autograd Hessian-vector product is unavailable for the current attention backend; falling back to finite differences.')
                _HVP_FALLBACK_WARNED = True

    hv = compute_hvp_state_finite_difference(
        model=model,
        few_loader=few_loader,
        class_tokens=class_tokens,
        logit_scale=logit_scale,
        device=device,
        vec_state=vec_state,
        fd_eps=fd_eps,
    )
    return hv, 'finite_difference'


def power_iteration_hessian(
    model: torch.nn.Module,
    few_loader,
    class_tokens: torch.Tensor,
    logit_scale: float,
    device: str,
    ref_state: MutableMapping[str, torch.Tensor],
    num_iters: int = 8,
    seed: int = 1,
    hvp_mode: str = 'auto',
    fd_eps: float = 1e-3,
) -> Tuple[StateDict, float, Dict[str, object]]:
    v = normalize_state(random_state_like(ref_state, seed=seed))
    eigval = 0.0
    used_mode = None
    current_mode = hvp_mode
    for _ in range(num_iters):
        hv, used_mode = compute_hvp_state(
            model, few_loader, class_tokens, logit_scale, device, v, mode=current_mode, fd_eps=fd_eps
        )
        if hvp_mode == 'auto' and used_mode == 'finite_difference':
            current_mode = 'finite_difference'
        hv_norm = state_norm(hv)
        if hv_norm <= 1e-12:
            break
        eigval = state_dot(v, hv)
        v = normalize_state(hv)
    hv, used_mode = compute_hvp_state(
        model, few_loader, class_tokens, logit_scale, device, v, mode=current_mode, fd_eps=fd_eps
    )
    eigval = state_dot(v, hv)
    info = {'hvp_mode_used': used_mode, 'hvp_fd_eps': float(fd_eps)}
    return v, float(eigval), info



def make_default_run_name(prefix: str, dataset: str, shots: int, backbone: str, seed: int, position: str, encoder: str, full_support_train: bool) -> str:
    bb = backbone.replace("/", "-")
    return (
        f"{prefix}_dataset_{dataset}_shots_{shots}_backbone_{bb}_opt_sgd_"
        f"seed_{seed}_pos_{position}_enc_{encoder}_fullsupport_{str(full_support_train).lower()}"
    )

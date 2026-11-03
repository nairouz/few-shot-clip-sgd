# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import csv, os, time, clip
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import autocast, GradScaler
from torch.optim import Adam, SGD
from utils import cls_acc

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_text_features(dataset, clip_model, device):
    template = dataset.template[0]
    texts = [template.format(c.replace('_', ' ')) for c in dataset.classnames]
    tokens = clip.tokenize(texts).to(device)
    txt_feat = clip_model.encode_text(tokens)
    return txt_feat / txt_feat.norm(dim=-1, keepdim=True)

@torch.no_grad()
def evaluate_zero_shot(clip_model, loader, dataset, device, logit_scale):
    text_features = build_text_features(dataset, clip_model, device)
    acc, tot = 0.0, 0
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits   = logit_scale * img_feat @ text_features.t()
        acc += cls_acc(logits, target) * target.size(0)
        tot += target.size(0)
    return acc / tot

@torch.no_grad()
def evaluate_tipf(tip_f, clip_model, loader, cache_vals, text_features, device, logit_scale):
    correct, n = 0, 0
    tip_f.eval()
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        clip_logits = logit_scale * img_feat @ text_features.t()
        logits = tip_f.tip_logits(img_feat, cache_vals, clip_logits)
        correct += (logits.argmax(1) == target).sum().item()
        n += target.size(0)
    return 100. * correct / n

def _is_adapter(model):
    return hasattr(model, "adapter") and isinstance(model.adapter, torch.nn.Module)

def save_clip_weights(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _is_adapter(model):
        torch.save({"adapter": model.adapter.state_dict(),
                    "alpha": getattr(model, "alpha", None),
                    "beta" : getattr(model, "beta",  None)}, path)
        print(f"✅ Adapter weights saved to {path}")
    else:
        torch.save(model.state_dict(), path)
        print(f"✅ Full model weights saved to {path}")

def load_clip_weights(model, path, device="cpu"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    try:  
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if _is_adapter(model) and "adapter" in ckpt:
        try:                                           
            model.adapter.load_state_dict(ckpt["adapter"], strict=True)
            print(f"🔄 Adapter weights restored from {path}")
        except RuntimeError as e:
            print(f"⚠️  Adapter shape mismatch ({e}); skipping weight matrix but keeping α/β.")
        if isinstance(ckpt.get("alpha"), torch.Tensor):
            model.alpha = ckpt["alpha"].item() if torch.is_tensor(ckpt["alpha"]) else ckpt["alpha"]
        if isinstance(ckpt.get("beta"), torch.Tensor):
            model.beta  = ckpt["beta"].item()  if torch.is_tensor(ckpt["beta"])  else ckpt["beta"]
    else:
        model.load_state_dict(ckpt)
        print(f"🔄 Full model weights restored from {path}")

def log_time_memory_txt(args, elapsed_sec: float, peak_mem_gb: float):
    backbone = args.backbone.replace("/", "-")
    optimiser = args.optimizer.lower()
    base_dir = "time_memory_with_aug" if getattr(args, "use_aug", False) else "time_memory_without_aug"
    fname = (
        f"./{base_dir}/time_memory_dataset_{args.dataset}"
        f"_shots_{args.shots}_backbone_{backbone}"
        f"_opt_{optimiser}_seed_{args.seed}.txt"
    )
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        f.write(f"Elapsed_time_seconds: {elapsed_sec:.2f}\n")
        f.write(f"Peak_gpu_memory_GB:   {peak_mem_gb:.2f}\n")
    print(f"⏱️  Time/Mem stats saved to {fname}")
    
def log_results_to_csv(args, epoch_idx, train_accuracy, test_accuracy, loss):
    backbone = args.backbone.replace("/", "-")
    opt      = args.optimizer.lower()
    base_dir = "results_with_aug" if getattr(args, "use_aug", False) else "results_without_aug"
    csv_file = f"./{base_dir}/training_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Dataset", "Optimizer", "Backbone", "Random Seed", "Shots", "Train Accuracy", "Test Accuracy", "Loss"])
        writer.writerow([epoch_idx, args.dataset, args.optimizer, args.backbone, args.seed, args.shots, train_accuracy, test_accuracy, loss])

def log_results_txt(args, ep, tr, te, loss):
    bb = args.backbone.replace("/", "-")
    base_dir = "results_with_aug" if getattr(args, "use_aug", False) else "results_without_aug"
    txt_file  = f"./{base_dir}/final_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{args.optimizer.lower()}_seed_{args.seed}.txt"
    file_exists = not os.path.isfile(txt_file)
    with open(txt_file, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Dataset", "Shots", "Seed", "Train Acc", "Test Acc", "Loss"])
        writer.writerow([ep, args.dataset, args.shots, args.seed, tr, te, loss])
# --------------------------------------------------------------------------- #
# TIP-Adapter-F core
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _build_cache(loader, clip_model, device):
    feats, labels = [], []
    for ims, y in tqdm(loader, desc="⋯ building cache"):
        ims = ims.to(device)
        y = y.to(device)
        f   = clip_model.encode_image(ims)
        f   = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
        labels.append(y)
    feats  = torch.cat(feats)                       # K × D
    labels = torch.cat(labels)                      # K
    D      = feats.size(1)
    C      = int(labels.max()) + 1
    keys   = feats.t().contiguous()                 # D × K
    vals   = torch.zeros(feats.size(0), C, device=device)
    vals[torch.arange(feats.size(0)), labels] = 1.
    return keys, vals                               # [D,K], [K,C]

class TipAdapterF(nn.Module):
    """Linear adapter initialised with cached keys."""
    def __init__(self, cache_keys, alpha=0.27, beta=20.):
        super().__init__()
        D, K = cache_keys.shape
        self.adapter = nn.Linear(D, K, bias=False)
        self.adapter.weight.data.copy_(cache_keys.t())    # K × D
        self.alpha = alpha
        self.beta  = beta

    def tip_logits(self, feats, cache_vals, clip_logits):
        aff   = self.adapter(feats)                       # N × K
        cache = torch.exp(-(self.beta - self.beta * aff)) # N × K
        cache = cache @ cache_vals                        # N × C
        return clip_logits + self.alpha * cache           # N × C

    def forward(self, feats, cache_vals, clip_logits, target):
        logits = self.tip_logits(feats, cache_vals, clip_logits)
        loss   = F.cross_entropy(logits, target)
        return loss, logits

# --------------------------------------------------------------------------- #
# Main training routine (replaces old CLIP-Adapter flow)
# --------------------------------------------------------------------------- #
def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader, val_loader):
    VALIDATION = False           # keep option for mid-epoch eval if desired
    device      = args.device
    clip_model.requires_grad_(False) 
    clip_model.eval()

    # 0-shot baseline ---------------------------------------------------------
    zs_acc = evaluate_zero_shot(clip_model, test_loader, dataset, device, logit_scale)
    print(f"\n**** Zero-shot CLIP test acc: {zs_acc:.2f}. ****\n")

    # ─────────── time & memory bookkeeping ───────────
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    
    # ------------------------------------------------------------------------ #
    # 1. Build cache & zeroshot classifier
    # ------------------------------------------------------------------------ #
    cache_keys, cache_vals = _build_cache(train_loader, clip_model, device)
    text_features = build_text_features(dataset, clip_model, device)

    # ------------------------------------------------------------------------ #
    # 2. Initialise adapter + optimiser
    # ------------------------------------------------------------------------ #
    tip_f    = TipAdapterF(cache_keys, alpha=args.alpha, beta=args.beta).to(device)
    
    # --------------------------------------------------------------------- #
    # Eval-only shortcut: load weights, run a single test, quit
    # --------------------------------------------------------------------- #
    if args.eval_only:
        weight_path = (args.weight_file if getattr(args, "weight_file", None) else os.path.join(args.save_path, f"dataset_{args.dataset}_shots_{args.shots}_backbone_{args.backbone.replace('/','-')}_opt_{args.optimizer.lower()}_seed_{args.seed}.pt"))
        load_clip_weights(tip_f, weight_path, device)
        acc_test = evaluate_tipf(tip_f, clip_model, test_loader, cache_vals, text_features, device, logit_scale)
        print(f"**** Final test accuracy: {acc_test:.2f}. ****\n")
        if args.exp in ["cross-transfer", "domain-generalization"]:
            log_results_txt(args, ep=0, tr=float('nan'), te=acc_test, loss=float('nan'))
        return

    # ------------------------------------------------------------------------ #
    # Initialise  optimiser
    # ------------------------------------------------------------------------ #
    optimizer= Adam(tip_f.adapter.parameters(), lr=args.lr) if args.optimizer == "Adam" else SGD(tip_f.adapter.parameters(), lr=args.lr)
    scaler   = GradScaler("cuda")

    # ------------------------------------------------------------------------ #
    # 3. Training loop
    # ------------------------------------------------------------------------ #
    epoch_idx = 0
    
    
    
    # ────────────────────── Training ──────────────────────
    while epoch_idx < args.max_epochs:
        epoch_idx += 1
        tip_f.train()
        tot_acc = tot_loss = tot_samples = 0

        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            with autocast("cuda"):
                img_feat = clip_model.encode_image(images)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                clip_logits = logit_scale * img_feat @ text_features.t()
                loss, logits = tip_f(img_feat, cache_vals, clip_logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            tot_acc   += cls_acc(logits, target) * target.size(0)
            tot_loss  += loss.item()             * target.size(0)
            tot_samples += target.size(0)

        train_acc  = tot_acc  / tot_samples
        epoch_loss = tot_loss / tot_samples

        # optional mid-training evaluation -----------------------------------
        if VALIDATION:
            acc_test = evaluate_tipf(tip_f, clip_model, test_loader, cache_vals, text_features, device, logit_scale)
            print(f"[Ep {epoch_idx:02d}] Train {train_acc:.2f} | Test {acc_test:.2f} | Loss {epoch_loss:.4f}")
            log_results_to_csv(args, epoch_idx, train_acc, acc_test, epoch_loss)
        else:
            print(f"[Ep {epoch_idx:02d}] Train {train_acc:.2f} | Loss {epoch_loss:.4f}")
        
        elapsed = time.perf_counter() - start_time
        print(f"Time for one epoch: {elapsed}" )

    # ─────────── write time & memory log ─────────────
    elapsed = time.perf_counter() - start_time
    peak_mem_bytes = torch.cuda.max_memory_allocated(args.device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    log_time_memory_txt(args, elapsed, peak_mem_gb)
    
    # ------------------------------------------------------------------------ #
    # 4. Final evaluation & saving
    # ------------------------------------------------------------------------ #
    tip_f.eval()
    acc_test = evaluate_tipf(tip_f, clip_model, test_loader, cache_vals, text_features, device, logit_scale)
    print(f"**** Final test accuracy: {acc_test:.2f}. ****\n")
    log_results_txt(args, epoch_idx, train_acc, acc_test, epoch_loss)

    if args.save_path:
        backbone = args.backbone.replace("/", "-")
        opt      = args.optimizer.lower()
        fname    = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"
        save_path= os.path.join(args.save_path, fname)
        save_clip_weights(tip_f, save_path)        
        print(f"✓ Weights saved to {save_path}")


#!/usr/bin/env python3
# train.py — LP++ (Linear Probe ++) baseline, final clean version
# ---------------------------------------------------------------------- #
import csv, os, clip
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from utils import cls_acc
import time

# ---------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------- #
@torch.no_grad()
def build_text_features(ds, clip_model, device):
    template = ds.template[0]
    tokens   = clip.tokenize([template.format(c.replace('_',' ')) for c in ds.classnames]).to(device)
    txt_f    = clip_model.encode_text(tokens)
    return txt_f / txt_f.norm(dim=-1, keepdim=True)         # C × D

@torch.no_grad()
def evaluate_zero_shot(clip_model, loader, ds, device, logit_scale):
    txt_f = build_text_features(ds, clip_model, device)
    acc, n = 0.0, 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        acc   += cls_acc(logit_scale * img_f @ txt_f.T, y) * y.size(0)
        n     += y.size(0)
    return acc / n

@torch.no_grad()
def _eval_lp(clf, alpha, clip_model, loader, txt_f, device):
    clf.eval()
    correct = n = 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = clf(img_f) + alpha * (img_f @ txt_f.T)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return 100. * correct / n

# ---------------------------------------------------------------------- #
# LP++ maths (paper §3.2)
# ---------------------------------------------------------------------- #
def _centroids(X, y, C):
    return torch.stack([X[y == c].sum(0) for c in range(C)])

def _lr_w(X):
    s_max = torch.linalg.svdvals(X).max()
    return 4 * X.size(0) / (s_max**2)

def _lr_alpha(X, text_f):
    m = ((X @ text_f.T) ** 2).sum(0).max()
    return X.size(0) / (4 * m)

def _alpha_init(X, y, shots, txt_f):
    cent_mean = torch.stack([X[y == c].mean(0) for c in range(txt_f.size(0))])
    alpha_tilde = (cent_mean @ txt_f.T).diag() * shots     
    return 250.0 / shots * alpha_tilde.mean() 

# ---------------------------------------------------------------------- #
# Simple CSV / TXT logging utilities
# ---------------------------------------------------------------------- #
def log_time_memory_txt(args, elapsed_sec: float, peak_mem_gb: float):
    backbone = args.backbone.replace("/", "-")
    optimiser = args.optimizer.lower()
    base_dir = "time_memory_with_aug" if getattr(args, "use_aug", False) else "time_memory_without_aug"
    fname = (
        f"./{base_dir}/time_memory/time_memory_dataset_{args.dataset}"
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

def log_results_to_txt(args, epoch_idx, train_accuracy, test_accuracy, loss):
    backbone = args.backbone.replace("/", "-")
    opt      = args.optimizer.lower()
    base_dir = "results_with_aug" if getattr(args, "use_aug", False) else "results_without_aug"
    txt_file = f"./{base_dir}/final_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.txt"
    file_exists = os.path.isfile(txt_file)
    with open(txt_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Dataset", "Shots", "Seed", "Train Accuracy", "Test Accuracy", "Loss"])
        writer.writerow([epoch_idx, args.dataset, args.shots, args.seed, train_accuracy, test_accuracy, loss])

# ---------------------------------------------------------------------- #
# Save / load state
# ---------------------------------------------------------------------- #
def _save_state(clf, alpha, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"classifier": clf.state_dict(), "alpha": alpha}, path)
    print(f"✅ Saved adapter to {path}")

def _load_state(clf, path, device):
    ckpt = torch.load(path, map_location=device)
    clf.load_state_dict(ckpt["classifier"])
    print(f"🔄 Loaded adapter from {path}")
    return ckpt["alpha"].to(device)

# ---------------------------------------------------------------------- #
# Main routine
# ---------------------------------------------------------------------- #
def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader, val_loader):

    device      = args.device
    VALIDATION  = False            # set True for mid‑epoch test accuracy
    clip_model.eval().requires_grad_(False)

    # ----- zero‑shot baseline ---------------------------------------- #
    zs = evaluate_zero_shot(clip_model, test_loader, dataset, device, logit_scale)
    print(f"\n**** Zero-shot CLIP test acc: {zs:.2f}. ****\n")

    # ─────────── time & memory bookkeeping ───────────
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    
    # ----- extract few‑shot features -------------------------------- #
    feats, labels = [], []
    with torch.no_grad():
        for ims, y in tqdm(train_loader, desc="⋯ extract"):
            ims, y = ims.to(device), y.to(device)
            f = clip_model.encode_image(ims)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
            labels.append(y)
    X = torch.cat(feats)
    y = torch.cat(labels)         # N×D,  N
    C, D = len(dataset.classnames), X.size(1)
    txt_f = build_text_features(dataset, clip_model, device)

    # ----- initialise w, α ------------------------------------------ #
    clf = nn.Linear(D, C, bias=True, dtype=X.dtype, device=device)
    clf.weight.data = _centroids(X, y, C)
    alpha = torch.full((1, C), _alpha_init(X, y, args.shots, txt_f), dtype=X.dtype, device=device, requires_grad=True)
    lr_w     = _lr_w(X)
    lr_alpha = _lr_alpha(X, txt_f)
    print(f"lr_w={lr_w:.2e} | lr_alpha={lr_alpha:.2e} | alpha_init={alpha.mean().item():.4f}")

    # ----- eval‑only shortcut --------------------------------------- #
    if args.eval_only:
        backbone = args.backbone.replace("/", "-")
        opt      = args.optimizer.lower()
        fname    = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"
        alpha = _load_state(clf, os.path.join(args.save_path, fname), device)
        te = _eval_lp(clf, alpha, clip_model, test_loader, txt_f, device)
        print(f"**** Test accuracy: {te:.2f}. ****")
        return

    # ----- optimiser (SGD / Adam) -------------------------------- #
    optim= Adam(clf.parameters(), lr=lr_w) if args.optimizer == "Adam" else SGD(clf.parameters(), lr=lr_w, momentum=0.9)
    epoch = 0
        
    # ────────────────────── Training ──────────────────────
    while epoch < args.max_epochs:
        epoch += 1
        clf.train()
        optim.zero_grad(set_to_none=True)

        logits_v = clf(X)                  # N×C
        logits_t = X @ txt_f.T             # N×C
        logits   = logits_v + alpha * logits_t
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()

        # manual α update **every epoch** (Alg. 1)
        alpha.data -= lr_alpha * alpha.grad.data
        alpha.grad.zero_()

        tr_acc = cls_acc(logits, y)

        # optional mid‑epoch evaluation
        if VALIDATION:
            te_acc = _eval_lp(clf, alpha, clip_model, test_loader, txt_f, device)
            log_results_to_csv(args, epoch, tr_acc, te_acc, loss.item())
            print(f"[Ep {epoch:02d}] Train {tr_acc:.2f} | Test {te_acc:.2f} | Loss {loss.item():.4f}")
        else:
            print(f"[Ep {epoch:02d}] Train {tr_acc:.2f} | Loss {loss.item():.4f}")
        
        elapsed = time.perf_counter() - start_time
        print(f"Time for one epoch: {elapsed}" )

    # ─────────── write time & memory log ─────────────
    elapsed = time.perf_counter() - start_time
    peak_mem_bytes = torch.cuda.max_memory_allocated(args.device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    log_time_memory_txt(args, elapsed, peak_mem_gb)
    
    # ----- final evaluation & save ---------------------------------- #
    te_acc = _eval_lp(clf, alpha, clip_model, test_loader, txt_f, device)
    print(f"**** Final test accuracy: {te_acc:.2f}. ****\n")
    log_results_to_txt(args, epoch, tr_acc, te_acc, loss.item())

    if args.save_path:
        backbone = args.backbone.replace("/", "-")
        opt      = args.optimizer.lower()
        fname    = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"
        save_path= os.path.join(args.save_path, fname)
        _save_state(clf, alpha.detach(), save_path)      
        print(f"✓ Weights saved to {save_path}")
        
        


# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import csv, os
import time
import clip
import torch
import torch.nn.functional as F
from utils import cls_acc                            
from torch import autocast, GradScaler
from torch.optim import Adam, SGD
from clip_adapter import CLIPAdapter

def _is_adapter(model):
    return hasattr(model, "adapter") and isinstance(model.adapter, torch.nn.Module)

def save_clip_weights(model, path):
    """Save only adapter weights if model is a CLIPAdapter, else full model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _is_adapter(model):
        torch.save({"adapter": model.adapter.state_dict(),
                    "alpha": getattr(model, "alpha", None)}, path)
        print(f"✅ Adapter weights saved to {path}")
    else:
        torch.save(model.state_dict(), path)
        print(f"✅ Full model weights saved to {path}")

def load_clip_weights(model, path, device="cpu"):
    try:  
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if _is_adapter(model) and "adapter" in ckpt:
        model.adapter.load_state_dict(ckpt["adapter"])
        if isinstance(ckpt.get("alpha"), torch.Tensor):
            model.alpha.data.copy_(ckpt["alpha"])
        print(f"🔄 Adapter weights restored from {path}")
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
    if args.exp == 'cross-transfer':
        f  = f"./{base_dir}/ct_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{args.optimizer.lower()}_seed_{args.seed}.txt"
    elif args.exp == 'domain-generalization':
        f  = f"./{base_dir}/dg_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{args.optimizer.lower()}_seed_{args.seed}.txt"
    else:
        f  = f"./{base_dir}/final_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{args.optimizer.lower()}_seed_{args.seed}.txt"
    new = not os.path.isfile(f)
    with open(f, "a", newline="") as fd:
        w = csv.writer(fd)
        if new:
            w.writerow(["Epoch","Dataset","Shots","Seed","Train Acc","Test Acc","Loss"])
        w.writerow([ep, args.dataset, args.shots, args.seed, tr, te, loss])

def evaluate_CLIP(clip_model, loader, dataset, device):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    acc, tot_samples = 0.0, 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    return acc / tot_samples

def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader, val_loader):
    VALIDATION = False
    clip_model.requires_grad_(False) 
    clip_model.eval()
    zs_acc = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
    print(f"\n**** Zero-shot CLIP test acc: {zs_acc:.2f}. ****\n")

    # ---------------- Wrap backbone with adapter ---------------------------
    clip_model = CLIPAdapter(
        clip_model,
        alpha=getattr(args, "alpha", 0.2),
        reduction=getattr(args, "reduction", 4),
        learn_alpha=False,
        logit_scale=logit_scale
    ).to(args.device)

    torch.cuda.empty_cache()

    # ---------------- Optional eval-only -----------------------------------
    if args.eval_only:
        weight_path = (args.weight_file if getattr(args, "weight_file", None) else os.path.join(args.save_path, f"dataset_{args.dataset}_shots_{args.shots}_backbone_{args.backbone.replace('/','-')}_opt_{args.optimizer.lower()}_seed_{args.seed}.pt"))
        load_clip_weights(clip_model, weight_path, args.device)
        acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
        print(f"**** Final test accuracy: {acc_test:.2f}. ****\n")
        if args.exp in ["cross-transfer", "domain-generalization"]:
            log_results_txt(args, ep=0, tr=float('nan'), te=acc_test, loss=float('nan'))
        return
    
    # ---------------- Optimiser (adapter params only) ----------------------
    opt_map = {"Adam": Adam, "SGD": SGD}
    optimizer = opt_map[args.optimizer](clip_model.adapter.parameters(), lr=args.lr)
    scaler = GradScaler("cuda")

    # ---------------- Tokenise class names once -----------------------------
    texts = clip.tokenize(
        [dataset.template[0].format(c.replace('_', ' ')) for c in dataset.classnames]
    ).to(args.device)

    # ─────────── time & memory bookkeeping ───────────
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    
    # ---------------- Training loop ----------------------------------------
    epoch_idx = 0
    while epoch_idx < args.max_epochs:
        epoch_idx += 1
        clip_model.train()
        tot_acc = tot_loss = tot_samples = 0

        for images, target in train_loader:
            images, target = images.to(args.device), target.to(args.device)
            with autocast("cuda"):
                img_feat = clip_model.encode_image(images)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = clip_model.encode_text(texts)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                logits = logit_scale * img_feat @ txt_feat.t()
                loss = F.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            tot_acc   += cls_acc(logits, target) * target.size(0)
            tot_loss  += loss.item() * target.size(0)
            tot_samples += target.size(0)

        train_acc  = tot_acc  / tot_samples
        epoch_loss = tot_loss / tot_samples

        if VALIDATION:
            clip_model.eval()
            acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
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
    
    # ---------------- Final evaluation & save weights ----------------------
    clip_model.eval()
    acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
    print(f"**** Final test accuracy: {acc_test:.2f}. ****\n")
    log_results_txt(args, epoch_idx, train_acc, acc_test, epoch_loss)
    if args.save_path:
        backbone = args.backbone.replace("/", "-")
        opt = args.optimizer.lower()  
        fname = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"
        save_path = os.path.join(args.save_path, fname)
        save_clip_weights(clip_model, save_path)
        print(f"✓ Weights saved to {save_path}")

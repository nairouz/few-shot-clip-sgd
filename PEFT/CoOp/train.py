#!/usr/bin/env python3
# train_coop.py — CoOp prompt‑tuning trainer (drop‑in, minimal fix)

import os, csv, clip, torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from utils import cls_acc
from torch.optim import Adam, SGD
#from torch.cuda.amp import autocast, GradScaler
from torch import autocast, GradScaler 

# ---------------------------------------------------------------------- #
# Helper utilities
# ---------------------------------------------------------------------- #
@torch.no_grad()
def build_text_features(ds, clip_model, device):
    template = ds.template[0]
    tokens   = clip.tokenize([template.format(c.replace('_', ' ')) for c in ds.classnames]).to(device)
    txt_f    = clip_model.encode_text(tokens)
    return txt_f / txt_f.norm(dim=-1, keepdim=True)

@torch.no_grad()
def evaluate_zero_shot(clip_model, loader, ds, device, logit_scale):
    txt_f = build_text_features(ds, clip_model, device)
    acc, n = 0., 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        acc   += cls_acc(logit_scale * img_f @ txt_f.T, y) * y.size(0)
        n     += y.size(0)
    return acc / n

@torch.no_grad()
def evaluate_zero_shot_b2n(clip_model, loader, ds, device, logit_scale):
    clip_model.eval()
    base_idx = torch.tensor(ds.base_cls, device=device, dtype=torch.long)
    new_idx  = torch.tensor(ds.new_cls,  device=device, dtype=torch.long)
    template   = ds.template[0]
    base_names = [template.format(ds.classnames[i].replace('_', ' ')) for i in base_idx.tolist()]
    new_names  = [template.format(ds.classnames[i].replace('_', ' ')) for i in new_idx.tolist()]
    tok_base = clip.tokenize(base_names).to(device)
    tok_new  = clip.tokenize(new_names).to(device)
    tf_base  = clip_model.encode_text(tok_base); tf_base = tf_base / tf_base.norm(dim=-1, keepdim=True)
    tf_new   = clip_model.encode_text(tok_new ); tf_new  = tf_new  / tf_new .norm(dim=-1, keepdim=True)
    map_base = torch.full((len(ds.classnames),), -1, dtype=torch.long, device=device)
    map_new  = torch.full((len(ds.classnames),), -1, dtype=torch.long, device=device)
    map_base[base_idx] = torch.arange(len(base_idx), device=device)
    map_new [new_idx ] = torch.arange(len(new_idx ), device=device)
    corr_b = corr_n = tot_b = tot_n = 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        yb = map_base[y]
        mask_b = (yb != -1)
        if mask_b.any():
            logits_b = logit_scale * img_f[mask_b] @ tf_base.T
            corr_b  += (logits_b.argmax(1) == yb[mask_b]).sum().item()
            tot_b   += int(mask_b.sum())
        yn = map_new[y]
        mask_n = (yn != -1)
        if mask_n.any():
            logits_n = logit_scale * img_f[mask_n] @ tf_new.T
            corr_n  += (logits_n.argmax(1) == yn[mask_n]).sum().item()
            tot_n   += int(mask_n.sum())
    acc_b = 100. * corr_b / max(tot_b, 1)
    acc_n = 100. * corr_n / max(tot_n, 1)
    h     = 2 * acc_b * acc_n / max(acc_b + acc_n, 1)
    return acc_b, acc_n, h

@torch.no_grad()
def _eval_coop(model, clip_model, loader, device):
    model.eval()
    corr = tot = 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = model(img_f)
        corr  += (logits.argmax(1) == y).sum().item()
        tot   += y.size(0)
    return 100. * corr / tot

@torch.no_grad()
def eval_coop_b2n(coop_model, clip_model, loader, ds, device):
    coop_model.eval()
    base_idx = torch.tensor(ds.base_cls, device=device, dtype=torch.long)
    new_idx  = torch.tensor(ds.new_cls,  device=device, dtype=torch.long)
    id_map_b = torch.full((len(ds.classnames),), -1, dtype=torch.long, device=device)
    id_map_b[base_idx] = torch.arange(base_idx.numel(), device=device)
    id_map_n = torch.full((len(ds.classnames),), -1, dtype=torch.long, device=device)
    id_map_n[new_idx]  = torch.arange(new_idx.numel(), device=device)
    corr_b = corr_n = tot_b = tot_n = 0
    for ims, y in loader:
        ims, y = ims.to(device), y.to(device)
        img_f  = clip_model.encode_image(ims)
        img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
        logits_all = coop_model(img_f)
        mask_b = id_map_b[y] != -1
        if mask_b.any():
            logits_b = logits_all[mask_b][:, base_idx]
            y_b      = id_map_b[y[mask_b]]
            corr_b  += (logits_b.argmax(1) == y_b).sum().item()
            tot_b   += y_b.numel()
        mask_n = id_map_n[y] != -1
        if mask_n.any():
            logits_n = logits_all[mask_n][:, new_idx]
            y_n      = id_map_n[y[mask_n]]
            corr_n  += (logits_n.argmax(1) == y_n).sum().item()
            tot_n   += y_n.numel()
    acc_b = 100. * corr_b / max(tot_b, 1)
    acc_n = 100. * corr_n / max(tot_n, 1)
    h     = 2 * acc_b * acc_n / max(acc_b + acc_n, 1)
    return acc_b, acc_n, h

# ---------------------------------------------------------------------- #
# CoOp modules (prompt learner, encoder, wrapper)
# ---------------------------------------------------------------------- #
class PromptLearner(nn.Module):
    """Learn N_CTX context tokens per class (CoOp)."""
    def __init__(self, n_ctx, classnames, clip_model, ctx_init=""):
        super().__init__()
        dtype = getattr(clip_model, "dtype", clip_model.token_embedding.weight.dtype)
        self.dtype = dtype


        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls, self.n_ctx = len(classnames), n_ctx

        if ctx_init:                                 # manual prompt
            with torch.no_grad():
                tok = clip.tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
                embeds = clip_model.token_embedding(tok).type(self.dtype)

            ctx = embeds[0, 1:1+n_ctx, :]
        else:                                        # random init
            ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype).normal_(std=0.02)
        self.ctx = nn.Parameter(ctx)                 # trainable

        prefix  = " ".join(["X"] * n_ctx)
        prompts = [f"{prefix} {c.replace('_',' ')}." for c in classnames]
            
        with torch.no_grad():
            tokenized = clip.tokenize(prompts).to(clip_model.token_embedding.weight.device)
            embeds    = clip_model.token_embedding(tokenized).type(self.dtype)

        self.register_buffer("token_prefix", embeds[:, :1, :])
        self.register_buffer("token_suffix", embeds[:, 1+n_ctx:, :])
        self.tokenized_prompts = tokenized

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], 1)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer        = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final           = clip_model.ln_final
        self.text_projection    = clip_model.text_projection
        self.dtype              = (clip_model.dtype if hasattr(clip_model, "dtype") else clip_model.token_embedding.weight.dtype)

    def forward(self, prompts, tokenized):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.size(0)), tokenized.argmax(-1)]
        return x @ self.text_projection

class CoOp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx    = getattr(cfg, "n_ctx", 16)
        ctx_init = getattr(cfg, "ctx_init", "")
        self.prompt_learner = PromptLearner(n_ctx, classnames, clip_model, ctx_init)
        self.tokenized      = self.prompt_learner.tokenized_prompts
        self.text_encoder   = TextEncoder(clip_model)
        self.image_encoder  = clip_model.visual
        self.logit_scale    = clip_model.logit_scale

    def forward(self, image_feats):
        prompts = self.prompt_learner()
        txt_f   = self.text_encoder(prompts, self.tokenized)
        txt_f   = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_feats @ txt_f.T

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
    
# ---------------------------------------------------------------------- #
# CSV / TXT logging (unchanged)
# ---------------------------------------------------------------------- #
def log_results_to_csv(args, epoch_idx, train_accuracy, test_accuracy, loss):
    backbone = args.backbone.replace("/", "-")
    opt      = args.optimizer.lower()
    base_dir = "results_with_aug" if getattr(args, "use_aug", False) else "results_without_aug"
    if args.exp == 'base-to-new':
        csv_path = f"./{base_dir}/b2n_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.csv"
    else:
        csv_path = f"./{base_dir}/training_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.csv"
    is_new_file = not os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if is_new_file:
            if args.exp == 'base-to-new':
                writer.writerow(["Epoch", "Dataset", "Optimizer", "Backbone", "Seed", "Shots", "Train Acc", "BaseAcc", "NewAcc", "H-Mean", "Loss"])
            else:
                writer.writerow(["Epoch", "Dataset", "Optimizer", "Backbone", "Seed", "Shots", "Train Acc", "Test Acc", "Loss"])
        if args.exp == 'base-to-new':
            base_acc, new_acc, hmean = test_accuracy
            writer.writerow([epoch_idx, args.dataset, args.optimizer, args.backbone, args.seed, args.shots, train_accuracy, base_acc, new_acc, hmean, loss])
        else:
            writer.writerow([epoch_idx, args.dataset, args.optimizer, args.backbone, args.seed, args.shots, train_accuracy, test_accuracy, loss])
            
def log_results_txt(args, ep, tr, te, loss):
    bb = args.backbone.replace("/", "-")
    opt = args.optimizer.lower()
    base_dir = "results_with_aug" if getattr(args, "use_aug", False) else "results_without_aug"
    if   args.exp == 'cross-transfer':
        fn = f"./{base_dir}/ct_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{opt}_seed_{args.seed}.txt"
    elif args.exp == 'domain-generalization':
        fn = f"./{base_dir}/dg_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{opt}_seed_{args.seed}.txt"
    elif args.exp == 'base-to-new':
        fn = f"./{base_dir}/b2n_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{opt}_seed_{args.seed}.txt"
    else:
        fn = f"./{base_dir}/final_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{bb}_opt_{opt}_seed_{args.seed}.txt"
    fresh = not os.path.isfile(fn)
    with open(fn, "a", newline="") as fd:
        w = csv.writer(fd)
        if fresh:
            if args.exp == 'base-to-new':
                w.writerow(["Epoch", "Dataset", "Shots", "Seed","Train Acc", "BaseAcc", "NewAcc", "H-Mean", "Loss"])
            else:
                w.writerow(["Epoch", "Dataset", "Shots", "Seed","Train Acc", "Test Acc", "Loss"])
        if args.exp == 'base-to-new':
            base_acc, new_acc, hmean = te     # `te` expected to be a tuple
            w.writerow([ep, args.dataset, args.shots, args.seed, tr, base_acc, new_acc, hmean, loss])
        else:
            w.writerow([ep, args.dataset, args.shots, args.seed, tr, te, loss])

# ---------------------------------------------------------------------- #
def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader, val_loader):

    device = args.device
    VALID  = False
    clip_model.eval().requires_grad_(False)

    if args.exp == "base-to-new":
        acc_test = evaluate_zero_shot_b2n(clip_model, test_loader, dataset, device, logit_scale)
        acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
        print(f"Zero-shot CLIP's : 🌱 Base acc: {acc_base:.2f} | 🆕 New acc: {acc_new:.2f} | H-Mean: {hmean:.2f}")
    zs_acc =  evaluate_zero_shot(clip_model, test_loader, dataset, device, logit_scale)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    # 1) build CoOp
    coop = CoOp(args, dataset.classnames, clip_model).to(device)

    # freeze everything except prompt learner
    for n, p in coop.named_parameters():
        p.requires_grad_(("prompt_learner" in n))

    # 2) eval‑only shortcut ────────────────────────────────────────────────
    if args.eval_only:
        weight_path = (args.weight_file if getattr(args, "weight_file", None) else os.path.join(args.save_path, f"dataset_{args.dataset}_shots_{args.shots}_backbone_{args.backbone.replace('/','-')}_opt_{args.optimizer.lower()}_seed_{args.seed}.pt"))
        try:  
            ckpt = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(weight_path, map_location=device)
        own_state  = coop.prompt_learner.state_dict()
        filtered   = {k: v for k, v in ckpt.items() if k in own_state and v.shape == own_state[k].shape}
        coop.prompt_learner.load_state_dict(filtered, strict=False)
        
        # --------------- choose the proper evaluator ------------------
        if args.exp == "base-to-new":
            test_acc = eval_coop_b2n(coop, clip_model, test_loader, dataset, device) 
            b, n, h = test_acc
            print(f"🌱 Base {b:.2f} | 🆕 New {n:.2f} | H-Mean {h:.2f}")
        else:
            test_acc = _eval_coop(coop, clip_model, test_loader, device)
            print(f"**** Final test accuracy: {test_acc:.2f}. ****\n")
        if args.exp in ["cross-transfer", "domain-generalization"]:
            log_results_txt(args, ep=0, tr=float('nan'), te=test_acc, loss=float('nan'))
        return

    # 3) optimiser
    opt_cls = Adam if args.optimizer == "Adam" else SGD
    optim   = opt_cls(coop.prompt_learner.parameters(), lr=args.lr)
    scaler = GradScaler("cuda")

    # ---- B2N: precompute compact base indexing ----
    if args.exp == "base-to-new":
        base_idx = torch.tensor(dataset.base_cls, device=device, dtype=torch.long)
        id_map   = torch.full((len(dataset.classnames),), -1, dtype=torch.long, device=device)
        id_map[base_idx] = torch.arange(base_idx.numel(), device=device)
    
    # ─────────── time & memory bookkeeping ───────────
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    
    # 4) training loop
    for ep in range(1, args.max_epochs + 1):
        coop.train()
        tot_n = tot_acc = tot_loss = 0

        for ims, y in train_loader:
            ims, y = ims.to(device), y.to(device)
            with autocast("cuda"):
                img_f  = clip_model.encode_image(ims)
                img_f /= img_f.norm(dim=-1, keepdim=True)
                logits = coop(img_f)
                if args.exp == "base-to-new":
                    logits_b = logits[:, base_idx]
                    y_b      = id_map[y]
                    if (y_b < 0).any():
                        keep     = (y_b != -1)
                        logits_b = logits_b[keep]
                        y_b      = y_b[keep]
                    loss = F.cross_entropy(logits_b, y_b)
                    batch_eff = logits_b.size(0)
                    tot_n    += batch_eff
                    tot_acc  += cls_acc(logits_b, y_b) * batch_eff
                    tot_loss += loss.item() * batch_eff
                else:
                    loss = F.cross_entropy(logits, y)
                    tot_n    += y.size(0)
                    tot_acc  += cls_acc(logits, y) * y.size(0)
                    tot_loss += loss.item() * y.size(0)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim) 
            scaler.step(optim)
            scaler.update()
            
        tr_acc = tot_acc / tot_n
        ep_loss= tot_loss / tot_n

        if VALID:
            coop.eval()
            if args.exp == "base-to-new":
                test_acc = eval_coop_b2n(coop, clip_model, test_loader, dataset, device) 
                b, n, h = test_acc
                print(f"[Epoch {ep:02d}] Train Acc: {tr_acc:.2f}, Base Acc: {b:.2f}, New Acc: {n:.2f}, H-Mean: {h:.2f}, Loss: {ep_loss:.4f}")
            else:
                test_acc = _eval_coop(coop, clip_model, test_loader, device)
                print(f"[Epoch {ep:02d}] Train Acc: {tr_acc:.2f}, Loss: {ep_loss:.4f}")
            log_results_to_csv(args, ep, tr_acc, test_acc, ep_loss)
        else:
            print(f"[Ep {ep:02d}] Train {tr_acc:.2f} | Loss {ep_loss:.4f}")
        elapsed = time.perf_counter() - start_time
        print(f"Time for one epoch: {elapsed}" )
        
    # ─────────── write time & memory log ─────────────
    elapsed = time.perf_counter() - start_time
    peak_mem_bytes = torch.cuda.max_memory_allocated(args.device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    log_time_memory_txt(args, elapsed, peak_mem_gb)
    
    # 5) final evaluation & save
    coop.eval()
    if args.exp == "base-to-new":
        test_acc = eval_coop_b2n(coop, clip_model, test_loader, dataset, device) 
        b, n, h = test_acc
        print(f"🌱 Base acc: {b:.2f} | 🆕 New acc: {n:.2f} | H-Mean: {h:.2f}")
    else:
        test_acc = _eval_coop(coop, clip_model, test_loader, device)
        print(f"**** Final test accuracy: {test_acc:.2f}. ****\n")
    log_results_txt(args, ep, tr_acc, test_acc, ep_loss)

    if (args.save_path is not None) and (args.exp == "few-shot"):
        fname = (f"dataset_{args.dataset}_shots_{args.shots}_backbone_{args.backbone.replace('/','-')}_opt_{args.optimizer.lower()}_seed_{args.seed}.pt")
        torch.save(coop.prompt_learner.state_dict(), os.path.join(args.save_path, fname))
        print("✓ prompt weights saved")

# ---------------------------------------------------------------------- #

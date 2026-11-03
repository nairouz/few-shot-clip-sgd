import csv
import os
import time
import clip
import torch
import torch.nn.functional as F
from utils import *
from torch import autocast, GradScaler  
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora, load_lora
from torch.optim import Adam, SGD  

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

def evaluate_CLIP_base_new(model, loader, dataset, device):
    model.eval()
    tmpl = dataset.template[0]
    base_names = [dataset.classnames[i] for i in dataset.base_cls]
    new_names  = [dataset.classnames[i] for i in dataset.new_cls]
    toks_base  = clip.tokenize([tmpl.format(c.replace('_',' ')) for c in base_names]).to(device)
    toks_new   = clip.tokenize([tmpl.format(c.replace('_',' ')) for c in new_names]).to(device)
    with torch.no_grad():
        tf_base = model.encode_text(toks_base); tf_base = tf_base / tf_base.norm(dim=-1, keepdim=True)
        tf_new  = model.encode_text(toks_new ); tf_new  = tf_new  / tf_new.norm (dim=-1, keepdim=True)
    map_base = torch.full((len(dataset.classnames),), -1, dtype=torch.long, device=device)
    map_new  = torch.full((len(dataset.classnames),), -1, dtype=torch.long, device=device)
    map_base[torch.tensor(dataset.base_cls, device=device)] = torch.arange(len(dataset.base_cls), device=device)
    map_new [torch.tensor(dataset.new_cls , device=device)] = torch.arange(len(dataset.new_cls ), device=device)
    corr_b = corr_n = tot_b = tot_n = 0
    with torch.no_grad():
        for imgs, tgt in loader:
            imgs, tgt = imgs.to(device), tgt.to(device)
            img = model.encode_image(imgs); img = img / img.norm(dim=-1, keepdim=True)
            mask_b = map_base[tgt] != -1
            if mask_b.any():
                logits_b = img[mask_b] @ tf_base.t()
                targ_b   = map_base[tgt[mask_b]]
                corr_b  += (logits_b.argmax(1) == targ_b).sum().item()
                tot_b   += targ_b.numel()
            mask_n = map_new[tgt] != -1
            if mask_n.any():
                logits_n = img[mask_n] @ tf_new.t()
                targ_n   = map_new[tgt[mask_n]]
                corr_n  += (logits_n.argmax(1) == targ_n).sum().item()
                tot_n   += targ_n.numel()
    acc_b = 100. * corr_b / max(tot_b, 1)
    acc_n = 100. * corr_n / max(tot_n, 1)
    hmean = 2 * acc_b * acc_n / max(acc_b + acc_n, 1)
    return acc_b, acc_n, hmean

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

def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader):
    VALIDATION = False
    clip_model.eval()
    
    if args.exp == "base-to-new":
        acc_test = evaluate_CLIP_base_new(clip_model, test_loader, dataset, args.device)
        acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
        print(f"Zero-shot CLIP's : 🌱 Base acc: {acc_base:.2f} | 🆕 New acc: {acc_new:.2f} | H-Mean: {hmean:.2f}")
    zs_acc = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(args.device)
    
    if args.eval_only:
        if args.exp in ["cross-transfer", "domain-generalization"]:
            original_name = args.dataset
            args.dataset  = args.source_dataset 
            load_lora(args, list_lora_layers)
            args.dataset  = original_name
        else:
            load_lora(args, list_lora_layers)
            
        if args.exp == "base-to-new":
            acc_test = evaluate_CLIP_base_new(clip_model, test_loader, dataset, args.device)
            acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
            print(f"[Epoch {epoch_idx:02d}] Train Acc: {train_accuracy:.2f}, Base Acc: {acc_base:.2f}, New Acc: {acc_new:.2f}, H-Mean: {hmean:.2f}, Loss: {loss_epoch:.4f}")
            print(f"🌱 Base acc: {acc_base:.2f} | 🆕 New acc: {acc_new:.2f} | H-Mean: {hmean:.2f}")
        else:
            acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
            print(f"**** Test accuracy: {acc_test:.2f}. ****\n")
        if args.exp in ["cross-transfer", "domain-generalization"]:
            log_results_txt(args, ep=0, tr=float('nan'), te=acc_test, loss=float('nan'))
        return

    mark_only_lora_as_trainable(clip_model)
    opt_map = {"Adam" : lambda p: Adam(p, lr=args.lr), "SGD" : lambda p: SGD(p, lr=args.lr)}
    optimizer = opt_map[args.optimizer](get_lora_parameters(clip_model))
    scaler = GradScaler("cuda")                       
    epoch_idx = 0
    tmpl = dataset.template[0] 
    tokens_all  = clip.tokenize([tmpl.format(c.replace('_',' ')) for c in dataset.classnames]).to(args.device)

    # For B2N training: compact base-class indexing
    if args.exp == "base-to-new":
        base_names = [dataset.classnames[i] for i in dataset.base_cls]
        tokens_base = clip.tokenize([tmpl.format(c.replace('_',' ')) for c in base_names]).to(args.device)
        id_map = torch.full((len(dataset.classnames),), -1, dtype=torch.long, device=args.device)
        id_map[torch.tensor(dataset.base_cls, device=args.device)] = torch.arange(len(dataset.base_cls), device=args.device)
    
    # ─────────── time & memory bookkeeping ───────────
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    
    # ────────────────────── Training ──────────────────────
    while epoch_idx < args.max_epochs:
        epoch_idx += 1
        clip_model.train()
        train_accuracy, tot_samples, loss_epoch = 0, 0, 0.0
        for images, target in train_loader:
            images, target = images.to(args.device), target.to(args.device)
            with autocast("cuda"):
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                if args.exp == "base-to-new":
                    tf_base = clip_model.encode_text(tokens_base)
                    tf_base = tf_base / tf_base.norm(dim=-1, keepdim=True)
                    logits = logit_scale * (image_features @ tf_base.t())
                    target_b = id_map[target]
                    loss = F.cross_entropy(logits, target_b)
                    train_accuracy += cls_acc(logits, target_b) * target.size(0)  
                else:
                    tf_all = clip_model.encode_text(tokens_all)
                    tf_all = tf_all / tf_all.norm(dim=-1, keepdim=True)
                    logits = logit_scale * (image_features @ tf_all.t())
                    loss = F.cross_entropy(logits, target)
                    train_accuracy += cls_acc(logits, target) * target.size(0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()    
            scaler.unscale_(optimizer)           
            scaler.step(optimizer)                         
            scaler.update()
            loss_epoch += loss.item() * target.size(0)
            tot_samples += target.size(0)
        train_accuracy /= tot_samples
        loss_epoch /= tot_samples
        if VALIDATION:
            clip_model.eval()
            if args.exp == "base-to-new":
                acc_test = evaluate_CLIP_base_new(clip_model, test_loader, dataset, args.device)
                acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
                print(f"[Epoch {epoch_idx:02d}] Train Acc: {train_accuracy:.2f}, Base Acc: {acc_base:.2f}, New Acc: {acc_new:.2f}, H-Mean: {hmean:.2f}, Loss: {loss_epoch:.4f}")
            else:
                acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
                print(f"[Epoch {epoch_idx:02d}] Train Acc: {train_accuracy:.2f}, Test Acc: {acc_test:.2f}, Loss: {loss_epoch:.4f}")
            log_results_to_csv(args, epoch_idx, train_accuracy, acc_test, loss_epoch)
        else:
            print(f"[Epoch {epoch_idx:02d}] Train Acc: {train_accuracy:.2f},  Loss {loss_epoch:.4f}") 
        elapsed = time.perf_counter() - start_time
        print(f"Time for one epoch: {elapsed}" )            

    # ─────────── write time & memory log ─────────────
    elapsed = time.perf_counter() - start_time
    peak_mem_bytes = torch.cuda.max_memory_allocated(args.device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    log_time_memory_txt(args, elapsed, peak_mem_gb)
    
    # ---------- Base–to–New evaluation & logging ---------------------------
    clip_model.eval()
    if args.exp == "base-to-new":
        acc_test = evaluate_CLIP_base_new(clip_model, test_loader, dataset, args.device)
        acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
        print(f"🌱 Base acc: {acc_base:.2f} | 🆕 New acc: {acc_new:.2f} | H-Mean: {hmean:.2f}")
    else:
        acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)      
        print(f"**** Final test accuracy: {acc_test:.2f}. ****\n")
    log_results_txt(args, epoch_idx, train_accuracy, acc_test, loss_epoch)
    if (args.save_path is not None) and (args.exp == "few-shot"):
        save_lora(args, list_lora_layers)
    return


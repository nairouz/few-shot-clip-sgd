# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import csv
import os
import clip
import torch
from utils import *

def log_results_txt(args, ep, tr, te, loss):
    bb = args.backbone.replace("/", "-")
    if args.exp == 'base-to-new':
        fn = f"./results/b2n_results_dataset_{args.dataset}_backbone_{bb}_seed_{args.seed}.txt"
    else:
        fn = f"./results/final_results_dataset_{args.dataset}_backbone_{bb}_seed_{args.seed}.txt"
    fresh = not os.path.isfile(fn)
    with open(fn, "a", newline="") as fd:
        w = csv.writer(fd)
        if fresh:
            if args.exp == 'base-to-new':
                w.writerow(["Epoch", "Dataset", "Shots", "Seed", "Train Acc", "BaseAcc", "NewAcc", "H-Mean", "Loss"])
            else:
                w.writerow(["Epoch", "Dataset", "Shots", "Seed", "Train Acc", "Test Acc", "Loss"])
        if args.exp == 'base-to-new':
            base_acc, new_acc, hmean = te   
            w.writerow([ep, args.dataset, args.seed, tr, base_acc, new_acc, hmean, loss])
        else:
            w.writerow([ep, args.dataset, args.seed, tr, te, loss])
        
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

def run_clip(args, clip_model, dataset, test_loader):
    clip_model.eval()
    
    if args.exp == "base-to-new":
        acc_test = evaluate_CLIP_base_new(clip_model, test_loader, dataset, args.device)
        acc_base, acc_new, hmean = acc_test[0], acc_test[1], acc_test[2]
        print(f"Zero-shot CLIP's : 🌱 Base acc: {acc_base:.2f} | 🆕 New acc: {acc_new:.2f} | H-Mean: {hmean:.2f}")
    else:
        acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
        print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc_test))
    log_results_txt(args, ep=0, tr=float('nan'), te=acc_test, loss=float('nan'))
        
    return


# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
from run_utils import *
from train import run_clip
from pathlib import Path
import json
import torchvision 
import os, getpass

def main():
    args = get_arguments()
    if args.exp in ['domain-generalization', 'cross-transfer']:
        args.eval_only = True
        if args.weight_file is None:
            fname = (f"dataset_{args.source_dataset}_shots_{args.shots}_backbone_{args.backbone.replace('/','-')}_opt_{args.optimizer.lower()}_seed_{args.seed}.pt")
            args.weight_file = os.path.join(args.save_path, fname)
    args.root_path = resolve_root(args.root_path)
    set_random_seed(args.seed)
    
    user = getpass.getuser()
    clip_model, preprocess = clip.load(args.backbone, args.device)
        
    logit_scale = 100
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    # ---------- BASE–TO–NEW preparation ----------------------------------
    if args.exp == "base-to-new":
        split_path = Path(args.split_dir).expanduser() / f"{args.dataset}_seed_{args.seed}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"❌ Split file not found: {split_path}\n Make sure you generated it with make_b2n_splits.py and passed --split_dir correctly.")
        with open(split_path) as f:
            split = json.load(f)
        base_cls, new_cls = split["base"], split["new"]
        # ---- keep only Base classes for training -------------------------
        def _get_label(sample):
            return sample.label if hasattr(sample, "label") else sample[1]

        if isinstance(dataset.train_x, torchvision.datasets.ImageFolder):
            new_samples = [(p, y) for (p, y) in dataset.train_x.samples if y in base_cls]
            dataset.train_x.samples = new_samples
            dataset.train_x.imgs    = new_samples           
            dataset.train_x.targets = [y for _, y in new_samples]
        else:
            dataset.train_x[:] = [s for s in dataset.train_x if _get_label(s) in base_cls]

        # expose split to evaluator
        dataset.base_cls, dataset.new_cls = base_cls, new_cls
    
    # ---------------------- Preparing Dataloaders  -------------------------
    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
    train_loader = None
    if not args.eval_only:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_tfm = train_transform if args.use_aug else preprocess
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tfm, is_train=True, shuffle=True, num_workers=8)
    run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader, val_loader)
if __name__ == '__main__':
    main()
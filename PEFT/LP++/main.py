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
import os, getpass

def main():
    args = get_arguments()
    args.root_path = resolve_root(args.root_path)
    set_random_seed(args.seed)
    
    user = getpass.getuser()
    clip_model, preprocess = clip.load(args.backbone, args.device)
    logit_scale = 100
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
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
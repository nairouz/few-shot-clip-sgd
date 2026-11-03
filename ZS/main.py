# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import torch
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
from run_utils import *
from train import run_clip
from pathlib import Path
import json
import torchvision 

def main():
    args = get_arguments()
    args.root_path = resolve_root(args.root_path)
    set_random_seed(args.seed)
    clip_model, preprocess = clip.load(args.backbone, args.device)
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, 16, preprocess)
    
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
        
    if args.dataset == 'imagenet':
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
    run_clip(args, clip_model, dataset, test_loader)
    
if __name__ == '__main__':
    main()
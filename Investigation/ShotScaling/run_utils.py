import argparse
import random
import numpy as np
import torch

DATASETS = ("eurosat", "oxford_pets", "dtd", "fgvc", "stanford_cars", "caltech101")

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_arguments():
    p = argparse.ArgumentParser(description="P2 shot-scaling and medium-dataset mechanism study")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--shots", type=int, choices=(1,2,4,8,16), required=True)
    p.add_argument("--seed", type=int, choices=(1,2,3), required=True)
    p.add_argument("--optimizer_variant", choices=("sgd","adam_raw","adam_m0"), required=True)
    p.add_argument("--protocol", choices=("augmented","no_augmentation_bridge"), default="augmented")
    p.add_argument("--root_path", required=True)
    p.add_argument("--result_root", default="results")
    p.add_argument("--clip_cache", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone", default="ViT-B/16", choices=("ViT-B/16",))
    p.add_argument("--batch_size", type=int, default=64, choices=(64,))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu_threads", type=int, default=8)
    p.add_argument("--pool", choices=("Taylor","Bool","local"), default="local")
    p.add_argument("--physical_gpu_id", default="unknown")
    p.add_argument("--diagnostic_batches", type=int, default=64, choices=(64,))
    p.add_argument("--diag_batch_size", type=int, default=16, choices=(16,))
    p.add_argument("--smoke", action="store_true", help="Allow fewer than 30 epochs; never use for production")
    args = p.parse_args()
    if args.epochs != 30 and not args.smoke:
        p.error("production P2 requires exactly 30 epochs")
    if args.protocol == "no_augmentation_bridge" and (args.dataset not in {"eurosat","oxford_pets","dtd"} or args.shots not in {1,4,16} or args.optimizer_variant not in {"sgd","adam_raw"}):
        p.error("invalid bridge cell")
    return args

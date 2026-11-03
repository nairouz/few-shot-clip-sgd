import random
import argparse  
import numpy as np 
import torch
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to run the model on (e.g., 'cuda' or 'cpu')")
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='~/datasets')
    parser.add_argument("--split_dir", default="~/splits", help="Directory containing pre-computed Base-to-New JSON splits")
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Location of trainable parameters
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    # Training arguments
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='which optimizer to use for training.')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--max_epochs", type=int, default=30, help="hard cap on training epochs (keeps early-stop loss check as well)")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--use_aug', action='store_true', help='Enable data augmentation for training images.')
    # Evaluation arguments
    parser.add_argument('--eval_only', default=False, action='store_true', help='evaluate the model after the loading the weights')
    parser.add_argument('--save_path', default='models', help='path to save the model after training, not saved if None')
    # ───────── experiment type & weight handling ──────────
    parser.add_argument('--exp', default='few-shot', choices=['few-shot', 'domain-generalization', 'cross-transfer', 'base-to-new'], help="few-shot = train & test on one dataset  •  domain-generalization = load ImageNet weights then test on ImageNet‑V2/A/R/Sketch  •  cross-transfer = load ImageNet weights then test on the 10 other datasets  •  base-to-new = (stub – will be added later)")
    parser.add_argument('--source_dataset', type=str, default='imagenet', help="dataset used to *train* the weights that will be loaded (relevant for DG / CT)")
    parser.add_argument('--weight_file', type=str, default=None, help="explicit path to a .pt prompt-weight file; if omitted the script builds the default path automatically")
    args = parser.parse_args()
    # If user left the default "models", route to with/without_aug automatically
    if args.save_path is None or args.save_path == 'models':
        args.save_path = 'models_with_aug' if args.use_aug else 'models_without_aug'
    return args


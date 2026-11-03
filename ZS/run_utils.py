# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

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
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Experiment type 
    parser.add_argument('--exp', default='few-shot', choices=['few-shot', 'base-to-new'], help="few-shot = train & test on one dataset  •  domain-generalization = load ImageNet weights then test on ImageNet‑V2/A/R/Sketch  •  cross-transfer = load ImageNet weights then test on the 10 other datasets  •  base-to-new = (stub – will be added later)")
    args = parser.parse_args()
    return args


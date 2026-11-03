# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import torch  
import os, json
from pathlib import Path

def save_clip_weights(clip_model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(clip_model.state_dict(), save_path)
    print(f"CLIP weights saved to {save_path}")
    
def load_clip_weights(clip_model, load_path, strict=True):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File {load_path} does not exist.")
    clip_model.load_state_dict(torch.load(load_path), strict=strict)
    print(f"CLIP weights loaded from {load_path}")
    return clip_model

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0] 
    return acc

def resolve_root(path:str) -> str:
    env_root = os.getenv("DATASETS_ROOT")
    if env_root:
        return str(Path(env_root).expanduser().resolve())
    return str(Path(path).expanduser().resolve())

def read_json(fpath: str):
    fpath = os.path.expanduser(fpath)      
    with open(fpath, "r") as f:
        return json.load(f)

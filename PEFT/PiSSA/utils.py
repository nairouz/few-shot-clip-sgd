import os, json
from pathlib import Path

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

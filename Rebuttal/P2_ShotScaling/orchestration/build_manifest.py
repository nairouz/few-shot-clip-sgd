#!/usr/bin/env python3
import argparse, itertools, json
from pathlib import Path
DATASETS=["eurosat","oxford_pets","dtd","fgvc","stanford_cars","caltech101"]
SHOTS=[1,2,4,8,16]; SEEDS=[1,2,3]; OPTS=["sgd","adam_raw","adam_m0"]

def build():
    bundles=[]
    for d,k,s in itertools.product(DATASETS,SHOTS,SEEDS):
        bundles.append({"bundle_id":f"augmented__{d}__k{k}__s{s}","protocol":"augmented",
            "dataset":d,"shots":k,"seed":s,"optimizers":OPTS})
    for d,k,s in itertools.product(DATASETS[:3],[1,4,16],SEEDS):
        bundles.append({"bundle_id":f"no_augmentation_bridge__{d}__k{k}__s{s}",
            "protocol":"no_augmentation_bridge","dataset":d,"shots":k,"seed":s,
            "optimizers":["sgd","adam_raw"]})
    assert sum(len(x["optimizers"]) for x in bundles)==324
    return bundles
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("orchestration/manifest.json"))
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(build(),indent=2)+"\n")

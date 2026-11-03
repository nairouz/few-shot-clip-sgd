#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from orchestration.build_manifest import build
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args();missing=[]
 for b in build():
  for o in b["optimizers"]:
   r=f"{b['protocol']}__{o}__{b['dataset']}__k{b['shots']}__s{b['seed']}__ViT-B-16"
   if not (a.root/b["protocol"]/r/"DONE.json").is_file():missing.append(r)
 print(json.dumps({"expected":324,"missing":missing},indent=2))

#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def load(root):
    rows=[]
    for summary in root.rglob("summary.json"):
        if not (summary.parent/"DONE.json").is_file(): continue
        s=json.loads(summary.read_text());c=json.loads((summary.parent/"config.json").read_text())
        rows.append({"protocol":c["protocol"],"optimizer":c["optimizer_variant"],"dataset":c["dataset"],
            "shots":c["shots"],"seed":c["seed"],"test_accuracy":s["evaluations"]["test"]["accuracy"],
            "run_dir":str(summary.parent)})
    return rows
def write_table(out,name,rows):
    out.mkdir(parents=True,exist_ok=True)
    (out/f"{name}.json").write_text(json.dumps(rows,indent=2)+"\n")
    if rows:
        with (out/f"{name}.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        headers=list(rows[0]);lines=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]
        lines += ["| "+" | ".join(str(r[h]) for h in headers)+" |" for r in rows]
        (out/f"{name}.md").write_text("\n".join(lines)+"\n")
def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=Path("../results"));p.add_argument("--out",type=Path,default=Path("tables"));a=p.parse_args()
    runs=load(a.results);write_table(a.out,"run_completeness",[{"expected_primary":270,"expected_bridge":54,"completed":len(runs)}])
    groups=defaultdict(list)
    for r in runs:
        if r["protocol"]=="augmented":groups[(r["dataset"],r["shots"],r["optimizer"])].append(r["test_accuracy"])
    accuracy=[{"dataset":k[0],"shots":k[1],"optimizer":k[2],"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else float("nan"),"n":len(v)} for k,v in sorted(groups.items())]
    write_table(a.out,"final_test_accuracy",accuracy)
    cells=defaultdict(dict)
    for r in runs:
        if r["protocol"]=="augmented":cells[(r["dataset"],r["shots"],r["seed"])][r["optimizer"]]=r["test_accuracy"]
    gaps=[]
    for key,v in sorted(cells.items()):
        if {"sgd","adam_raw"}<=v.keys():
            raw=v["sgd"]-v["adam_raw"];m0=v.get("adam_m0",float("nan"))
            closure=100*(m0-v["adam_raw"])/raw if abs(raw)>1e-12 else float("nan")
            gaps.append({"dataset":key[0],"shots":key[1],"seed":key[2],"raw_gap":raw,"m0_gap":v["sgd"]-m0,"m0_closure":closure})
    write_table(a.out,"accuracy_gaps",gaps)
if __name__=="__main__":main()

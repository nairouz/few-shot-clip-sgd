#!/usr/bin/env python3
import argparse,pandas as pd,matplotlib.pyplot as plt
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("csv");p.add_argument("--x",required=True);p.add_argument("--y",default="raw_gap");p.add_argument("--out",required=True);a=p.parse_args();d=pd.read_csv(a.csv)
 fig,ax=plt.subplots();ax.scatter(d[a.x],d[a.y]);ax.set(xlabel=a.x,ylabel=a.y);fig.tight_layout();fig.savefig(a.out,dpi=300)

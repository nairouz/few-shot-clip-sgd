#!/usr/bin/env python3
"""Compute Spearman correlations after aggregation; q checkpoints are never treated as samples."""
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("csv",type=Path);a=p.parse_args();d=pd.read_csv(a.csv)
 print(spearmanr(d["mechanism_metric"],d["raw_gap"],nan_policy="omit"))

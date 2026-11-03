#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import argparse,pandas as pd
from plot_common import shot_plot
def main():
 p=argparse.ArgumentParser();p.add_argument("csv");p.add_argument("--metric",required=True);p.add_argument("--out",required=True);a=p.parse_args();shot_plot(pd.read_csv(a.csv),a.metric,a.out,"P2 shot scaling")
if __name__=="__main__":main()

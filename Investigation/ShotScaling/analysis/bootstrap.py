#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import numpy as np
def cluster_bootstrap(values,clusters,statistic,resamples=10000,seed=2026):
 rng=np.random.default_rng(seed);unique=np.unique(clusters);draws=[]
 for _ in range(resamples):
  chosen=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(clusters==x) for x in chosen]);draws.append(statistic(values[idx]))
 return np.quantile(draws,[.025,.975])

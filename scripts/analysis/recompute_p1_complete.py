# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Reproduce complete P1 matched-update tables from verified per-run exports."""
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'results/inputs'
OUTPUT = ROOT / 'results/generated'
ROWS = OUTPUT / 'manuscript_rows'
if not INPUT.is_dir():
    raise SystemExit(f"Local result exports are required in {INPUT}. Experimental results are not bundled; input requirements are documented in docs/ANALYSIS.md.")
OUTPUT.mkdir(parents=True, exist_ok=True)
ROWS.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(INPUT / 'p1_complete_per_seed.csv', float_precision='round_trip')
pairing = pd.read_csv(INPUT / 'p1_complete_pairing.csv')
keys = ['dataset', 'shots', 'seed']
datasets = sorted(runs.dataset.unique())
assert len(datasets) == 10 and len(runs) == 300
assert set(runs.variant) == {'P1_Adam_M0', 'P1_Adam_Mt_Global'}
assert (runs.final_epoch == 30).all()
assert not runs.duplicated(keys + ['variant']).any()
expected = set(itertools.product(datasets, [1, 2, 4, 8, 16], [1, 2, 3]))
for _, group in runs.groupby('variant'):
    assert set(zip(group.dataset, group.shots, group.seed)) == expected
assert len(pairing) == 150 and not pairing.duplicated(keys).any()
for column in ['support_manifest_exact_equality', 'root_normalized_support_equality',
               'common_config_fields_equal', 'same_git_sha']:
    assert pairing[column].all()

paired = runs.pivot(index=keys, columns='variant', values='final_test_accuracy')
paired = paired.rename(columns={'P1_Adam_M0': 'Adam_M0', 'P1_Adam_Mt_Global': 'Adam_Mt'})
paired['Mt_minus_M0'] = paired.Adam_Mt - paired.Adam_M0
paired.to_csv(OUTPUT / 'p1_complete_paired_accuracy.csv')
cells = paired.groupby(['dataset', 'shots']).agg(['mean', 'std'])
cells.columns = ['_'.join(c) for c in cells.columns]
cells.to_csv(OUTPUT / 'p1_complete_dataset_shot_summary.csv')
shots = paired.groupby('shots').mean()
shots['n_pairs'] = paired.groupby('shots').size()
shots.to_csv(OUTPUT / 'p1_complete_shot_summary.csv')

names = {'caltech101': 'Caltech101', 'dtd': 'DTD', 'eurosat': 'EuroSAT',
         'fgvc': 'FGVC-Aircraft', 'food101': 'Food101', 'oxford_flowers': 'Flowers102',
         'oxford_pets': 'Oxford-Pets', 'stanford_cars': 'Stanford-Cars',
         'sun397': 'SUN397', 'ucf101': 'UCF101'}
tex = []
for (dataset, k), row in cells.iterrows():
    tex.append(names[dataset] + f' & {k} & ${row.Adam_M0_mean:.2f}\\pm{row.Adam_M0_std:.2f}$ & ${row.Adam_Mt_mean:.2f}\\pm{row.Adam_Mt_std:.2f}$' + r' \\')
(ROWS / 'p1_complete_accuracy_rows.tex').write_text('\n'.join(tex) + '\n')
tex = []
for k, row in shots.iterrows():
    tex.append(f'{k} & {row.Adam_M0:.2f} & {row.Adam_Mt:.2f} & ${row.Mt_minus_M0:+.2f}$' + r' \\')
mean = paired.mean()
tex.extend([r'\midrule', f'Mean & {mean.Adam_M0:.2f} & {mean.Adam_Mt:.2f} & ${mean.Mt_minus_M0:+.2f}$' + r' \\'])
(ROWS / 'p1_complete_by_shot_rows.tex').write_text('\n'.join(tex) + '\n')

# Whole-dataset resampling retains all paired shot counts and seeds.
bootstrap_input = pd.read_csv(OUTPUT / 'p1_complete_paired_accuracy.csv')
cluster_means = bootstrap_input.groupby('dataset').Mt_minus_M0.mean().to_numpy()
rng = np.random.default_rng(20261103)
draws = cluster_means[rng.integers(0, len(cluster_means), size=(20000, len(cluster_means)))].mean(axis=1)
low, high = np.percentile(draws, [2.5, 97.5])
pd.DataFrame([{
    'estimand': 'Adam-Mt minus Adam-M0 fixed-epoch30 accuracy',
    'estimate': cluster_means.mean(), 'ci95_low': low, 'ci95_high': high,
    'resamples': 20000, 'cluster': 'dataset', 'n_datasets': 10,
    'n_paired_runs': 150, 'bootstrap_seed': 20261103,
}]).to_csv(OUTPUT / 'p1_complete_matching_bootstrap.csv', index=False)
print(f'Complete P1: 150 pairs; Adam-M0 {mean.Adam_M0:.6f}; Adam-Mt {mean.Adam_Mt:.6f}.')

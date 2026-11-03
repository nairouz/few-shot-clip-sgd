#!/usr/bin/env python3
"""
Create one CSV per (backbone, optimiser) that summarises *few-shot all-to-all*
runs.

• Reads only the FIRST data row in every result file (defensive against
accidental reruns that appended extra lines).
• Accepts optional --datasets / --shot_values overrides.
• Writes to ./summary_results_* with the pattern
results_few_shot_summary_backbone_<backbone>_opt_<optimiser>.csv
where slashes in <backbone> are replaced by hyphens.
"""

import argparse, csv, glob, os, re, statistics, sys
from typing import List

# ───────── choose aug/no-aug bucket ─────────
USE_AUG = True  # set True for augmented runs, False for non-augmented
AUG_SUFFIX = "with_aug" if USE_AUG else "without_aug"
DEFAULT_RESULTS_DIR = f"./results_{AUG_SUFFIX}"
DEFAULT_OUTPUT_DIR  = f"./summary_results_{AUG_SUFFIX}"

# ─────────── defaults that match your sweeps ───────────
DATASETS = [
    "oxford_pets", "eurosat", "ucf101", "sun397", "caltech101",
    "dtd", "fgvc", "food101", "oxford_flowers", "stanford_cars",
    "imagenet"
]
SHOT_VALUES = [16, 8, 4, 2, 1]
SEEDS = [1, 2, 3]

# ─────────── CLI helpers ────────────
def csv_list(arg: str, cast=str) -> List:
    return [] if not arg else [cast(x) for x in arg.split(",")]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--backbone", required=True, help='e.g. "ViT-B/16" or "ViT-B-16"')
    p.add_argument("--optimizer", required=True, help='adam | sgd | … (lower/upper-case accepted)')
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--datasets",    default="", help="Comma-separated list to override the built-ins")
    p.add_argument("--shot_values", default="", help="Comma-separated ints to override the built-ins")
    return p.parse_args()

def canonical_backbone(name: str) -> str:
    return name.replace("/", "-")

# ─────────── column finder & value extractor ───────────
def _find_test_acc_idx(header):
    norm_cols = ["".join(c.lower().split()) for c in header]
    for i, n in enumerate(norm_cols):
        if "test" in n and "acc" in n:      # matches 'testaccuracy' etc.
            return i
    raise ValueError(f"No test-accuracy column in header: {header}")

def extract_first_accuracy(txt_file: str) -> float:
    """
    Return the *first* data-row accuracy value, ignoring any later rows that
    may have been appended by mistake.
    """
    with open(txt_file, newline="") as f:
        reader = csv.reader(f)
        header = next(row for row in reader if row)   # skip blank lines
        data   = next((r for r in reader if r), None)
        if data is None:
            raise ValueError(f"No data row found in {txt_file}")
    idx = _find_test_acc_idx(header)
    return float(data[idx])

# ─────────── aggregation ───────────
def main() -> None:
    a          = parse_args()
    datasets   = csv_list(a.datasets)            or DATASETS
    shot_vals  = csv_list(a.shot_values, int)    or SHOT_VALUES
    bb_safe    = canonical_backbone(a.backbone)
    opt_safe   = a.optimizer.lower()

    pattern = os.path.join(
        a.results_dir,
        f"final_results_dataset_*_shots_*_backbone_{bb_safe}_opt_{opt_safe}_seed_*.txt"
    )
    regex = re.compile(
        rf"final_results_dataset_(?P<ds>.+?)_shots_(?P<shot>\d+)_"
        rf"backbone_{re.escape(bb_safe)}_opt_{opt_safe}_seed_(?P<seed>\d+)\.txt$"
    )

    # index files by (dataset, shot, seed)
    file_map = {}
    for fp in glob.glob(pattern):
        m = regex.search(os.path.basename(fp))
        if m:
            key = (m.group("ds"), int(m.group("shot")), int(m.group("seed")))
            file_map[key] = fp

    summary = {}
    for ds in datasets:
        for shot in shot_vals:
            accs = []
            for seed in SEEDS:
                fp = file_map.get((ds, shot, seed))
                if fp:
                    try:
                        accs.append(extract_first_accuracy(fp))
                    except Exception as e:
                        print(f"⚠️  {e}", file=sys.stderr)
                else:
                    print(f"⚠️  Missing file for {ds}  •  {shot}-shot  •  seed={seed}", file=sys.stderr)
            if accs:
                mu  = statistics.mean(accs)
                sd  = statistics.pstdev(accs) if len(accs) > 1 else 0.0
                summary[(ds, shot)] = (mu, sd)

    os.makedirs(a.output_dir, exist_ok=True)
    out_csv = f"results_few_shot_summary_backbone_{bb_safe}_opt_{opt_safe}.csv"
    out_path = os.path.join(a.output_dir, out_csv)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Shots", "Mean (%)", "Std (%)"])
        for ds in datasets:
            for shot in shot_vals:
                mu, sd = summary.get((ds, shot), ("NA", "NA"))
                w.writerow([
                    ds, shot,
                    f"{mu:.2f}" if isinstance(mu, float) else mu,
                    f"{sd:.2f}" if isinstance(sd, float) else sd
                ])

    print(f"✅  Summary written to {out_path}")

if __name__ == "__main__":
    main()

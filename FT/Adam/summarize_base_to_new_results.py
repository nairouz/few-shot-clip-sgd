#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import argparse, csv, glob, os, re, statistics, sys
from typing import List, Dict, Tuple

# ───────── choose aug/no-aug bucket ─────────
USE_AUG = True  # set True for augmented runs, False for non-augmented
AUG_SUFFIX = "with_aug" if USE_AUG else "without_aug"
DEFAULT_RESULTS_DIR = f"./results_{AUG_SUFFIX}"
DEFAULT_OUTPUT_DIR  = f"./summary_results_{AUG_SUFFIX}"

# ─────────── defaults matching your sweep ────────────
DATASETS    = [
    "oxford_pets", "eurosat", "ucf101", "sun397", "caltech101", "dtd",
    "fgvc", "food101", "oxford_flowers", "stanford_cars", "imagenet"
]
SHOT_VALUES = [16, 8, 4, 2, 1]
SEEDS       = [1, 2, 3]

# ─────────── CLI helpers ────────────
def csv_list(arg: str, cast=str) -> List:
    return [] if not arg else [cast(x) for x in arg.split(",")]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--backbone", required=True, help='e.g. "ViT-B/16"')
    p.add_argument("--optimizer", default="", help="omit to assume filenames have no _opt_<opt> segment")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--datasets", default="", help="comma list to override default DATASETS order")
    p.add_argument("--shot_values", default="", help="comma list to override default SHOT_VALUES order")
    return p.parse_args()

def canonical_backbone(name: str) -> str:
    return name.replace("/", "-")

# ─────────── robust column finder ────────────
def _find_idx(header, key_words):
    """
    Return idx of column whose canonicalised name contains *all* keywords.
    key_words is a list like ['base', 'acc'] or ['h', 'mean'].
    """
    for i, col in enumerate(header):
        norm = col.lower().replace(" ", "").replace("_", "")
        if all(k in norm for k in key_words):
            return i
    raise ValueError(f"Missing column matching {key_words} in header {header}")

def extract_metrics(txt_file: str) -> Tuple[float, float, float]:
    """Return (base_acc, new_acc, h_mean) from the *last* data row."""
    with open(txt_file, newline="") as f:
        reader = csv.reader(f)
        header = next(row for row in reader if row)   # skip blank lines
        rows   = [r for r in reader if r]
    idx_b = _find_idx(header, ["base", "acc"])
    idx_n = _find_idx(header, ["new",  "acc"])
    idx_h = _find_idx(header, ["h",    "mean"])
    try:
        last = rows[-1]
        return float(last[idx_b]), float(last[idx_n]), float(last[idx_h])
    except (ValueError, IndexError):
        raise ValueError(f"Malformed metrics in {txt_file}")

# ─────────── main ────────────
def main():
    args        = parse_args()
    datasets    = csv_list(args.datasets)         or DATASETS
    shot_vals   = csv_list(args.shot_values, int) or SHOT_VALUES
    bb_safe     = canonical_backbone(args.backbone)
    opt_given   = bool(args.optimizer)
    opt_safe    = args.optimizer.lower()

    # ---- filename patterns ----
    if opt_given:
        pattern = os.path.join(
            args.results_dir,
            f"b2n_results_dataset_*_shots_*_backbone_{bb_safe}_opt_{opt_safe}_seed_*.txt"
        )
        regex = re.compile(
            rf"b2n_results_dataset_(?P<ds>.+?)_shots_(?P<shot>\d+)_"
            rf"backbone_{re.escape(bb_safe)}_opt_{opt_safe}_seed_(?P<seed>\d+)\.txt$"
        )
    else:
        pattern = os.path.join(
            args.results_dir,
            f"b2n_results_dataset_*_shots_*_backbone_{bb_safe}_seed_*.txt"
        )
        regex = re.compile(
            rf"b2n_results_dataset_(?P<ds>.+?)_shots_(?P<shot>\d+)_"
            rf"backbone_{re.escape(bb_safe)}_seed_(?P<seed>\d+)\.txt$"
        )

    # ---- map (dataset, shot, seed) → file path ----
    file_map: Dict[Tuple[str, int, int], str] = {}
    for fp in glob.glob(pattern):
        m = regex.search(os.path.basename(fp))
        if m:
            key = (m.group("ds"), int(m.group("shot")), int(m.group("seed")))
            file_map[key] = fp

    # ---- aggregate ----
    summary: Dict[Tuple[str, int], Tuple[Tuple[float,float], ...]] = {}
    for ds in datasets:
        for shot in shot_vals:
            base_vals, new_vals, h_vals = [], [], []
            for seed in SEEDS:
                key = (ds, shot, seed)
                if key in file_map:
                    try:
                        b, n, h = extract_metrics(file_map[key])
                        base_vals.append(b)
                        new_vals.append(n)
                        h_vals.append(h)
                    except Exception as e:
                        print(f"⚠️  {e}", file=sys.stderr)
                else:
                    print(f"⚠️  Missing file for {ds} • {shot}-shot • seed={seed}", file=sys.stderr)
            if base_vals:
                stats = (
                    (statistics.mean(base_vals), statistics.pstdev(base_vals) if len(base_vals) > 1 else 0.0),
                    (statistics.mean(new_vals), statistics.pstdev(new_vals)  if len(new_vals)  > 1 else 0.0),
                    (statistics.mean(h_vals), statistics.pstdev(h_vals)    if len(h_vals)    > 1 else 0.0),
                )
                summary[(ds, shot)] = stats

    # ---- write CSV ----
    os.makedirs(args.output_dir, exist_ok=True)
    csv_name = (f"results_b2n_summary_backbone_{bb_safe}"
                f"{'_opt_' + opt_safe if opt_given else ''}.csv")
    csv_path = os.path.join(args.output_dir, csv_name)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "Shots",
            "BaseAccMean (%)", "BaseAccStd (%)",
            "NewAccMean (%)",  "NewAccStd (%)",
            "H-MeanMean (%)",  "H-MeanStd (%)"
        ])
        for ds in datasets:
            for shot in shot_vals:
                if (ds, shot) in summary:
                    (μb, σb), (μn, σn), (μh, σh) = summary[(ds, shot)]
                    w.writerow([
                        ds, shot,
                        f"{μb:.2f}", f"{σb:.2f}",
                        f"{μn:.2f}", f"{σn:.2f}",
                        f"{μh:.2f}", f"{σh:.2f}"
                    ])
                else:
                    w.writerow([ds, shot] + ["NA"] * 6)

    print(f"✅  Base-to-New summary written to {csv_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-ready directional-curvature ratio plot for Experiment H.

The script reads all Experiment H encoder-alignment CSV files in a log directory
and plots

    log10 Q_kappa^{(e)}(t)

where Q_kappa compares the full-train and support directional curvatures along
the normalized full-train gradient direction.

Inputs
------
  - <run_name>_config.json
  - <run_name>_encoder_alignment.csv

Output
------
  - expH_directional_curvature_ratio.(png|pdf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_SIZE = 12
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": BASE_SIZE + 1,
    "axes.titlesize": BASE_SIZE + 3,
    "axes.labelsize": BASE_SIZE + 2,
    "xtick.labelsize": BASE_SIZE + 1,
    "ytick.labelsize": BASE_SIZE + 1,
    "legend.fontsize": BASE_SIZE + 1,
    "legend.title_fontsize": BASE_SIZE + 1,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.6,
    "lines.markersize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_VISION = "#1f77b4"
COL_TEXT = "#ff7f0e"
MARK_VISION = "o"
MARK_TEXT = "s"
DATASET_TITLE = {
    "eurosat": "EuroSAT",
    "oxford_pets": "Oxford-Pets",
    "dtd": "DTD",
}
DATASET_ORDER = ["eurosat", "oxford_pets", "dtd"]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--datasets",
        type=str,
        default="eurosat,oxford_pets,dtd",
        help="Comma-separated dataset order. Default: eurosat,oxford_pets,dtd.",
    )
    return ap.parse_args()


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.tick_params(axis="both", pad=4)


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fig.text(0.5, 0.035, xlabel, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.018, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])


def _load_runs(logdir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for cfg_path in sorted(logdir.glob("experimentH_*_config.json")):
        run_name = cfg_path.name[:-len("_config.json")]
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        dataset = str(cfg.get("dataset", ""))
        csv_path = logdir / f"{run_name}_encoder_alignment.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        df["dataset"] = dataset
        out[dataset] = df
    return out


def _compute_log_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = np.abs(df["curv_full_on_gfull"].to_numpy(dtype=float)) + EPS
    den = np.abs(df["curv_few_on_gfull"].to_numpy(dtype=float)) + EPS
    df["log10_Q_kappa"] = np.log10(num / den)
    df["Q_kappa"] = num / den
    return df


def plot_directional_curvature_ratio(runs: Dict[str, pd.DataFrame], outdir: Path, dataset_order: List[str]) -> None:
    available = [d for d in dataset_order if d in runs]
    if not available:
        raise RuntimeError("No matching Experiment H runs found in the provided log directory.")

    fig, axes = plt.subplots(1, len(available), figsize=(5.7 * len(available), 4.1), sharex=True, sharey=True)
    if len(available) == 1:
        axes = [axes]

    styles = {
        "vision": dict(color=COL_VISION, marker=MARK_VISION, linestyle="-", label="Vision"),
        "text": dict(color=COL_TEXT, marker=MARK_TEXT, linestyle="--", label="Text"),
    }

    ymin, ymax = 0.0, 0.0
    for ax, dataset in zip(axes, available):
        df = _compute_log_ratio(runs[dataset])
        for encoder in ["vision", "text"]:
            sub = df[df["encoder"] == encoder].sort_values("epoch")
            if sub.empty:
                continue
            sty = styles[encoder]
            y = sub["log10_Q_kappa"].to_numpy(dtype=float)
            ymin = min(ymin, float(np.nanmin(y)))
            ymax = max(ymax, float(np.nanmax(y)))
            ax.plot(sub["epoch"], y, **sty)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_title(DATASET_TITLE.get(dataset, dataset))
        ax.set_xlabel("")
        ax.set_xlim(-0.5, 30.5)
        _style_axis(ax)

    # Add a small margin. Keep the y-axis shared to make cross-dataset comparison direct.
    margin = 0.08 * max(1.0, ymax - ymin)
    for ax in axes:
        ax.set_ylim(ymin - margin, ymax + margin)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.3,
    )
    plt.tight_layout(rect=[0.045, 0.075, 1.0, 0.90])
    _add_global_xy_labels(
        fig,
        xlabel=r"Checkpoint epoch $t$",
        ylabel=r"$\log_{10} Q_{\kappa}^{(e)}(t)$",
    )
    _save_png_pdf(fig, outdir / "expH_directional_curvature_ratio.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset_order = [x.strip() for x in args.datasets.split(",") if x.strip()]
    runs = _load_runs(args.logdir)
    plot_directional_curvature_ratio(runs, args.outdir, dataset_order)
    print(f"Saved Experiment H directional-curvature ratio plot to: {args.outdir}")


if __name__ == "__main__":
    main()

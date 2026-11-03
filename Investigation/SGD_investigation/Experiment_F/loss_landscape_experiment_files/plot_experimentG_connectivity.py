#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Experiment G — Basin connectivity across SGD seeds.

Required inputs (from measure_experimentG_connectivity.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_pairwise_interpolation.csv
  - <run_name>_pairwise_summary.csv

Figures emitted:
  A) expG_seed_trajectories.(png|pdf)
  B) expG_pairwise_interpolation_losses.(png|pdf)
  C) expG_pairwise_barriers.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Style (matched to Experiment 2 plotting style)
# ============================================================================
BASE_SIZE = 13
mpl.rcParams.update({
    "font.size": BASE_SIZE + 1,
    "axes.titlesize": BASE_SIZE + 1,
    "axes.labelsize": BASE_SIZE + 1,
    "xtick.labelsize": BASE_SIZE + 1,
    "ytick.labelsize": BASE_SIZE + 1,
    "legend.fontsize": BASE_SIZE + 1,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "font.weight": "semibold",
    "axes.labelweight": "semibold",
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.4,
    "lines.markersize": 6,
})

PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    pair_interp = pd.read_csv(logdir / f"{run_name}_pairwise_interpolation.csv")
    pair_summary = pd.read_csv(logdir / f"{run_name}_pairwise_summary.csv")
    return cfg, traj, pair_interp, pair_summary


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fs = mpl.rcParams.get("axes.labelsize", BASE_SIZE + 1)
    fw = mpl.rcParams.get("axes.labelweight", "semibold")
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=fs, fontweight=fw)
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=fs, fontweight=fw)


def _style_axis(ax: plt.Axes, *, which: str = "major") -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0, which=which)


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def plot_A_seed_trajectories(traj: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("few_loss", "Support loss", "few_loss"),
        ("full_loss", "Full-train loss", "full_loss"),
        ("test_loss", "Test loss", "test_loss"),
    ]
    seeds = sorted(traj["seed"].unique())
    for ax, (col, title, _) in zip(axes, panels):
        for idx, seed in enumerate(seeds):
            sub = traj[traj["seed"] == seed].sort_values("epoch")
            ax.plot(sub["epoch"], sub[col], marker="o", color=SEED_COLORS[idx % len(SEED_COLORS)], label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=min(len(seeds), 3))
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch t", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expG_seed_trajectories.png")
    plt.close(fig)


def plot_B_pairwise_interpolation_losses(pair_interp: pd.DataFrame, pair_summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("few_pair_loss", "Pair-support loss", "few_pair"),
        ("full_loss", "Full-train loss", "full"),
        ("test_loss", "Test loss", "test"),
    ]
    pair_labels = list(pair_summary.sort_values(["seed_a", "seed_b"])["pair_label"])
    color_map = {label: PAIR_COLORS[i % len(PAIR_COLORS)] for i, label in enumerate(pair_labels)}

    for ax, (col, title, prefix) in zip(axes, panels):
        for label in pair_labels:
            sub = pair_interp[pair_interp["pair_label"] == label].sort_values("alpha")
            barrier = float(pair_summary.loc[pair_summary["pair_label"] == label, f"{prefix}_barrier"].iloc[0])
            ax.plot(sub["alpha"], sub[col], color=color_map[label], label=f"{label} (b={barrier:.3f})")
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Interpolation coefficient $\alpha$", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expG_pairwise_interpolation_losses.png")
    plt.close(fig)


def plot_C_pairwise_barriers(pair_summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("few_pair_barrier", "Pair-support barrier"),
        ("full_barrier", "Full-train barrier"),
        ("test_barrier", "Test barrier"),
    ]
    pair_summary = pair_summary.sort_values(["seed_a", "seed_b"]).copy()
    labels = pair_summary["pair_label"].tolist()
    x = np.arange(len(labels))
    colors = [PAIR_COLORS[i % len(PAIR_COLORS)] for i in range(len(labels))]

    for ax, (col, title) in zip(axes, panels):
        vals = pair_summary[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("")
        _style_axis(ax)

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Seed pair", ylabel="Barrier above linear interpolation")
    _save_png_pdf(fig, outdir / "expG_pairwise_barriers.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, pair_interp, pair_summary = _load_bundle(args.logdir, args.run_name)
    plot_A_seed_trajectories(traj, args.outdir)
    plot_B_pairwise_interpolation_losses(pair_interp, pair_summary, args.outdir)
    plot_C_pairwise_barriers(pair_summary, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

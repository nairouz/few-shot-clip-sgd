#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment A — 1D interpolation from initialization to final SGD solution.

Required inputs (from measure_experimentA_interpolation.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_checkpoint_projection.csv
  - <run_name>_interpolation.csv
  - <run_name>_summary.json

Figures emitted:
  A) expA_interpolation_losses.(png|pdf)
  B) expA_checkpoint_projection.(png|pdf)
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

COL_MAIN = "#1f77b4"
COL_MILE = "#7f7f7f"
COL_ORTH = "#d62728"
COL_END = "#2ca02c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    proj = pd.read_csv(logdir / f"{run_name}_checkpoint_projection.csv")
    interp = pd.read_csv(logdir / f"{run_name}_interpolation.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, proj, interp, summary


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


def plot_A_interpolation_losses(interp: pd.DataFrame, proj: pd.DataFrame, summary: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("few_loss", "Support loss", "few_loss_curve"),
        ("full_loss", "Full-train loss", "full_loss_curve"),
        ("test_loss", "Test loss", "test_loss_curve"),
    ]

    for idx, (ax, (col, title, key)) in enumerate(zip(axes, panels)):
        ax.plot(interp["alpha"], interp[col], color=COL_MAIN)
        ax.scatter([interp["alpha"].iloc[0], interp["alpha"].iloc[-1]], [interp[col].iloc[0], interp[col].iloc[-1]], color=COL_END, zorder=3)
        for _, row in proj.sort_values("epoch").iterrows():
            x = float(row["alpha_proj_clipped"])
            ax.axvline(x, color=COL_MILE, linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
        # annotate checkpoint epochs only on the first panel to keep the layout clean
        if idx == 0:
            y0, y1 = ax.get_ylim()
            dy = y1 - y0
            base_y = y1 - 0.06 * dy
            for j, (_, row) in enumerate(proj.sort_values("epoch").iterrows()):
                x = float(row["alpha_proj_clipped"])
                y = base_y - 0.06 * dy * (j % 2)
                ax.text(x, y, f"t={int(row['epoch'])}", rotation=90, ha="center", va="top", color=COL_MILE, fontsize=BASE_SIZE - 2)

        info = summary[key]
        ax.text(
            0.98,
            0.97,
            f"min@α={info['alpha_min']:.2f}\nbarrier={info['barrier']:.3f}\nwidth₅%={info['width_5pct']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=BASE_SIZE - 1,
        )

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Interpolation coefficient $\alpha$", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expA_interpolation_losses.png")
    plt.close(fig)


def plot_B_checkpoint_projection(proj: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True)

    axes[0].plot(proj["epoch"], proj["alpha_proj"], marker="o", color=COL_MAIN)
    axes[0].axhline(0.0, color=COL_MILE, linestyle=":", linewidth=1.0)
    axes[0].axhline(1.0, color=COL_MILE, linestyle=":", linewidth=1.0)
    axes[0].set_title(r"Checkpoint projection on $[\Theta_0,\Theta_{final}]$")
    axes[0].set_xlabel("")
    _style_axis(axes[0])

    axes[1].plot(proj["epoch"], proj["orth_rel"], marker="o", color=COL_ORTH)
    axes[1].set_yscale("log")
    axes[1].set_title("Relative orthogonal deviation")
    axes[1].set_xlabel("")
    _style_axis(axes[1], which="both")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch t", ylabel="Value")
    _save_png_pdf(fig, outdir / "expA_checkpoint_projection.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, proj, interp, summary = _load_bundle(args.logdir, args.run_name)
    plot_A_interpolation_losses(interp, proj, summary, args.outdir)
    plot_B_checkpoint_projection(proj, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

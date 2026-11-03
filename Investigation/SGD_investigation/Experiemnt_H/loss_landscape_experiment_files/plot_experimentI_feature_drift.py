#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment I — Feature-drift landscape along an SGD trajectory.

Required inputs (from measure_experimentI_feature_drift.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_block_drift.csv
  - <run_name>_summary.json

Figures emitted:
  A) expI_losses_and_gaps.(png|pdf)
  B) expI_feature_drift.(png|pdf)
  C) expI_block_parameter_drift.(png|pdf)
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
# Style (matched to the uploaded Experiment-2-style plotting code)
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

COL_FEW = "#d62728"   # red
COL_FULL = "#1f77b4"  # blue
COL_TEST = "#2ca02c"  # green
COL_ALL = "#1f77b4"
COL_VISION = "#2ca02c"
COL_TEXT = "#d62728"
COL_TEXTDRIFT = "#000000"
COL_MILE = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    blocks = pd.read_csv(logdir / f"{run_name}_block_drift.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, blocks, summary


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


def plot_A_losses_and_gaps(traj: pd.DataFrame, summary: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)

    # Losses
    axes[0].plot(traj["epoch"], traj["few_loss"], marker="o", color=COL_FEW, label="Support")
    axes[0].plot(traj["epoch"], traj["full_loss"], marker="o", color=COL_FULL, label="Full-train")
    axes[0].plot(traj["epoch"], traj["test_loss"], marker="o", color=COL_TEST, label="Test")
    axes[0].set_title("Cross-entropy losses")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])
    best_full = summary["best_full_loss"]
    axes[0].text(
        0.98,
        0.97,
        f"best full @ t={best_full['epoch']}\nfull={best_full['full_loss']:.3f}\n"
        f"test={best_full['test_acc']:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=BASE_SIZE - 1,
    )

    # Accuracies
    axes[1].plot(traj["epoch"], traj["few_acc"], marker="o", color=COL_FEW, label="Support")
    axes[1].plot(traj["epoch"], traj["full_acc"], marker="o", color=COL_FULL, label="Full-train")
    axes[1].plot(traj["epoch"], traj["test_acc"], marker="o", color=COL_TEST, label="Test")
    axes[1].set_title("Accuracies")
    _style_axis(axes[1])
    best_test = summary["best_test_acc"]
    axes[1].text(
        0.98,
        0.97,
        f"best test @ t={best_test['epoch']}\nacc={best_test['test_acc']:.2f}\n"
        f"drift={best_test['param_drift_all']:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=BASE_SIZE - 1,
    )

    # Support-fitting gap
    axes[2].plot(traj["epoch"], traj["gap_full_minus_few_loss"], marker="o", color=COL_FULL, label=r"$L_{full}-L_{few}$")
    axes[2].plot(traj["epoch"], traj["gap_test_minus_few_loss"], marker="o", color=COL_TEST, label=r"$L_{test}-L_{few}$")
    axes[2].set_title("Support-fitting gap")
    axes[2].legend(frameon=False)
    _style_axis(axes[2])

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch t", ylabel="Value")
    _save_png_pdf(fig, outdir / "expI_losses_and_gaps.png")
    plt.close(fig)


def plot_B_feature_drift(traj: pd.DataFrame, summary: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)

    # Parameter drift
    axes[0].plot(traj["epoch"], traj["param_drift_all"], marker="o", color=COL_ALL, label="All trainable")
    axes[0].plot(traj["epoch"], traj["param_drift_vision"], marker="o", color=COL_VISION, label="Vision")
    axes[0].plot(traj["epoch"], traj["param_drift_text"], marker="o", color=COL_TEXT, label="Text")
    axes[0].set_title("Relative parameter drift")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])
    axes[0].text(
        0.98,
        0.97,
        f"corr(drift,full loss)={summary['corr_param_drift_all_vs_full_loss']:.3f}\n"
        f"corr(drift,test acc)={summary['corr_param_drift_all_vs_test_acc']:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=BASE_SIZE - 1,
    )

    # Image feature drift
    axes[1].plot(traj["epoch"], traj["image_drift_few"], marker="o", color=COL_FEW, label="Support")
    axes[1].plot(traj["epoch"], traj["image_drift_full"], marker="o", color=COL_FULL, label="Full-train")
    axes[1].plot(traj["epoch"], traj["image_drift_test"], marker="o", color=COL_TEST, label="Test")
    axes[1].set_title("Image-embedding drift")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    axes[1].text(
        0.98,
        0.97,
        f"corr(img full drift,full loss)={summary['corr_image_drift_full_vs_full_loss']:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=BASE_SIZE - 1,
    )

    # Text / margin drift
    axes[2].plot(traj["epoch"], traj["text_drift"], marker="o", color=COL_TEXTDRIFT, label="Text drift")
    axes[2].plot(traj["epoch"], traj["margin_drift_few"], marker="o", linestyle="--", color=COL_FEW, label="Support margin drift")
    axes[2].plot(traj["epoch"], traj["margin_drift_full"], marker="o", linestyle="--", color=COL_FULL, label="Full margin drift")
    axes[2].plot(traj["epoch"], traj["margin_drift_test"], marker="o", linestyle="--", color=COL_TEST, label="Test margin drift")
    axes[2].set_title("Text / class-margin drift")
    axes[2].legend(frameon=False)
    _style_axis(axes[2])
    axes[2].text(
        0.98,
        0.97,
        f"corr(text drift,full loss)={summary['corr_text_drift_vs_full_loss']:.3f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=BASE_SIZE - 1,
    )

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch t", ylabel="Drift / value")
    _save_png_pdf(fig, outdir / "expI_feature_drift.png")
    plt.close(fig)


def plot_C_block_parameter_drift(blocks: pd.DataFrame, cfg: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharey=False)
    epochs = sorted(blocks["epoch"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {ep: cmap(i / max(len(epochs) - 1, 1)) for i, ep in enumerate(epochs)}

    for ax, enc in zip(axes, ["vision", "text"]):
        sub = blocks[blocks["encoder"] == enc].copy()
        for ep in epochs:
            ss = sub[sub["epoch"] == ep].sort_values("block")
            ax.plot(ss["block"], ss["rel_param_drift"], marker="o", color=colors[ep], label=f"t={int(ep)}")
        ax.set_title(f"{enc.capitalize()} block drift")
        ax.set_xlabel("")
        ax.set_xticks(sorted(sub["block"].unique()))
        _style_axis(ax)

    axes[1].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel="Relative parameter drift")
    _save_png_pdf(fig, outdir / "expI_block_parameter_drift.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, blocks, summary = _load_bundle(args.logdir, args.run_name)
    plot_A_losses_and_gaps(traj, summary, args.outdir)
    plot_B_feature_drift(traj, summary, args.outdir)
    plot_C_block_parameter_drift(blocks, cfg, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

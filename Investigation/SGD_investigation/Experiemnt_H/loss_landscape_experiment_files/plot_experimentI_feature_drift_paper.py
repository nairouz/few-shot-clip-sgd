#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paper-ready plots for Experiment I — low-drift SGD trajectories.

For each run, emits three dataset-specific figures:
  - expI_<dataset>_losses_gaps.(png|pdf)
  - expI_<dataset>_feature_drift.(png|pdf)
  - expI_<dataset>_block_drift.(png|pdf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    "lines.markersize": 6.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_FEW = "#d62728"
COL_FULL = "#1f77b4"
COL_TEST = "#2ca02c"
COL_ALL = "#111111"
COL_VISION = "#1f77b4"
COL_TEXT = "#ff7f0e"
EPOCH_CMAP = plt.get_cmap("viridis")
DATASET_TITLE = {
    "eurosat": "EuroSAT",
    "oxford_pets": "Oxford-Pets",
    "dtd": "DTD",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    return ap.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    block = pd.read_csv(logdir / f"{run_name}_block_drift.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, block, summary


def _style_axis(ax: plt.Axes, *, yscale: str = "linear") -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.tick_params(axis="both", pad=4)
    if yscale == "log":
        ax.set_yscale("log")


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])


def plot_losses_gaps(cfg, traj: pd.DataFrame, outdir: Path) -> None:
    dataset = cfg["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.1), sharex=True)

    # losses
    axes[0].plot(traj["epoch"], traj["few_loss"], color=COL_FEW, marker="o", label=r"$L_{\mathrm{few}}$")
    axes[0].plot(traj["epoch"], traj["full_loss"], color=COL_FULL, marker="o", label=r"$L_{\mathrm{full}}$")
    axes[0].plot(traj["epoch"], traj["test_loss"], color=COL_TEST, marker="o", label=r"$L_{\mathrm{test}}$")
    axes[0].set_title("Cross-entropy losses")
    _style_axis(axes[0])

    # accuracies
    axes[1].plot(traj["epoch"], traj["few_acc"], color=COL_FEW, marker="o", label=r"$\mathrm{Acc}_{\mathrm{few}}$")
    axes[1].plot(traj["epoch"], traj["full_acc"], color=COL_FULL, marker="o", label=r"$\mathrm{Acc}_{\mathrm{full}}$")
    axes[1].plot(traj["epoch"], traj["test_acc"], color=COL_TEST, marker="o", label=r"$\mathrm{Acc}_{\mathrm{test}}$")
    axes[1].set_title("Accuracies")
    _style_axis(axes[1])

    # support-fitting gap
    axes[2].plot(traj["epoch"], traj["gap_full_minus_few_loss"], color=COL_FULL, marker="o", label=r"$\Delta_{\mathrm{full-few}}(t)$")
    axes[2].plot(traj["epoch"], traj["gap_test_minus_few_loss"], color=COL_TEST, marker="o", label=r"$\Delta_{\mathrm{test-few}}(t)$")
    axes[2].set_title("Support-fitting gap")
    _style_axis(axes[2])

    handles, labels = [], []
    for ax in axes:
        h,l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    # unique legend entries preserving order
    uniq = []
    seen = set()
    for h,l in zip(handles, labels):
        if l not in seen:
            uniq.append((h,l)); seen.add(l)
    fig.legend([u[0] for u in uniq], [u[1] for u in uniq], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, columnspacing=1.3, handlelength=2.1)
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.84])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel="Value")
    _save_png_pdf(fig, outdir / f"expI_{dataset}_losses_and_gaps.png")
    plt.close(fig)


def plot_feature_drift(cfg, traj: pd.DataFrame, outdir: Path) -> None:
    dataset = cfg["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.1), sharex=True)

    axes[0].plot(traj["epoch"], traj["param_drift_all"], color=COL_ALL, marker="o", label=r"$D_{\mathrm{param}}^{(\mathrm{all})}(t)$")
    axes[0].plot(traj["epoch"], traj["param_drift_vision"], color=COL_VISION, marker="o", label=r"$D_{\mathrm{param}}^{(\mathrm{vision})}(t)$")
    axes[0].plot(traj["epoch"], traj["param_drift_text"], color=COL_TEXT, marker="o", label=r"$D_{\mathrm{param}}^{(\mathrm{text})}(t)$")
    axes[0].set_title("Relative parameter drift")
    _style_axis(axes[0])

    axes[1].plot(traj["epoch"], traj["image_drift_few"], color=COL_FEW, marker="o", label=r"$D_{\mathrm{img},\mathrm{few}}(t)$")
    axes[1].plot(traj["epoch"], traj["image_drift_full"], color=COL_FULL, marker="o", label=r"$D_{\mathrm{img},\mathrm{full}}(t)$")
    axes[1].plot(traj["epoch"], traj["image_drift_test"], color=COL_TEST, marker="o", label=r"$D_{\mathrm{img},\mathrm{test}}(t)$")
    axes[1].set_title("Image-embedding drift")
    _style_axis(axes[1])

    axes[2].plot(traj["epoch"], traj["text_drift"], color=COL_TEXT, marker="o", label=r"$D_{\mathrm{text}}(t)$")
    axes[2].plot(traj["epoch"], traj["margin_drift_full"], color=COL_FULL, linestyle="--", marker="o", label=r"$D_{\mathrm{margin},\mathrm{full}}(t)$")
    axes[2].plot(traj["epoch"], traj["margin_drift_test"], color=COL_TEST, linestyle="--", marker="o", label=r"$D_{\mathrm{margin},\mathrm{test}}(t)$")
    axes[2].plot(traj["epoch"], traj["margin_drift_few"], color=COL_FEW, linestyle=":", marker="o", label=r"$D_{\mathrm{margin},\mathrm{few}}(t)$")
    axes[2].set_title("Text / class-margin drift")
    _style_axis(axes[2])

    handles, labels = [], []
    for ax in axes:
        h,l = ax.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
    uniq=[]; seen=set()
    for h,l in zip(handles, labels):
        if l not in seen:
            uniq.append((h,l)); seen.add(l)
    fig.legend([u[0] for u in uniq], [u[1] for u in uniq], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=5, frameon=False, columnspacing=1.15, handlelength=2.1)
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.84])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel="Drift")
    _save_png_pdf(fig, outdir / f"expI_{dataset}_feature_drift.png")
    plt.close(fig)


def plot_block_drift(cfg, block: pd.DataFrame, outdir: Path) -> None:
    dataset = cfg["dataset"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)
    epochs = sorted(block['epoch'].unique())
    colors = {ep: EPOCH_CMAP(i / max(len(epochs)-1,1)) for i, ep in enumerate(epochs)}
    for ax, enc in zip(axes, ['vision','text']):
        sub = block[block['encoder']==enc].copy()
        for ep in epochs:
            ss = sub[sub['epoch']==ep].sort_values('block')
            ax.plot(ss['block'], ss['rel_param_drift'], color=colors[ep], marker='o', label=fr"$t={int(ep)}$")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xticks(sorted(sub['block'].unique()))
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, columnspacing=1.2, handlelength=2.1)
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.84])
    _add_global_xy_labels(fig, xlabel=r"Transformer block $\ell$", ylabel=r"$D_{\mathrm{param}}^{(\ell)}(t)$")
    _save_png_pdf(fig, outdir / f"expI_{dataset}_block_parameter_drift.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, block, summary = _load_bundle(args.logdir, args.run_name)
    plot_losses_gaps(cfg, traj, args.outdir)
    plot_feature_drift(cfg, traj, args.outdir)
    plot_block_drift(cfg, block, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

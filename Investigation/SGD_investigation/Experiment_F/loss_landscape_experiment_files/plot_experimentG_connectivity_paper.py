#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paper-style plotting for Experiment G — basin connectivity across SGD seeds.

Inputs (from measure_experimentG_connectivity.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_pairwise_interpolation.csv
  - <run_name>_pairwise_summary.csv

Main paper figure emitted:
  - expG_connectivity.(png|pdf)

Optional diagnostics:
  - expG_seed_trajectories.(png|pdf)
  - expG_pairwise_barriers.(png|pdf)
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
    "lines.markersize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PAIR_LABELS = {
    "1↔2": r"$s_1 \leftrightarrow s_2$",
    "1↔3": r"$s_1 \leftrightarrow s_3$",
    "2↔3": r"$s_2 \leftrightarrow s_3$",
}
PANEL_TITLES = {
    "few_pair_loss": r"$L_{\mathrm{few}}^{(a,b)}$",
    "full_loss": r"$L_{\mathrm{full}}$",
    "test_loss": r"$L_{\mathrm{test}}$",
}
BARRIER_TITLES = {
    "few_pair_barrier": r"$B_{\mathrm{few}}^{(a,b)}$",
    "full_barrier": r"$B_{\mathrm{full}}$",
    "test_barrier": r"$B_{\mathrm{test}}$",
}
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    return ap.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    pair_interp = pd.read_csv(logdir / f"{run_name}_pairwise_interpolation.csv")
    pair_summary = pd.read_csv(logdir / f"{run_name}_pairwise_summary.csv")
    return cfg, traj, pair_interp, pair_summary


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])


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


def _pair_label(label: str) -> str:
    return PAIR_LABELS.get(label, label)


def plot_connectivity(pair_interp: pd.DataFrame, pair_summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), sharex=True)
    pair_summary = pair_summary.sort_values(["seed_a", "seed_b"]).copy()
    pair_labels = list(pair_summary["pair_label"])
    color_map = {label: PAIR_COLORS[i % len(PAIR_COLORS)] for i, label in enumerate(pair_labels)}
    cols = ["few_pair_loss", "full_loss", "test_loss"]

    all_handles = []
    all_labels = []
    for ax, col in zip(axes, cols):
        ymin, ymax = np.inf, -np.inf
        for label in pair_labels:
            sub = pair_interp[pair_interp["pair_label"] == label].sort_values("alpha")
            x = sub["alpha"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            color = color_map[label]
            h, = ax.plot(x, y, color=color, label=_pair_label(label))
            idx = int(np.argmin(y))
            ax.plot([x[idx]], [y[idx]], marker="o", markersize=5.2, color=color, markeredgecolor="white", markeredgewidth=0.6)
            ymin = min(ymin, float(np.min(y)))
            ymax = max(ymax, float(np.max(y)))
            if col == cols[0]:
                all_handles.append(h)
                all_labels.append(_pair_label(label))
        ax.set_title(PANEL_TITLES[col])
        ax.set_xlabel("")
        ax.set_xlim(-0.02, 1.02)
        pad = 0.06 * (ymax - ymin + 1e-8)
        ax.set_ylim(ymin - pad, ymax + pad)
        _style_axis(ax)

    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.92])
    _add_global_xy_labels(fig, xlabel=r"Interpolation coefficient $\alpha$", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expG_connectivity.png")
    plt.close(fig)


def plot_seed_trajectories(traj: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), sharex=True)
    panels = [("few_loss", r"$L_{\mathrm{few}}$"), ("full_loss", r"$L_{\mathrm{full}}$"), ("test_loss", r"$L_{\mathrm{test}}$")]
    seeds = sorted(traj["seed"].unique())
    for ax, (col, title) in zip(axes, panels):
        for idx, seed in enumerate(seeds):
            sub = traj[traj["seed"] == seed].sort_values("epoch")
            ax.plot(sub["epoch"], sub[col], marker="o", color=SEED_COLORS[idx % len(SEED_COLORS)], label=fr"$s_{int(seed)}$")
        ax.set_title(title)
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(handles), frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.92])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expG_seed_trajectories.png")
    plt.close(fig)


def plot_barriers(pair_summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), sharex=True, sharey=True)
    pair_summary = pair_summary.sort_values(["seed_a", "seed_b"]).copy()
    x = np.arange(len(pair_summary))
    xticklabels = [_pair_label(lbl) for lbl in pair_summary["pair_label"]]
    cols = ["few_pair_barrier", "full_barrier", "test_barrier"]
    for ax, col in zip(axes, cols):
        vals = pair_summary[col].to_numpy(dtype=float)
        colors = [PAIR_COLORS[i % len(PAIR_COLORS)] for i in range(len(vals))]
        ax.bar(x, vals, color=colors)
        ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.1)
        ax.set_title(BARRIER_TITLES[col])
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(-0.002, max(0.01, float(np.max(vals)) + 0.005))
        _style_axis(ax)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.95])
    _add_global_xy_labels(fig, xlabel="Seed pair", ylabel="Barrier above endpoint chord")
    _save_png_pdf(fig, outdir / "expG_pairwise_barriers.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, pair_interp, pair_summary = _load_bundle(args.logdir, args.run_name)
    plot_connectivity(pair_interp, pair_summary, args.outdir)
    plot_seed_trajectories(traj, args.outdir)
    plot_barriers(pair_summary, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

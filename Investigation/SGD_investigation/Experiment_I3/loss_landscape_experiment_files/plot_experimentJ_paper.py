#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paper-style summary plots for Experiment J.

This script reads one Experiment J log directory containing the three dataset runs
(EuroSAT, Oxford-Pets, DTD) for a single protocol J1/J2/J3/J4 and generates the
paper-ready 2x3 summary figure used in the NeurIPS source.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_SIZE = 13
mpl.rcParams.update({
    "font.size": BASE_SIZE + 1,
    "axes.titlesize": BASE_SIZE + 1,
    "axes.labelsize": BASE_SIZE + 1,
    "xtick.labelsize": BASE_SIZE + 1,
    "ytick.labelsize": BASE_SIZE + 1,
    "legend.fontsize": BASE_SIZE,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.3,
    "lines.markersize": 5.5,
})

VARIANT_ORDER = [
    "fullbatch_nomom",
    "div2_nomom",
    "div4_nomom",
    "fullbatch_noise",
    "fullbatch_momentum",
]
VARIANT_LABEL = {
    "fullbatch_nomom": "full batch",
    "div2_nomom": "half batch",
    "div4_nomom": "quarter batch",
    "fullbatch_noise": "full batch + noise",
    "fullbatch_momentum": "full batch + momentum",
}
COLORS = {
    "fullbatch_nomom": "#1f77b4",
    "div2_nomom": "#ff7f0e",
    "div4_nomom": "#2ca02c",
    "fullbatch_noise": "#d62728",
    "fullbatch_momentum": "#9467bd",
}
MARKERS = {
    "fullbatch_nomom": "o",
    "div2_nomom": "s",
    "div4_nomom": "^",
    "fullbatch_noise": "D",
    "fullbatch_momentum": "P",
}
DATASET_ORDER = ["eurosat", "oxford_pets", "dtd"]
DATASET_TITLE = {
    "eurosat": "EuroSAT",
    "oxford_pets": "Oxford-Pets",
    "dtd": "DTD",
}
PROTOCOL_META = {
    "J1": {
        "xcol": "epoch",
        "xlabel": r"Epoch $t$",
        "outfile": "expJ1_epoch_matched_summary",
    },
    "J2": {
        "xcol": "update_budget_ratio",
        "xlabel": r"Update-budget ratio $\xi_{\mathrm{upd}}$",
        "outfile": "expJ2_update_matched_summary",
    },
    "J3": {
        "xcol": "example_budget_ratio",
        "xlabel": r"Example-budget ratio $\xi_{\mathrm{ex}}$",
        "outfile": "expJ3_example_matched_summary",
    },
    "J4": {
        "xcol": "update_budget_ratio",
        "xlabel": r"Common budget ratio $\xi$",
        "outfile": "expJ4_update_example_matched_summary",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--protocol", choices=["J1", "J2", "J3", "J4"], required=True)
    return ap.parse_args()


def load_trajectories(logdir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for fp in glob.glob(str(logdir / "*_trajectory.csv")):
        df = pd.read_csv(fp)
        if "dataset" in df.columns:
            dataset = str(df["dataset"].iloc[0])
        else:
            name = Path(fp).name
            if "dataset_eurosat" in name:
                dataset = "eurosat"
            elif "dataset_oxford_pets" in name:
                dataset = "oxford_pets"
            elif "dataset_dtd" in name:
                dataset = "dtd"
            else:
                continue
        out[dataset] = df.copy()
    return out


def _save(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)


def _style(ax: plt.Axes, logy: bool = False) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0, which="both" if logy else "major")
    if logy:
        ax.set_yscale("log")


def make_summary(trajectories: dict[str, pd.DataFrame], protocol: str, outdir: Path) -> Path:
    meta = PROTOCOL_META[protocol]
    xcol = meta["xcol"]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex=False)
    legend_handles = []
    legend_labels = []

    for j, dataset in enumerate(DATASET_ORDER):
        df = trajectories[dataset].copy()
        top_ax = axes[0, j]
        bot_ax = axes[1, j]

        for variant in VARIANT_ORDER:
            sub = df[df["variant"] == variant].sort_values(xcol)
            if sub.empty:
                continue
            ln, = top_ax.plot(
                sub[xcol], sub["test_acc"],
                color=COLORS[variant], marker=MARKERS[variant],
                label=VARIANT_LABEL[variant],
            )
            bot_ax.plot(
                sub[xcol], sub["full_curv_grad"],
                color=COLORS[variant], marker=MARKERS[variant],
                label=VARIANT_LABEL[variant],
            )
            if variant not in legend_labels:
                legend_handles.append(ln)
                legend_labels.append(variant)

        top_ax.set_title(DATASET_TITLE[dataset])
        _style(top_ax, logy=False)
        _style(bot_ax, logy=True)
        bot_ax.set_xlabel(meta["xlabel"])

        if j == 0:
            top_ax.set_ylabel("Test accuracy (%)")
            bot_ax.set_ylabel(r"$\chi_{\mathrm{full}}^{(v)}$")

        if xcol != "epoch":
            top_ax.set_xlim(-0.02, 1.02)
            bot_ax.set_xlim(-0.02, 1.02)
        else:
            xvals = np.sort(df[xcol].unique())
            if len(xvals):
                xmin, xmax = xvals.min(), xvals.max()
                top_ax.set_xlim(xmin, xmax)
                bot_ax.set_xlim(xmin, xmax)

    fig.legend(
        legend_handles,
        [VARIANT_LABEL[v] for v in legend_labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.92])
    outbase = outdir / meta["outfile"]
    _save(fig, outbase)
    plt.close(fig)
    return outbase.with_suffix(".pdf")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    traj = load_trajectories(args.logdir)
    if set(DATASET_ORDER) - set(traj.keys()):
        missing = set(DATASET_ORDER) - set(traj.keys())
        raise FileNotFoundError(f"Missing datasets in {args.logdir}: {sorted(missing)}")
    out = make_summary(traj, args.protocol, args.outdir)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

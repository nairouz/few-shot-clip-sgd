#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment F — Per-block loss landscape and sensitivity map.

Required inputs (from measure_experimentF_block_sensitivity.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_block_landscape.csv
  - <run_name>_block_summary.csv
  - <run_name>_summary.json

Figures emitted:
  A) expF_full_delta_at1.(png|pdf)
  B) expF_test_delta_at1.(png|pdf)
  C) expF_full_safe_radius.(png|pdf)
  D) expF_test_safe_radius.(png|pdf)
  E) expF_full_curvature.(png|pdf)
  F) expF_test_curvature.(png|pdf)
  G) expF_full_alpha_min.(png|pdf)
  H) expF_test_alpha_min.(png|pdf)
  I) expF_gap_full_minus_few_at1.(png|pdf)
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
# Style (matched to Experiment 2 / Experiment A plotting style)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    curves = pd.read_csv(logdir / f"{run_name}_block_landscape.csv")
    summ = pd.read_csv(logdir / f"{run_name}_block_summary.csv")
    compact = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, curves, summ, compact


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


def _epoch_colors(epochs):
    cmap = mpl.colormaps.get_cmap("tab10")
    keys = sorted(epochs)
    denom = max(len(keys) - 1, 1)
    return {ep: cmap(i / denom) for i, ep in enumerate(keys)}


def _plot_metric_by_block(summary: pd.DataFrame, metric: str, title: str, ylabel: str, outpath: Path, *, yscale: str = "linear", refline0: bool = False):
    epochs = sorted(summary["epoch"].unique().tolist())
    colors = _epoch_colors(epochs)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharex=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = summary[summary["encoder"] == enc].copy()
        for ep in epochs:
            ss = sub[sub["epoch"] == ep].sort_values("block")
            ax.plot(ss["block"], ss[metric], marker="o", color=colors[ep], label=f"t={ep}")
        if refline0:
            ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.0)
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both" if yscale == "log" else "major")
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=ylabel)
    _save_png_pdf(fig, outpath)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, curves, summary, compact = _load_bundle(args.logdir, args.run_name)

    _plot_metric_by_block(
        summary,
        metric="full_delta_at1",
        title="Full-train loss increase at α=1",
        ylabel=r"$L_{full}(\alpha=1)-L_{full}(0)$",
        outpath=args.outdir / "expF_full_delta_at1.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="test_delta_at1",
        title="Test loss increase at α=1",
        ylabel=r"$L_{test}(\alpha=1)-L_{test}(0)$",
        outpath=args.outdir / "expF_test_delta_at1.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="full_safe_radius",
        title="Positive safe radius (full-train)",
        ylabel=r"Safe radius $\alpha_{safe}^{full}$",
        outpath=args.outdir / "expF_full_safe_radius.png",
    )
    _plot_metric_by_block(
        summary,
        metric="test_safe_radius",
        title="Positive safe radius (test)",
        ylabel=r"Safe radius $\alpha_{safe}^{test}$",
        outpath=args.outdir / "expF_test_safe_radius.png",
    )
    _plot_metric_by_block(
        summary,
        metric="full_curvature0",
        title="Local curvature at α=0 (full-train)",
        ylabel=r"Curvature $\kappa_{full}(0)$",
        outpath=args.outdir / "expF_full_curvature.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="test_curvature0",
        title="Local curvature at α=0 (test)",
        ylabel=r"Curvature $\kappa_{test}(0)$",
        outpath=args.outdir / "expF_test_curvature.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="full_alpha_min",
        title="Best blockwise scale (full-train)",
        ylabel=r"$\alpha^{*}_{full}$",
        outpath=args.outdir / "expF_full_alpha_min.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="test_alpha_min",
        title="Best blockwise scale (test)",
        ylabel=r"$\alpha^{*}_{test}$",
        outpath=args.outdir / "expF_test_alpha_min.png",
        refline0=True,
    )
    _plot_metric_by_block(
        summary,
        metric="gap_full_minus_few_at1",
        title="Support vs full-train mismatch at α=1",
        ylabel=r"$\Delta L_{full}(1)-\Delta L_{few}(1)$",
        outpath=args.outdir / "expF_gap_full_minus_few_at1.png",
        refline0=True,
    )

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

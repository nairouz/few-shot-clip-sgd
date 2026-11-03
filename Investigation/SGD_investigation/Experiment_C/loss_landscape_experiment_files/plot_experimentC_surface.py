#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment C — 2D local surfaces around SGD checkpoints.

Required inputs (from measure_experimentC_surface.py):
  - <surface_run_name>_surface.npz
  - <surface_run_name>_meta.json

Figures emitted:
  A) expC_surface_contours.(png|pdf)
  B) expC_surface_cross_sections.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Style — matched to Experiment A plotting style
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
    meta = json.load(open(logdir / f"{run_name}_meta.json", "r"))
    surf = np.load(logdir / f"{run_name}_surface.npz")
    return meta, surf


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


def _contour_levels(z: np.ndarray, n_levels: int = 18):
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    if np.isclose(zmin, zmax):
        zmax = zmin + 1e-6
    return np.linspace(zmin, zmax, n_levels)


def plot_A_surface_contours(meta: dict, surf, outdir: Path) -> None:
    alphas = surf["alphas"]
    betas = surf["betas"]
    A, B = np.meshgrid(alphas, betas)
    panels = [
        (surf["few_surface"], "Support loss"),
        (surf["full_surface"], "Full-train loss"),
        (surf["test_surface"], "Test loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.0), sharex=True, sharey=True)
    for ax, (Z, title) in zip(axes, panels):
        levels = _contour_levels(Z)
        cf = ax.contourf(A, B, Z, levels=levels, cmap="Blues")
        ax.contour(A, B, Z, levels=levels[::2], colors="white", linewidths=0.8, alpha=0.85)
        ax.axvline(0.0, color=COL_MILE, linestyle=":", linewidth=1.0)
        ax.axhline(0.0, color=COL_MILE, linestyle=":", linewidth=1.0)
        ax.scatter([0.0], [0.0], color=COL_END, s=35, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.047, pad=0.02)
        cbar.ax.tick_params(labelsize=BASE_SIZE)

    fig.suptitle(
        f"Experiment C — checkpoint t={meta['checkpoint_epoch']} | probe={meta['probe']} | dataset={meta['dataset']}",
        y=1.02,
    )
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.97])
    _add_global_xy_labels(fig, xlabel=r"SGD direction coefficient $\alpha$", ylabel=r"Orthogonal probe coefficient $\beta$")
    _save_png_pdf(fig, outdir / "expC_surface_contours.png")
    plt.close(fig)


def plot_B_surface_cross_sections(meta: dict, surf, outdir: Path) -> None:
    alphas = surf["alphas"]
    betas = surf["betas"]
    j0 = int(np.argmin(np.abs(alphas)))
    i0 = int(np.argmin(np.abs(betas)))
    panels = [
        (surf["few_surface"], "Support loss"),
        (surf["full_surface"], "Full-train loss"),
        (surf["test_surface"], "Test loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=False)
    for ax, (Z, title) in zip(axes, panels):
        alpha_cut = Z[i0, :]
        beta_cut = Z[:, j0]
        ax.plot(alphas, alpha_cut, color=COL_MAIN, marker="o", markevery=max(1, len(alphas)//8), label="Along SGD direction (β=0)")
        ax.plot(betas, beta_cut, color=COL_ORTH, marker="o", markevery=max(1, len(betas)//8), label="Along probe direction (α=0)")
        ax.axvline(0.0, color=COL_MILE, linestyle=":", linewidth=1.0)
        ax.scatter([0.0], [float(Z[i0, j0])], color=COL_END, s=35, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)

    fig.suptitle(
        f"Central cuts — checkpoint t={meta['checkpoint_epoch']} | probe={meta['probe']} | dataset={meta['dataset']}",
        y=1.02,
    )
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.97])
    _add_global_xy_labels(fig, xlabel=r"Coordinate along the selected axis", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expC_surface_cross_sections.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    meta, surf = _load_bundle(args.logdir, args.run_name)
    plot_A_surface_contours(meta, surf, args.outdir)
    plot_B_surface_cross_sections(meta, surf, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

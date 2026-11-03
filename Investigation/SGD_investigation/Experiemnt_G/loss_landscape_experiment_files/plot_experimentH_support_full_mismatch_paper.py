#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-ready plots for Experiment H — support/full geometric alignment.

Required inputs (from measure_experimentH_support_full_mismatch.py):
  - <run_name>_config.json
  - <run_name>_encoder_alignment.csv
  - <run_name>_surface.csv

Figures emitted:
  A) expH_<dataset>_alignment.(png|pdf)
     Four panels: gradient cosine, gradient-norm ratio, dominant Hessian
     eigenspace alignment, and surface correlation.
  B) expH_<dataset>_surface_<encoder>. (png|pdf)
     Local support/full surfaces at t=0 and t=30 for the selected encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
SURF_CMAP = "viridis"
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
    enc = pd.read_csv(logdir / f"{run_name}_encoder_alignment.csv")
    surf = pd.read_csv(logdir / f"{run_name}_surface.csv")
    return cfg, enc, surf


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


def _surface_corr(sub: pd.DataFrame) -> float:
    if sub.shape[0] < 2:
        return float("nan")
    return float(sub[["few_loss", "full_loss"]].corr().iloc[0, 1])


def plot_alignment_summary(cfg: Dict, enc: pd.DataFrame, surf: pd.DataFrame, outdir: Path) -> None:
    dataset = cfg["dataset"]
    fig, axes = plt.subplots(1, 4, figsize=(18.4, 4.2), sharex=True)
    encoders = [e for e in ["vision", "text"] if e in enc["encoder"].unique()]
    styles = {
        "vision": dict(color=COL_VISION, marker=MARK_VISION, linestyle="-", label="Vision"),
        "text": dict(color=COL_TEXT, marker=MARK_TEXT, linestyle="--", label="Text"),
    }

    for encoder in encoders:
        sub = enc[enc["encoder"] == encoder].sort_values("epoch")
        sty = styles[encoder]
        axes[0].plot(sub["epoch"], sub["grad_cosine"], **sty)
        axes[1].plot(sub["epoch"], np.maximum(sub["grad_norm_ratio"], 1e-8), **sty)
        axes[2].plot(sub["epoch"], sub["hessian_eigvec_abs_cosine"], **sty)
        # surface correlation over the sampled 2D grid
        surf_corr = []
        for t in sorted(sub["epoch"].unique()):
            ss = surf[(surf["encoder"] == encoder) & (surf["epoch"] == t)]
            surf_corr.append(_surface_corr(ss))
        axes[3].plot(sorted(sub["epoch"].unique()), surf_corr, **sty)

    axes[0].set_title(r"$A_g^{(e)}(t)=\cos(g_{\mathrm{few}},g_{\mathrm{full}})$")
    axes[1].set_title(r"$R_g^{(e)}(t)=\|g_{\mathrm{few}}\|_2/\|g_{\mathrm{full}}\|_2$")
    axes[2].set_title(r"$A_H^{(e)}(t)=|\langle v_{1,\mathrm{few}},v_{1,\mathrm{full}}\rangle|$")
    axes[3].set_title(r"$C_{\mathrm{surf}}^{(e)}(t)$")

    for ax in [axes[0], axes[2], axes[3]]:
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[2].set_ylim(0.0, 1.0)
    axes[1].set_ylim(1e-2, 1e1)
    _style_axis(axes[1], yscale="log")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_xlim(-0.5, 30.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=1.6, handlelength=2.3)
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.90])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel="Alignment / correlation")
    _save_png_pdf(fig, outdir / f"expH_{dataset}_alignment.png")
    plt.close(fig)


def _pivot_surface(sub: pd.DataFrame, value_col: str):
    piv = sub.pivot(index="beta_units", columns="alpha_units", values=value_col).sort_index().sort_index(axis=1)
    X = piv.columns.to_numpy(dtype=float)
    Y = piv.index.to_numpy(dtype=float)
    Z = piv.to_numpy(dtype=float)
    return X, Y, Z


def _plot_surface_panel(ax: plt.Axes, sub: pd.DataFrame, value_col: str, title: str, vmin: float, vmax: float):
    X, Y, Z = _pivot_surface(sub, value_col)
    XX, YY = np.meshgrid(X, Y)
    cs = ax.contourf(XX, YY, Z, levels=20, cmap=SURF_CMAP, vmin=vmin, vmax=vmax)
    ax.contour(XX, YY, Z, levels=8, colors="white", linewidths=0.5, alpha=0.45)
    meta = sub.iloc[0]
    # support-gradient axis arrow (black) and projected full-gradient arrow (red)
    radius = 0.58 * max(np.max(np.abs(X)), np.max(np.abs(Y)))
    ax.annotate("", xy=(radius, 0.0), xytext=(0.0, 0.0), arrowprops=dict(color="black", width=1.2, headwidth=7))
    ax.annotate("", xy=(radius * float(meta["full_arrow_x"]), radius * float(meta["full_arrow_y"])), xytext=(0.0, 0.0), arrowprops=dict(color="#d62728", width=1.2, headwidth=7))
    ax.scatter([0.0], [0.0], color="white", s=24, edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.25, zorder=0)
    return cs


def plot_surface_summary(cfg: Dict, surf: pd.DataFrame, outdir: Path, *, encoder: str) -> None:
    dataset = cfg["dataset"]
    surf = surf[surf["encoder"] == encoder].copy()
    if surf.empty:
        return
    epochs = [0, 30] if 30 in surf["epoch"].unique() else [int(surf["epoch"].min()), int(surf["epoch"].max())]
    surf0 = surf[surf["epoch"] == epochs[0]]
    surf1 = surf[surf["epoch"] == epochs[1]]

    few_vmin = min(float(surf0["few_loss"].min()), float(surf1["few_loss"].min()))
    few_vmax = max(float(surf0["few_loss"].max()), float(surf1["few_loss"].max()))
    full_vmin = min(float(surf0["full_loss"].min()), float(surf1["full_loss"].min()))
    full_vmax = max(float(surf0["full_loss"].max()), float(surf1["full_loss"].max()))

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.3), sharex=True, sharey=True)
    cs0 = _plot_surface_panel(axes[0], surf0, "few_loss", rf"${{L}}_{{\mathrm{{few}}}}$ at $t={epochs[0]}$", few_vmin, few_vmax)
    cs1 = _plot_surface_panel(axes[1], surf0, "full_loss", rf"${{L}}_{{\mathrm{{full}}}}$ at $t={epochs[0]}$", full_vmin, full_vmax)
    cs2 = _plot_surface_panel(axes[2], surf1, "few_loss", rf"${{L}}_{{\mathrm{{few}}}}$ at $t={epochs[1]}$", few_vmin, few_vmax)
    cs3 = _plot_surface_panel(axes[3], surf1, "full_loss", rf"${{L}}_{{\mathrm{{full}}}}$ at $t={epochs[1]}$", full_vmin, full_vmax)

    for ax in axes:
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.tick_params(axis="both", pad=4)

    # put colorbars outside the surface panels so they never overlap
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.82, wspace=0.18)
    cax1 = fig.add_axes([0.895, 0.57, 0.012, 0.22])
    cbar1 = fig.colorbar(cs0, cax=cax1)
    cbar1.set_label(r"$L_{\mathrm{few}}$")
    cbar1.locator = MaxNLocator(nbins=5)
    cbar1.update_ticks()
    cax2 = fig.add_axes([0.895, 0.23, 0.012, 0.22])
    cbar2 = fig.colorbar(cs1, cax=cax2)
    cbar2.set_label(r"$L_{\mathrm{full}}$")
    cbar2.locator = MaxNLocator(nbins=5)
    cbar2.update_ticks()

    # small legend above the row, not on the surfaces
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.6, label=r"$u_{\mathrm{few},t}^{(%s)}$" % encoder),
        Line2D([0], [0], color="#d62728", lw=2.6, label=r"projected $u_{\mathrm{full},t}^{(%s)}$" % encoder),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.48, 0.98), ncol=2, frameon=False, columnspacing=1.5, handlelength=2.4)
    fig.text(0.5, 0.05, r"$\alpha$ (support-gradient axis, in step units)", ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.5, r"$\beta$ (full-gradient orthogonal axis, in step units)", ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])
    _save_png_pdf(fig, outdir / f"expH_{dataset}_surface_{encoder}.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, enc, surf = _load_bundle(args.logdir, args.run_name)
    plot_alignment_summary(cfg, enc, surf, args.outdir)
    for encoder in ["vision", "text"]:
        plot_surface_summary(cfg, surf, args.outdir, encoder=encoder)
    print(f"Saved Experiment H paper plots to: {args.outdir}")


if __name__ == "__main__":
    main()

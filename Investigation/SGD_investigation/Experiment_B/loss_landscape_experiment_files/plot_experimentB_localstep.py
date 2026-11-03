#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment B — Local 1D landscape along the actual SGD step.

Required inputs (from measure_experimentB_localstep.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_direction_stats.csv
  - <run_name>_localstep_curves.csv
  - <run_name>_safe_radius.csv
  - <run_name>_summary.json

Figures emitted:
  A) expB_localstep_losses.(png|pdf)
  B) expB_safe_radius.(png|pdf)
  C) expB_minimizer_multiplier.(png|pdf)
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
# Style (paper-ready, matched to the notation used in neurips_2026.tex)
# ============================================================================
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

COL_MAIN = "#1f77b4"
COL_RANDOM = "#7f7f7f"
COL_HESS = "#d62728"
COL_STEP = "#2ca02c"

DIRECTION_COLOR = {
    "sgd_step": COL_MAIN,
    "random": COL_RANDOM,
    "hessian_top": COL_HESS,
}
DIRECTION_MARKER = {
    "sgd_step": "o",
    "random": "s",
    "hessian_top": "^",
}
DIRECTION_LINESTYLE = {
    "sgd_step": "-",
    "random": "-.",
    "hessian_top": "--",
}
DIRECTION_LABEL = {
    "sgd_step": r"$u_t^{\mathrm{sgd}}$",
    "random": r"$u_t^{\mathrm{rand}}$",
    "hessian_top": r"$u_t^{\mathrm{top}}$",
}
LOSS_TITLES = {
    "few": r"$L_{\mathrm{few}}$",
    "full": r"$L_{\mathrm{full}}$",
    "test": r"$L_{\mathrm{test}}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    dstat = pd.read_csv(logdir / f"{run_name}_direction_stats.csv")
    curves = pd.read_csv(logdir / f"{run_name}_localstep_curves.csv")
    safe = pd.read_csv(logdir / f"{run_name}_safe_radius.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, dstat, curves, safe, summary


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fs = mpl.rcParams.get("axes.labelsize", BASE_SIZE + 1)
    fw = mpl.rcParams.get("axes.labelweight", "semibold")
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=fs, fontweight=fw)
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=fs, fontweight=fw)


def _style_axis(ax: plt.Axes, *, which: str = "major") -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0, which=which)
    ax.tick_params(axis="both", pad=4)


def _clean_boundary_series(values: pd.Series | np.ndarray, *, lo: float = 0.0, hi: float = 3.0, tol: float = 5e-2):
    arr = np.asarray(values, dtype=float).copy()
    arr[np.abs(arr - lo) <= tol] = lo
    arr[np.abs(arr - hi) <= tol] = hi
    return arr


def _set_epoch_ticks(ax: plt.Axes, epochs: np.ndarray) -> None:
    ax.set_xticks(epochs)
    ax.set_xlim(float(np.min(epochs)) - 0.5, float(np.max(epochs)) + 0.5)


def _loss_panel_title(key: str) -> str:
    return LOSS_TITLES[key]


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def plot_A_localstep_losses(curves: pd.DataFrame, outdir: Path) -> None:
    sgd = curves[curves["direction"] == "sgd_step"].copy()
    epochs = np.array(sorted(sgd["epoch"].unique()), dtype=float)
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(epochs)))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9), sharex=True)
    panels = [
        ("few_loss", "few"),
        ("full_loss", "full"),
        ("test_loss", "test"),
    ]

    for ax, (col, loss_key) in zip(axes, panels):
        for color, epoch in zip(colors, epochs):
            sub = sgd[sgd["epoch"] == epoch].sort_values("step_multiplier")
            ax.plot(sub["step_multiplier"], sub[col], color=color, label=fr"$t={int(epoch)}$")
        ax.axvline(0.0, color=COL_RANDOM, linestyle=":", linewidth=1.0, alpha=0.9)
        ax.axvline(1.0, color=COL_STEP, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.set_title(_loss_panel_title(loss_key))
        ax.set_xlabel("")
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, ncol=2, title=r"Checkpoint $t$", columnspacing=0.9, handlelength=1.8)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(
        fig,
        xlabel=r"Multiplier $c$ in $\Theta_t + c\,u_t^{\mathrm{sgd}}$",
        ylabel=r"$\phi_{\ell, u_t^{\mathrm{sgd}}}(c; t)$",
    )
    _save_png_pdf(fig, outdir / "expB_localstep_losses.png")
    plt.close(fig)


def _plot_direction_summary(safe: pd.DataFrame, outdir: Path, *, kind: str, ylabel: str, fname: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9), sharex=True, sharey=True)
    panels = [
        (f"few_{kind}", "few"),
        (f"full_{kind}", "full"),
        (f"test_{kind}", "test"),
    ]
    epochs = np.array(sorted(safe["epoch"].unique()), dtype=float)

    for ax, (col, loss_key) in zip(axes, panels):
        for direction in ["sgd_step", "random", "hessian_top"]:
            if direction not in set(safe["direction"]):
                continue
            sub = safe[safe["direction"] == direction].sort_values("epoch")
            y = _clean_boundary_series(sub[col])
            ax.plot(
                sub["epoch"],
                y,
                marker=DIRECTION_MARKER[direction],
                linestyle=DIRECTION_LINESTYLE[direction],
                color=DIRECTION_COLOR[direction],
                label=DIRECTION_LABEL[direction],
            )
        ax.axhline(1.0, color=COL_STEP, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_title(_loss_panel_title(loss_key))
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 3.15)
        ax.set_yticks([0, 1, 2, 3])
        _set_epoch_ticks(ax, epochs)
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc="center left")
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel=ylabel)
    _save_png_pdf(fig, outdir / fname)
    plt.close(fig)


def plot_B_safe_radius(safe: pd.DataFrame, outdir: Path) -> None:
    _plot_direction_summary(
        safe,
        outdir,
        kind="safe_noninc",
        ylabel=r"$\rho_{\ell,u}(t)$",
        fname="expB_safe_radius.png",
    )


def plot_C_minimizer_multiplier(safe: pd.DataFrame, outdir: Path) -> None:
    _plot_direction_summary(
        safe,
        outdir,
        kind="multiplier_at_min",
        ylabel=r"$c_{\ell,u}^{\star}(t)$",
        fname="expB_minimizer_multiplier.png",
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, dstat, curves, safe, summary = _load_bundle(args.logdir, args.run_name)
    plot_A_localstep_losses(curves, args.outdir)
    plot_B_safe_radius(safe, args.outdir)
    plot_C_minimizer_multiplier(safe, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

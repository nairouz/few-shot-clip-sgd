#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Experiment E — directional curvature with paper notation.

Required inputs (from measure_experimentE_directional_curvature.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_directional_curvature.csv
  - <run_name>_block_curvature.csv
  - <run_name>_summary.json

Figures emitted:
  A) expE_directional_curvature_over_time.(png|pdf)
  B) expE_block_curvature_full_t00.(png|pdf)
  C) expE_block_curvature_full_tXX.(png|pdf)
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
# Style: matched to the Experiment B paper figures
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

COL_SGD = "#1f77b4"
COL_FULL = "#ff7f0e"
COL_TOP = "#d62728"
COL_RAND = "#7f7f7f"

DISPLAY_COLOR = {
    "sgd_step_few_grad": COL_SGD,
    "sgd_step": COL_SGD,
    "few_grad": COL_SGD,
    "full_grad": COL_FULL,
    "top_eig_full": COL_TOP,
    "random": COL_RAND,
}
DISPLAY_MARKER = {
    "sgd_step_few_grad": "o",
    "sgd_step": "o",
    "few_grad": "o",
    "full_grad": "s",
    "top_eig_full": "^",
    "random": "D",
}
DISPLAY_LINESTYLE = {
    "sgd_step_few_grad": "-",
    "sgd_step": "-",
    "few_grad": "-",
    "full_grad": "-.",
    "top_eig_full": "--",
    "random": ":",
}
DISPLAY_NAME = {
    "sgd_step_few_grad": r"$u_t^{\mathrm{sgd}} \equiv u_t^{\mathrm{few}}$",
    "sgd_step": r"$u_t^{\mathrm{sgd}}$",
    "few_grad": r"$u_t^{\mathrm{few}}$",
    "full_grad": r"$u_t^{\mathrm{full}}$",
    "top_eig_full": r"$u_t^{\mathrm{top}}$",
    "random": r"$u_t^{\mathrm{rand}}$",
}
PLOT_ORDER = ["sgd_step_few_grad", "sgd_step", "few_grad", "full_grad", "top_eig_full", "random"]
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
    parser.add_argument(
        "--block_epochs",
        type=str,
        default="0,30",
        help="Comma-separated checkpoint epochs used for the full-loss block decomposition plots.",
    )
    parser.add_argument(
        "--yscale",
        type=str,
        default="symlog",
        choices=["linear", "symlog"],
        help="Y-scale used in the plots.",
    )
    parser.add_argument(
        "--linthresh",
        type=float,
        default=1.0,
        help="linthresh used when --yscale symlog.",
    )
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    global_df = pd.read_csv(logdir / f"{run_name}_directional_curvature.csv")
    block_df = pd.read_csv(logdir / f"{run_name}_block_curvature.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, global_df, block_df, summary


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])


def _style_axis(ax: plt.Axes, *, which: str = "major", yscale: str = "linear", linthresh: float = 1.0) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0, which=which)
    ax.tick_params(axis="both", pad=4)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def _should_merge_sgd_and_few(global_df: pd.DataFrame) -> bool:
    piv = global_df.pivot_table(index=["epoch", "loss_name"], columns="direction_family", values="curvature", aggfunc="mean")
    if "sgd_step" not in piv.columns or "few_grad" not in piv.columns:
        return False
    a = piv["sgd_step"].to_numpy(dtype=float)
    b = piv["few_grad"].to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return False
    return np.allclose(a[valid], b[valid], rtol=1e-8, atol=1e-10)


def _prepare_global_curvature(global_df: pd.DataFrame) -> pd.DataFrame:
    df = global_df.copy()
    if _should_merge_sgd_and_few(df):
        df.loc[df["direction_family"].isin(["sgd_step", "few_grad"]), "direction_family"] = "sgd_step_few_grad"
    return df


def _prepare_block_curvature(block_df: pd.DataFrame) -> pd.DataFrame:
    df = block_df.copy()
    tmp = block_df.rename(columns={"global_curvature": "curvature"}).assign(loss_name=block_df["loss_name"])
    if _should_merge_sgd_and_few(tmp):
        df.loc[df["direction_family"].isin(["sgd_step", "few_grad"]), "direction_family"] = "sgd_step_few_grad"
    return df


def _plot_random_with_band(ax: plt.Axes, grouped: pd.DataFrame, *, xcol: str, ycol: str):
    x = grouped[xcol].to_numpy(dtype=float)
    mean = grouped["mean"].to_numpy(dtype=float)
    std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(
        x,
        mean,
        marker=DISPLAY_MARKER["random"],
        linestyle=DISPLAY_LINESTYLE["random"],
        color=DISPLAY_COLOR["random"],
        label=DISPLAY_NAME["random"],
    )
    ax.fill_between(x, mean - std, mean + std, color=DISPLAY_COLOR["random"], alpha=0.15)


def plot_A_directional_curvature_over_time(global_df: pd.DataFrame, outdir: Path, *, yscale: str, linthresh: float) -> None:
    df = _prepare_global_curvature(global_df)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), sharex=True)
    losses = ["few", "full", "test"]
    legend_handles = None
    legend_labels = None

    for ax, loss_name in zip(axes, losses):
        sub = df[df["loss_name"] == loss_name].copy()
        for direction in PLOT_ORDER:
            s = sub[sub["direction_family"] == direction].sort_values("epoch")
            if s.empty:
                continue
            if direction == "random":
                grouped = s.groupby("epoch")["curvature"].agg(["mean", "std"]).reset_index()
                _plot_random_with_band(ax, grouped, xcol="epoch", ycol="curvature")
            else:
                grouped = s.groupby("epoch")["curvature"].mean().reset_index()
                ax.plot(
                    grouped["epoch"],
                    grouped["curvature"],
                    marker=DISPLAY_MARKER[direction],
                    linestyle=DISPLAY_LINESTYLE[direction],
                    color=DISPLAY_COLOR[direction],
                    label=DISPLAY_NAME[direction],
                )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_title(LOSS_TITLES[loss_name])
        ax.set_xlabel("")
        _style_axis(ax, which="both" if yscale == "symlog" else "major", yscale=yscale, linthresh=linthresh)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.92])
    _add_global_xy_labels(
        fig,
        xlabel=r"Checkpoint epoch $t$",
        ylabel=r"$\kappa_{\ell}(u;t)=u^{\top} H_{\ell}(\Theta_t) u$",
    )
    _save_png_pdf(fig, outdir / "expE_directional_curvature_over_time.png")
    plt.close(fig)


def plot_block_curvature_for_epoch(block_df: pd.DataFrame, outdir: Path, *, epoch: int, yscale: str, linthresh: float) -> None:
    df = _prepare_block_curvature(block_df)
    sub = df[(df["loss_name"] == "full") & (df["epoch"] == epoch)].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    legend_handles = None
    legend_labels = None
    for ax, encoder in zip(axes, ["vision", "text"]):
        enc_df = sub[sub["encoder"] == encoder].copy()
        for direction in PLOT_ORDER:
            ddf = enc_df[enc_df["direction_family"] == direction].copy()
            if ddf.empty:
                continue
            if direction == "random":
                grouped = ddf.groupby("block")["curvature_contrib"].agg(["mean", "std"]).reset_index()
                _plot_random_with_band(ax, grouped, xcol="block", ycol="curvature_contrib")
            else:
                grouped = ddf.groupby("block")["curvature_contrib"].mean().reset_index()
                ax.plot(
                    grouped["block"],
                    grouped["curvature_contrib"],
                    marker=DISPLAY_MARKER[direction],
                    linestyle=DISPLAY_LINESTYLE[direction],
                    color=DISPLAY_COLOR[direction],
                    label=DISPLAY_NAME[direction],
                )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_title(f"{encoder.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both" if yscale == "symlog" else "major", yscale=yscale, linthresh=linthresh)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )
    plt.tight_layout(rect=[0.04, 0.07, 1.0, 0.88])
    _add_global_xy_labels(
        fig,
        xlabel=r"Transformer block $b$",
        ylabel=r"Block contribution to $\kappa_{\mathrm{full}}(u;t)$",
    )
    _save_png_pdf(fig, outdir / f"expE_block_curvature_full_t{epoch:02d}.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, _, global_df, block_df, _ = _load_bundle(args.logdir, args.run_name)
    plot_A_directional_curvature_over_time(global_df, args.outdir, yscale=args.yscale, linthresh=args.linthresh)

    epochs = sorted({int(x) for x in args.block_epochs.split(",") if x.strip()})
    available_epochs = sorted(set(int(x) for x in block_df["epoch"].unique().tolist()))
    for epoch in epochs:
        if epoch not in available_epochs:
            continue
        plot_block_curvature_for_epoch(block_df, args.outdir, epoch=epoch, yscale=args.yscale, linthresh=args.linthresh)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

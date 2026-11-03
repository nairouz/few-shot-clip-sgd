#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paper-style plotting utilities for Experiment 4 counterfactuals.

Expected files (produced by measure_experiment4_counterfactuals.py):
  - <run_name>_config.json
  - <run_name>_counterfactual_per_iteration.csv
  - <run_name>_counterfactual_per_block.csv
  - <run_name>_probe_per_iteration.csv

Figures emitted:
  A) exp4_counterfactual_alignment_per_iteration.(png|pdf)
  B) exp4_counterfactual_normratio_per_iteration.(png|pdf)
  C) exp4_counterfactual_alignment_per_block.(png|pdf)
  D) exp4_counterfactual_normratio_per_block.(png|pdf)
  E) exp4_probe_delta_few_ref.(png|pdf)
  F) exp4_probe_delta_test.(png|pdf)
"""

from __future__ import annotations

import argparse
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

COL_RED = "#d62728"   # u_ff
COL_BLUE = "#1f77b4"  # u_fr
COL_GREEN = "#2ca02c" # u_f0
COL_ORANGE = "#ff7f0e"# u_flag
COL_GREY = "#7f7f7f"  # u_rr or baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()



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



def _group_curve(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("global_step", sort=True)[value_col]
        .agg(med="median", p25=lambda x: np.percentile(x, 25), p75=lambda x: np.percentile(x, 75))
        .reset_index()
    )



def _group_curve_by_block(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("block", sort=True)[value_col]
        .agg(med="median", p25=lambda x: np.percentile(x, 25), p75=lambda x: np.percentile(x, 75))
        .reset_index()
    )



def plot_alignment_per_iteration(df_iter: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    specs = [
        ("cos_u_ff_u_rr", COL_RED, r"$\cos(u_{ff},u_{rr})$"),
        ("cos_u_fr_u_rr", COL_BLUE, r"$\cos(u_{fr},u_{rr})$"),
        ("cos_u_f0_u_rr", COL_GREEN, r"$\cos(u_{f0},u_{rr})$"),
        ("cos_u_flag_u_rr", COL_ORANGE, r"$\cos(u_{f,t-\tau},u_{rr})$"),
    ]
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_iter[(df_iter["scope"] == "encoder") & (df_iter["encoder"] == enc)].copy()
        for col, color, label in specs:
            grp = _group_curve(sub, col)
            ax.plot(grp["global_step"], grp["med"], marker="o", color=color, label=label)
            ax.fill_between(grp["global_step"], grp["p25"], grp["p75"], alpha=0.18, color=color)
        ax.axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Cosine")
    _save_png_pdf(fig, outdir / "exp4_counterfactual_alignment_per_iteration.png")
    plt.close(fig)



def plot_normratio_per_iteration(df_iter: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    specs = [
        ("normratio_u_ff", COL_RED, r"$\|u_{ff}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_fr", COL_BLUE, r"$\|u_{fr}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_f0", COL_GREEN, r"$\|u_{f0}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_flag", COL_ORANGE, r"$\|u_{f,t-\tau}\|_2 / \|u_{rr}\|_2$"),
    ]
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_iter[(df_iter["scope"] == "encoder") & (df_iter["encoder"] == enc)].copy()
        for col, color, label in specs:
            grp = _group_curve(sub, col)
            ax.plot(grp["global_step"], grp["med"], marker="o", color=color, label=label)
            ax.fill_between(grp["global_step"], grp["p25"], grp["p75"], alpha=0.18, color=color)
        ax.axhline(1.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Norm ratio (log)")
    _save_png_pdf(fig, outdir / "exp4_counterfactual_normratio_per_iteration.png")
    plt.close(fig)



def plot_alignment_per_block(df_block: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    specs = [
        ("cos_u_ff_u_rr", COL_RED, r"$\cos(u_{ff},u_{rr})$"),
        ("cos_u_fr_u_rr", COL_BLUE, r"$\cos(u_{fr},u_{rr})$"),
        ("cos_u_f0_u_rr", COL_GREEN, r"$\cos(u_{f0},u_{rr})$"),
        ("cos_u_flag_u_rr", COL_ORANGE, r"$\cos(u_{f,t-\tau},u_{rr})$"),
    ]
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_block[(df_block["scope"] == "block") & (df_block["encoder"] == enc)].copy()
        for col, color, label in specs:
            grp = _group_curve_by_block(sub, col)
            x = grp["block"].to_numpy(dtype=float)
            ax.plot(x, grp["med"], marker="o", color=color, label=label)
            ax.fill_between(x, grp["p25"], grp["p75"], alpha=0.18, color=color)
        ax.axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel="Cosine")
    _save_png_pdf(fig, outdir / "exp4_counterfactual_alignment_per_block.png")
    plt.close(fig)



def plot_normratio_per_block(df_block: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    specs = [
        ("normratio_u_ff", COL_RED, r"$\|u_{ff}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_fr", COL_BLUE, r"$\|u_{fr}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_f0", COL_GREEN, r"$\|u_{f0}\|_2 / \|u_{rr}\|_2$"),
        ("normratio_u_flag", COL_ORANGE, r"$\|u_{f,t-\tau}\|_2 / \|u_{rr}\|_2$"),
    ]
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_block[(df_block["scope"] == "block") & (df_block["encoder"] == enc)].copy()
        for col, color, label in specs:
            grp = _group_curve_by_block(sub, col)
            x = grp["block"].to_numpy(dtype=float)
            ax.plot(x, grp["med"], marker="o", color=color, label=label)
            ax.fill_between(x, grp["p25"], grp["p75"], alpha=0.18, color=color)
        ax.axhline(1.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel="Norm ratio (log)")
    _save_png_pdf(fig, outdir / "exp4_counterfactual_normratio_per_block.png")
    plt.close(fig)



def plot_probe_delta_few_ref(df_probe: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharex=True)
    specs = [
        ("delta_few_loss_u_ff", COL_RED, "u_ff"),
        ("delta_few_loss_u_fr", COL_BLUE, "u_fr"),
        ("delta_few_loss_u_f0", COL_GREEN, "u_f0"),
        ("delta_few_loss_u_flag", COL_ORANGE, "u_f,t-tau"),
        ("delta_few_loss_u_rr", COL_GREY, "u_rr"),
    ]
    for col, color, label in specs:
        axes[0].plot(df_probe["global_step"], df_probe[col], marker="o", color=color, label=label)
    axes[0].axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
    axes[0].set_title("Few-shot loss change")
    _style_axis(axes[0])

    specs_ref = [
        ("delta_ref_loss_u_ff", COL_RED, "u_ff"),
        ("delta_ref_loss_u_fr", COL_BLUE, "u_fr"),
        ("delta_ref_loss_u_f0", COL_GREEN, "u_f0"),
        ("delta_ref_loss_u_flag", COL_ORANGE, "u_f,t-tau"),
        ("delta_ref_loss_u_rr", COL_GREY, "u_rr"),
    ]
    for col, color, label in specs_ref:
        axes[1].plot(df_probe["global_step"], df_probe[col], marker="o", color=color, label=label)
    axes[1].axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
    axes[1].set_title("Reference loss change")
    _style_axis(axes[1])
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Probe loss change")
    _save_png_pdf(fig, outdir / "exp4_probe_delta_few_ref.png")
    plt.close(fig)



def plot_probe_delta_test(df_probe: pd.DataFrame, outdir: Path) -> None:
    if "delta_test_loss_u_ff" not in df_probe.columns:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    specs = [
        ("delta_test_loss_u_ff", COL_RED, "u_ff"),
        ("delta_test_loss_u_fr", COL_BLUE, "u_fr"),
        ("delta_test_loss_u_f0", COL_GREEN, "u_f0"),
        ("delta_test_loss_u_flag", COL_ORANGE, "u_f,t-tau"),
        ("delta_test_loss_u_rr", COL_GREY, "u_rr"),
    ]
    any_finite = False
    for col, color, label in specs:
        if col not in df_probe.columns:
            continue
        y = df_probe[col].to_numpy(dtype=float)
        if np.isfinite(y).any():
            any_finite = True
        ax.plot(df_probe["global_step"], y, marker="o", color=color, label=label)
    if not any_finite:
        plt.close(fig)
        return
    ax.axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.legend(frameon=False, ncol=2)
    _style_axis(ax)
    plt.tight_layout(rect=[0.06, 0.08, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Test loss change")
    _save_png_pdf(fig, outdir / "exp4_probe_delta_test.png")
    plt.close(fig)



def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_iter = pd.read_csv(args.logdir / f"{args.run_name}_counterfactual_per_iteration.csv")
    df_block = pd.read_csv(args.logdir / f"{args.run_name}_counterfactual_per_block.csv")
    df_probe = pd.read_csv(args.logdir / f"{args.run_name}_probe_per_iteration.csv")

    plot_alignment_per_iteration(df_iter, args.outdir)
    plot_normratio_per_iteration(df_iter, args.outdir)
    plot_alignment_per_block(df_block, args.outdir)
    plot_normratio_per_block(df_block, args.outdir)
    plot_probe_delta_few_ref(df_probe, args.outdir)
    plot_probe_delta_test(df_probe, args.outdir)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

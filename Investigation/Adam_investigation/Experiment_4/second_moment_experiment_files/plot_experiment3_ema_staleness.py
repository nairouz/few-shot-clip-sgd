#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Paper-style plotting utilities for Experiment 3 (EMA staleness audit).

Expected files (produced by measure_experiment3_ema_staleness.py):
  - <run_name>_config.json
  - <run_name>_per_iteration.csv
  - <run_name>_per_block.csv

Figures emitted:
  A) exp3_v_normratio_per_iteration
  B) exp3_v_mismatch_per_iteration
  C) exp3_precond_alignment_per_iteration
  D) exp3_precond_normratio_per_iteration
  E) exp3_v_mismatch_per_block
  F) exp3_precond_alignment_per_block
"""

from __future__ import annotations

import argparse
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

SRC_COLORS = {
    "instant": "#1f77b4",
    "batch": "#d62728",
    "ema": "#2ca02c",
}
GREY = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def load_bundle(logdir: Path, run_name: str):
    cfg_path = logdir / f"{run_name}_config.json"
    iter_csv = logdir / f"{run_name}_per_iteration.csv"
    block_csv = logdir / f"{run_name}_per_block.csv"
    missing = [p for p in [cfg_path, iter_csv, block_csv] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required log files:\n" + "\n".join(str(p) for p in missing))
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg, pd.read_csv(iter_csv), pd.read_csv(block_csv)


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


def _curve(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("global_step", sort=True)[value_col]
        .agg(med="median", p25=lambda x: np.percentile(x, 25), p75=lambda x: np.percentile(x, 75))
        .reset_index()
    )


def _curve_block(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("block", sort=True)[value_col]
        .agg(med="median", p25=lambda x: np.percentile(x, 25), p75=lambda x: np.percentile(x, 75))
        .reset_index()
    )


def _plot_sources_per_iteration(df_iter: pd.DataFrame, outdir: Path, *, col: str, ylabel: str, filename: str, yscale: str = "linear", add_oracle: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_enc = df_iter[(df_iter["scope"] == "encoder") & (df_iter["encoder"] == enc)].copy()
        if sub_enc.empty:
            continue
        oracle_drawn = False
        for src in ["instant", "batch", "ema"]:
            sub = sub_enc[sub_enc["moment_source"] == src].sort_values("global_step")
            if sub.empty:
                continue
            grp = _curve(sub, col)
            ax.plot(grp["global_step"], grp["med"], marker="o", color=SRC_COLORS[src], label=src)
            ax.fill_between(grp["global_step"], grp["p25"], grp["p75"], alpha=0.18, color=SRC_COLORS[src])
            if add_oracle and not oracle_drawn:
                grp_or = _curve(sub, "oracle_cos_vs_ref" if "cos" in col else "oracle_over_ref_l2")
                ax.plot(grp_or["global_step"], grp_or["med"], color=GREY, linestyle="--", label="oracle baseline")
                oracle_drawn = True
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_yscale(yscale)
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=ylabel)
    _save_png_pdf(fig, outdir / filename)
    plt.close(fig)


def _plot_sources_per_block(df_block: pd.DataFrame, outdir: Path, *, col: str, ylabel: str, filename: str, yscale: str = "linear", add_oracle: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_enc = df_block[(df_block["scope"] == "block") & (df_block["encoder"] == enc)].copy()
        if sub_enc.empty:
            continue
        oracle_drawn = False
        for src in ["instant", "batch", "ema"]:
            sub = sub_enc[sub_enc["moment_source"] == src].copy()
            if sub.empty:
                continue
            grp = _curve_block(sub, col)
            x = grp["block"].to_numpy(dtype=float)
            ax.plot(x, grp["med"], marker="o", color=SRC_COLORS[src], label=src)
            ax.fill_between(x, grp["p25"], grp["p75"], alpha=0.18, color=SRC_COLORS[src])
            if add_oracle and not oracle_drawn:
                grp_or = _curve_block(sub, "oracle_cos_vs_ref" if "cos" in col else "oracle_over_ref_l2")
                ax.plot(grp_or["block"], grp_or["med"], color=GREY, linestyle="--", label="oracle baseline")
                oracle_drawn = True
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_yscale(yscale)
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=ylabel)
    _save_png_pdf(fig, outdir / filename)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _, df_iter, df_block = load_bundle(args.logdir, args.run_name)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # convert to ratio on the fly by dividing with ref_v_l2
    for df in (df_iter, df_block):
        df["source_v_norm_ratio"] = df["source_v_l2"] / df["ref_v_l2"]
    _plot_sources_per_iteration(
        df_iter,
        args.outdir,
        col="source_v_norm_ratio",
        ylabel=r"$\|v_{source}\|_2 / \|v_{ref}\|_2$ (log)",
        filename="exp3_v_normratio_per_iteration.png",
        yscale="log",
    )
    _plot_sources_per_iteration(
        df_iter,
        args.outdir,
        col="v_mean_abs_log10_ratio",
        ylabel=r"mean $|\log_{10}(v_{source}/v_{ref})|$",
        filename="exp3_v_mismatch_per_iteration.png",
    )
    _plot_sources_per_iteration(
        df_iter,
        args.outdir,
        col="source_cos_vs_ref",
        ylabel=r"$\cos(u_{source},u_{rr})$",
        filename="exp3_precond_alignment_per_iteration.png",
        add_oracle=True,
    )
    _plot_sources_per_iteration(
        df_iter,
        args.outdir,
        col="source_over_ref_l2",
        ylabel=r"$\|u_{source}\|_2 / \|u_{rr}\|_2$ (log)",
        filename="exp3_precond_normratio_per_iteration.png",
        yscale="log",
        add_oracle=True,
    )
    _plot_sources_per_block(
        df_block,
        args.outdir,
        col="v_mean_abs_log10_ratio",
        ylabel=r"mean $|\log_{10}(v_{source}/v_{ref})|$",
        filename="exp3_v_mismatch_per_block.png",
    )
    _plot_sources_per_block(
        df_block,
        args.outdir,
        col="source_cos_vs_ref",
        ylabel=r"$\cos(u_{source},u_{rr})$",
        filename="exp3_precond_alignment_per_block.png",
        add_oracle=True,
    )

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

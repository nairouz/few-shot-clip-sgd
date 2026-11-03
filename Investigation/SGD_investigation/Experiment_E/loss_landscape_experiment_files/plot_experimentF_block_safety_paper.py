#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-ready summary plot for Experiment F.

This script aggregates the blockwise summaries across the checkpoint set
T={0,1,2,5,10,20,30} and produces the summary figure used in the paper.

Inputs (expected in --logdir):
  - <run_name>_block_summary.csv
  - <run_name>_config.json

Output:
  - expF_block_safety_summary.(png|pdf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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

COL_VISION = "#1f77b4"
COL_TEXT = "#ff7f0e"
LINE_VISION = "-"
LINE_TEXT = "--"
MARK_VISION = "o"
MARK_TEXT = "s"

DATASET_ORDER = ["eurosat", "oxford_pets", "dtd"]
DATASET_TITLE = {
    "eurosat": "EuroSAT",
    "oxford_pets": "Oxford-Pets",
    "dtd": "DTD",
}
ENC_STYLE = {
    "vision": dict(color=COL_VISION, linestyle=LINE_VISION, marker=MARK_VISION, label="Vision"),
    "text": dict(color=COL_TEXT, linestyle=LINE_TEXT, marker=MARK_TEXT, label="Text"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument(
        "--run_names",
        type=str,
        default=None,
        help="Optional comma-separated run names. If omitted, the script scans the logdir for Experiment F runs.",
    )
    return p.parse_args()


def _style_axis(ax: plt.Axes, *, which: str = "major") -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0, which=which)
    ax.tick_params(axis="both", pad=4)


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel_left_top: str, ylabel_left_bottom: str) -> None:
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.74, ylabel_left_top, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])
    fig.text(0.02, 0.27, ylabel_left_bottom, ha="center", va="center", rotation=90, fontsize=mpl.rcParams["axes.labelsize"])


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def _discover_runs(logdir: Path) -> list[str]:
    runs = []
    for cfg in sorted(logdir.glob("experimentF_*_config.json")):
        stem = cfg.name[:-12]
        runs.append(stem)
    return runs


def _dataset_from_run(run_name: str) -> str:
    m = re.search(r"dataset_([^_]+(?:_[^_]+)?)_shots", run_name)
    if not m:
        raise ValueError(f"Could not infer dataset from run name: {run_name}")
    return m.group(1)


def _load_aggregate(logdir: Path, run_name: str) -> pd.DataFrame:
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    df = pd.read_csv(logdir / f"{run_name}_block_summary.csv")
    out = (
        df.groupby(["encoder", "block"], as_index=False)
          .agg(
              mean_full_delta=("full_delta_at1", "mean"),
              mean_gap=("gap_full_minus_few_at1", "mean"),
              mean_full_alpha=("full_alpha_min", "mean"),
              pos_full_delta_frac=("full_delta_at1", lambda x: float(np.mean(np.asarray(x) > 0.0))),
          )
    )
    out["dataset"] = cfg["dataset"]
    out["run_name"] = run_name
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_names = [x.strip() for x in args.run_names.split(",")] if args.run_names else _discover_runs(args.logdir)
    if not run_names:
        raise SystemExit(f"No Experiment F runs found in {args.logdir}")

    frames = []
    for run_name in run_names:
        frames.append(_load_aggregate(args.logdir, run_name))
    df = pd.concat(frames, ignore_index=True)

    available = {d for d in df["dataset"].unique()}
    datasets = [d for d in DATASET_ORDER if d in available]
    if not datasets:
        raise SystemExit("No recognized datasets found for Experiment F summary plot.")

    fig, axes = plt.subplots(2, len(datasets), figsize=(15.0, 6.5), sharex=True)
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].copy()
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        for enc in ["vision", "text"]:
            ss = sub[sub["encoder"] == enc].sort_values("block")
            sty = ENC_STYLE[enc]
            ax_top.plot(ss["block"], ss["mean_full_delta"], **sty)
            ax_bot.plot(ss["block"], ss["mean_gap"], **sty)

        ax_top.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        ax_bot.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        ax_top.set_title(DATASET_TITLE.get(ds, ds))
        _style_axis(ax_top)
        _style_axis(ax_bot)
        ax_top.set_xlabel("")
        ax_bot.set_xlabel("")
        ax_bot.set_xticks(range(12))
        ax_bot.set_xlim(-0.3, 11.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.2,
    )
    plt.tight_layout(rect=[0.05, 0.06, 1.0, 0.92])
    _add_global_xy_labels(
        fig,
        xlabel=r"Transformer block $\ell$",
        ylabel_left_top=r"$\overline{\Delta}_{\mathrm{full}}^{(\ell)}(1)$",
        ylabel_left_bottom=r"$\overline{\Gamma}^{(\ell)}(1)$",
    )
    _save_png_pdf(fig, args.outdir / "expF_block_safety_summary.png")
    plt.close(fig)
    print(f"Saved Experiment F paper figure to: {args.outdir}")


if __name__ == "__main__":
    main()

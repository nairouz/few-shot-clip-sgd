#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment 6: compare SGD vs Adam feature-drift audits.

Required inputs:
  - <run_name_sgd>_config.json
  - <run_name_sgd>_per_iteration.csv
  - <run_name_sgd>_per_block.csv
  - <run_name_adam>_config.json
  - <run_name_adam>_per_iteration.csv
  - <run_name_adam>_per_block.csv

Figures emitted:
  A) exp6_param_drift_compare.(png|pdf)
  B) exp6_block_drift_compare.(png|pdf)
  C) exp6_feature_drift_compare.(png|pdf)
  D) exp6_logit_margin_compare.(png|pdf)
  E) exp6_accuracy_loss_compare.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Style (matched to Experiment 2)
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

COL_SGD = "#1f77b4"
COL_ADAM = "#d62728"
COL_SPLIT_FEW = "#1f77b4"
COL_SPLIT_REF = "#2ca02c"
COL_SPLIT_TEST = "#ff7f0e"
COL_GREY = "#7f7f7f"


# ============================================================================
# CLI / I/O
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing the feature-drift log bundles.")
    parser.add_argument("--run_name_sgd", type=str, required=True, help="Run name for the SGD trajectory.")
    parser.add_argument("--run_name_adam", type=str, required=True, help="Run name for the Adam trajectory.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory where the plots will be saved.")
    return parser.parse_args()


def _load_run(logdir: Path, run_name: str) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    cfg_path = logdir / f"{run_name}_config.json"
    iter_csv = logdir / f"{run_name}_per_iteration.csv"
    block_csv = logdir / f"{run_name}_per_block.csv"
    missing = [p for p in [cfg_path, iter_csv, block_csv] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required log files:\n" + "\n".join(str(p) for p in missing))
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg, pd.read_csv(iter_csv), pd.read_csv(block_csv)


def _assert_compatible(cfg_sgd: Dict[str, object], cfg_adam: Dict[str, object]) -> None:
    keys = ["dataset", "shots", "backbone", "position", "encoder", "ref_mode"]
    mismatches = []
    for k in keys:
        if cfg_sgd.get(k) != cfg_adam.get(k):
            mismatches.append((k, cfg_sgd.get(k), cfg_adam.get(k)))
    if mismatches:
        msg = "Experiment 6 requires comparable SGD/Adam runs. Mismatches found:\n"
        msg += "\n".join(f"  {k}: SGD={a!r}, Adam={b!r}" for k, a, b in mismatches)
        raise ValueError(msg)


# ============================================================================
# Common helpers
# ============================================================================

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



def _final_block_df(df: pd.DataFrame) -> pd.DataFrame:
    max_step = int(df["global_step"].max())
    return df[df["global_step"] == max_step].copy().sort_values(["encoder", "block"])



def _plot_split_curves(ax: plt.Axes, df_sgd: pd.DataFrame, df_adam: pd.DataFrame, *, y_prefix: str, title: str, ylabel_scale: str = "linear") -> None:
    mapping = {
        "few": ("Support", COL_SPLIT_FEW),
        "ref": ("Full train", COL_SPLIT_REF),
        "test": ("Test", COL_SPLIT_TEST),
    }
    for split, (label, color) in mapping.items():
        col = f"{split}_{y_prefix}"
        ax.plot(df_sgd["global_step"], df_sgd[col], marker="o", color=color, linestyle="-", label=f"SGD: {label}")
        ax.plot(df_adam["global_step"], df_adam[col], marker="o", color=color, linestyle="--", label=f"Adam: {label}")
    if ylabel_scale == "log":
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
    _style_axis(ax, which="both" if ylabel_scale == "log" else "major")


# ============================================================================
# Plot A: parameter drift vs iteration
# ============================================================================

def plot_A_param_drift_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        col = f"{enc}_param_drift_rel"
        ax.plot(df_sgd["global_step"], df_sgd[col], marker="o", color=COL_SGD, label="SGD")
        ax.plot(df_adam["global_step"], df_adam[col], marker="o", color=COL_ADAM, label="Adam")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"Parameter drift $\|\Theta_t-\Theta_0\|_2 / \|\Theta_0\|_2$")
    _save_png_pdf(fig, outdir / "exp6_param_drift_compare.png")
    plt.close(fig)


# ============================================================================
# Plot B: final per-block parameter drift
# ============================================================================

def plot_B_block_drift_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    final_sgd = _final_block_df(df_sgd)
    final_adam = _final_block_df(df_adam)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = final_sgd[final_sgd["encoder"] == enc].sort_values("block")
        sub_adam = final_adam[final_adam["encoder"] == enc].sort_values("block")
        ax.plot(sub_sgd["block"], sub_sgd["param_drift_rel"], marker="o", color=COL_SGD, label="SGD")
        ax.plot(sub_adam["block"], sub_adam["param_drift_rel"], marker="o", color=COL_ADAM, label="Adam")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=r"Final block drift $\|\Theta_t-\Theta_0\|_2 / \|\Theta_0\|_2$")
    _save_png_pdf(fig, outdir / "exp6_block_drift_compare.png")
    plt.close(fig)


# ============================================================================
# Plot C: image/text feature drift
# ============================================================================

def plot_C_feature_drift_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=False)

    _plot_split_curves(
        axes[0], df_sgd, df_adam,
        y_prefix="image_feature_drift",
        title="Image-feature drift",
    )
    axes[0].legend(frameon=False, ncol=1)

    axes[1].plot(df_sgd["global_step"], df_sgd["text_feature_drift"], marker="o", color=COL_SGD, label="SGD")
    axes[1].plot(df_adam["global_step"], df_adam["text_feature_drift"], marker="o", color=COL_ADAM, label="Adam")
    axes[1].set_title("Text-feature drift")
    axes[1].set_xlabel("")
    _style_axis(axes[1])
    axes[1].legend(frameon=False)

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"Drift to init $= 1 - \mathrm{cos}(\cdot, \cdot_0)$")
    _save_png_pdf(fig, outdir / "exp6_feature_drift_compare.png")
    plt.close(fig)


# ============================================================================
# Plot D: logit drift and margin drift
# ============================================================================

def plot_D_logit_margin_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=False)

    _plot_split_curves(
        axes[0], df_sgd, df_adam,
        y_prefix="logit_delta_abs_mean",
        title="Mean absolute logit drift",
    )
    axes[0].legend(frameon=False, ncol=1)

    _plot_split_curves(
        axes[1], df_sgd, df_adam,
        y_prefix="gt_margin_delta_abs_mean",
        title="Ground-truth margin drift",
    )

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Drift magnitude")
    _save_png_pdf(fig, outdir / "exp6_logit_margin_compare.png")
    plt.close(fig)


# ============================================================================
# Plot E: support / full-train / test accuracy and loss
# ============================================================================

def plot_E_accuracy_loss_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=False)

    _plot_split_curves(
        axes[0], df_sgd, df_adam,
        y_prefix="acc",
        title="Accuracy",
    )
    axes[0].legend(frameon=False, ncol=1)

    _plot_split_curves(
        axes[1], df_sgd, df_adam,
        y_prefix="loss",
        title="Loss",
    )

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Metric value")
    _save_png_pdf(fig, outdir / "exp6_accuracy_loss_compare.png")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg_sgd, df_sgd_iter, df_sgd_block = _load_run(args.logdir, args.run_name_sgd)
    cfg_adam, df_adam_iter, df_adam_block = _load_run(args.logdir, args.run_name_adam)
    _assert_compatible(cfg_sgd, cfg_adam)

    plot_A_param_drift_compare(df_sgd_iter, df_adam_iter, args.outdir)
    plot_B_block_drift_compare(df_sgd_block, df_adam_block, args.outdir)
    plot_C_feature_drift_compare(df_sgd_iter, df_adam_iter, args.outdir)
    plot_D_logit_margin_compare(df_sgd_iter, df_adam_iter, args.outdir)
    plot_E_accuracy_loss_compare(df_sgd_iter, df_adam_iter, args.outdir)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

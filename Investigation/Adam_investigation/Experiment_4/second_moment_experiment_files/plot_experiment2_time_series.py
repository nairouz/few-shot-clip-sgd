#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Experiment 2: time-series numerator vs denominator collapse.

This script compares an SGD trajectory and an Adam trajectory using the log
bundles produced by ``measure_second_moment.py``.

Experiment 2 is specifically about *when* the Adam pathology emerges:
- Is the few-shot gradient already too small?
- Does it become small only after training starts?
- Does the second moment collapse faster than the gradient?
- Does Adam's own trajectory amplify the pathology relative to SGD?

The measurement code used here is exactly the same as Experiment 3. This script
only adds trajectory-comparison plots built from the existing CSV logs.

Required inputs:
  - <run_name_sgd>_config.json
  - <run_name_sgd>_per_iteration.csv
  - <run_name_adam>_config.json
  - <run_name_adam>_per_iteration.csv

Figures emitted:
  A) exp2_grad_alignment_compare.(png|pdf)
  B) exp2_num_den_ratio_compare.(png|pdf)
  C) exp2_precond_alignment_compare.(png|pdf)
  D) exp2_precond_normratio_compare.(png|pdf)
  E) exp2_loss_compare.(png|pdf)
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
# Style (matched to plot_second_moment.py)
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
COL_GRAD = "#1f77b4"
COL_V = "#d62728"
COL_UFF = "#d62728"
COL_UFR = "#1f77b4"
COL_GREY = "#7f7f7f"


# ============================================================================
# CLI / I/O
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing the second-moment log bundles.")
    parser.add_argument("--run_name_sgd", type=str, required=True, help="Run name for the SGD trajectory.")
    parser.add_argument("--run_name_adam", type=str, required=True, help="Run name for the Adam trajectory.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory where the plots will be saved.")
    return parser.parse_args()



def _load_run(logdir: Path, run_name: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    cfg_path = logdir / f"{run_name}_config.json"
    iter_csv = logdir / f"{run_name}_per_iteration.csv"
    missing = [p for p in [cfg_path, iter_csv] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required log files:\n" + "\n".join(str(p) for p in missing))
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    df_iter = pd.read_csv(iter_csv)
    return cfg, df_iter



def _assert_compatible(cfg_sgd: Dict[str, object], cfg_adam: Dict[str, object]) -> None:
    keys = ["dataset", "shots", "backbone", "position", "encoder", "ref_mode", "v_estimator"]
    mismatches = []
    for k in keys:
        if cfg_sgd.get(k) != cfg_adam.get(k):
            mismatches.append((k, cfg_sgd.get(k), cfg_adam.get(k)))
    if mismatches:
        msg = "Experiment 2 requires comparable SGD/Adam runs. Mismatches found:\n"
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



def _prepare_iter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["scope"] == "encoder"].copy().sort_values(["encoder", "global_step"])
    df["grad_norm_ratio"] = df["few_grad_l2"] / df["ref_grad_l2"]
    df["v_norm_ratio"] = df["few_v_l2"] / df["ref_v_l2"]
    df["u_ff_norm_ratio"] = df["adam_l2_fewfew"] / df["adam_l2_refref"]
    df["u_fr_norm_ratio"] = df["adam_l2_fewref"] / df["adam_l2_refref"]
    return df



def _loss_df(df: pd.DataFrame) -> pd.DataFrame:
    # few_loss / ref_loss are duplicated for vision/text, so keeping the vision
    # rows gives a clean one-row-per-step loss table.
    sub = df[df["encoder"] == "vision"].copy().sort_values("global_step")
    return sub[["global_step", "few_loss", "ref_loss"]]


# ============================================================================
# Plot A: gradient alignment vs iteration
# ============================================================================

def plot_A_grad_alignment_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        sub_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        ax.plot(sub_sgd["global_step"], sub_sgd["grad_cosine"], marker="o", color=COL_SGD, label=r"SGD: $\cos(m_{few}, m_{ref})$")
        ax.plot(sub_adam["global_step"], sub_adam["grad_cosine"], marker="o", color=COL_ADAM, label=r"Adam: $\cos(m_{few}, m_{ref})$")
        ax.plot(sub_sgd["global_step"], sub_sgd["grad_sign_agreement"], marker="o", linestyle="--", color=COL_SGD, alpha=0.9, label=r"SGD: sign agreement")
        ax.plot(sub_adam["global_step"], sub_adam["grad_sign_agreement"], marker="o", linestyle="--", color=COL_ADAM, alpha=0.9, label=r"Adam: sign agreement")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Alignment")
    _save_png_pdf(fig, outdir / "exp2_grad_alignment_compare.png")
    plt.close(fig)


# ============================================================================
# Plot B: numerator vs denominator norm ratios
# ============================================================================

def plot_B_num_den_ratio_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        sub_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        ax.plot(sub_sgd["global_step"], sub_sgd["grad_norm_ratio"], marker="o", linestyle="-", color=COL_GRAD, label=r"SGD: $\|m_{few}\|_2 / \|m_{ref}\|_2$")
        ax.plot(sub_sgd["global_step"], sub_sgd["v_norm_ratio"], marker="o", linestyle="--", color=COL_V, label=r"SGD: $\|v_{few}\|_2 / \|v_{ref}\|_2$")
        ax.plot(sub_adam["global_step"], sub_adam["grad_norm_ratio"], marker="o", linestyle="-", color=COL_GRAD, alpha=0.45, label=r"Adam: $\|m_{few}\|_2 / \|m_{ref}\|_2$")
        ax.plot(sub_adam["global_step"], sub_adam["v_norm_ratio"], marker="o", linestyle="--", color=COL_V, alpha=0.45, label=r"Adam: $\|v_{few}\|_2 / \|v_{ref}\|_2$")
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Norm ratio (log)")
    _save_png_pdf(fig, outdir / "exp2_num_den_ratio_compare.png")
    plt.close(fig)


# ============================================================================
# Plot C: preconditioned-direction alignment
# ============================================================================

def plot_C_precond_alignment_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        sub_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        ax.plot(sub_sgd["global_step"], sub_sgd["adam_cos_fewfew_vs_fewref"], marker="o", linestyle="--", color=COL_SGD, label=r"SGD: $\cos(u_{ff},u_{fr})$")
        ax.plot(sub_sgd["global_step"], sub_sgd["adam_cos_fewref_vs_refref"], marker="o", linestyle="-", color=COL_SGD, label=r"SGD: $\cos(u_{fr},u_{rr})$")
        ax.plot(sub_adam["global_step"], sub_adam["adam_cos_fewfew_vs_fewref"], marker="o", linestyle="--", color=COL_ADAM, label=r"Adam: $\cos(u_{ff},u_{fr})$")
        ax.plot(sub_adam["global_step"], sub_adam["adam_cos_fewref_vs_refref"], marker="o", linestyle="-", color=COL_ADAM, label=r"Adam: $\cos(u_{fr},u_{rr})$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Cosine")
    _save_png_pdf(fig, outdir / "exp2_precond_alignment_compare.png")
    plt.close(fig)


# ============================================================================
# Plot D: preconditioned-direction norm ratios
# ============================================================================

def plot_D_precond_normratio_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        sub_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        ax.plot(sub_sgd["global_step"], sub_sgd["u_ff_norm_ratio"], marker="o", linestyle="--", color=COL_SGD, label=r"SGD: $\|u_{ff}\|_2 / \|u_{rr}\|_2$")
        ax.plot(sub_sgd["global_step"], sub_sgd["u_fr_norm_ratio"], marker="o", linestyle="-", color=COL_SGD, label=r"SGD: $\|u_{fr}\|_2 / \|u_{rr}\|_2$")
        ax.plot(sub_adam["global_step"], sub_adam["u_ff_norm_ratio"], marker="o", linestyle="--", color=COL_ADAM, label=r"Adam: $\|u_{ff}\|_2 / \|u_{rr}\|_2$")
        ax.plot(sub_adam["global_step"], sub_adam["u_fr_norm_ratio"], marker="o", linestyle="-", color=COL_ADAM, label=r"Adam: $\|u_{fr}\|_2 / \|u_{rr}\|_2$")
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Norm ratio (log)")
    _save_png_pdf(fig, outdir / "exp2_precond_normratio_compare.png")
    plt.close(fig)


# ============================================================================
# Plot E: support/reference loss comparison
# ============================================================================

def plot_E_loss_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, outdir: Path) -> None:
    loss_sgd = _loss_df(df_sgd)
    loss_adam = _loss_df(df_adam)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=False)

    axes[0].plot(loss_sgd["global_step"], loss_sgd["few_loss"], marker="o", color=COL_SGD, label="SGD")
    axes[0].plot(loss_adam["global_step"], loss_adam["few_loss"], marker="o", color=COL_ADAM, label="Adam")
    axes[0].set_title("Support loss")
    axes[0].set_xlabel("")
    _style_axis(axes[0])

    axes[1].plot(loss_sgd["global_step"], loss_sgd["ref_loss"], marker="o", color=COL_SGD, label="SGD")
    axes[1].plot(loss_adam["global_step"], loss_adam["ref_loss"], marker="o", color=COL_ADAM, label="Adam")
    axes[1].set_title("Reference loss")
    axes[1].set_xlabel("")
    _style_axis(axes[1])

    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Loss")
    _save_png_pdf(fig, outdir / "exp2_loss_compare.png")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg_sgd, df_sgd = _load_run(args.logdir, args.run_name_sgd)
    cfg_adam, df_adam = _load_run(args.logdir, args.run_name_adam)
    _assert_compatible(cfg_sgd, cfg_adam)

    df_sgd = _prepare_iter_df(df_sgd)
    df_adam = _prepare_iter_df(df_adam)

    plot_A_grad_alignment_compare(df_sgd, df_adam, args.outdir)
    plot_B_num_den_ratio_compare(df_sgd, df_adam, args.outdir)
    plot_C_precond_alignment_compare(df_sgd, df_adam, args.outdir)
    plot_D_precond_normratio_compare(df_sgd, df_adam, args.outdir)
    plot_E_loss_compare(df_sgd, df_adam, args.outdir)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

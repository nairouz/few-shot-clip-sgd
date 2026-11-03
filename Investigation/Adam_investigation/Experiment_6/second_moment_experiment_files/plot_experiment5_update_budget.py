#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 5: update-budget matched Adam vs SGD.

This script compares three trajectories built from the log bundles emitted by
``measure_second_moment.py``:

  1) SGD baseline,
  2) raw Adam baseline,
  3) Adam with the learning rate rescaled once at step 0 so that its first-step
     global relative update norm matches the SGD baseline.

The question is:

    Is Adam bad mainly because of wrong scale, or because of wrong geometry?

Required inputs:
  - <run_name_sgd>_config.json
  - <run_name_sgd>_per_iteration.csv
  - <run_name_sgd>_per_block.csv
  - <run_name_adam>_config.json
  - <run_name_adam>_per_iteration.csv
  - <run_name_adam>_per_block.csv
  - <run_name_adam_matched>_config.json
  - <run_name_adam_matched>_per_iteration.csv
  - <run_name_adam_matched>_per_block.csv

Figures emitted:
  A) exp5_global_update_budget_compare.(png|pdf)
  B) exp5_encoder_update_budget_compare.(png|pdf)
  C) exp5_block_step0_update_budget_compare.(png|pdf)
  D) exp5_geometry_compare.(png|pdf)
  E) exp5_loss_compare.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# Style (matched to Experiment 2 / plot_second_moment.py)
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
COL_MATCH = "#2ca02c"
COL_GREY = "#7f7f7f"


# ============================================================================
# CLI / I/O
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing the second-moment log bundles.")
    parser.add_argument("--run_name_sgd", type=str, required=True, help="Run name for the SGD trajectory.")
    parser.add_argument("--run_name_adam", type=str, required=True, help="Run name for the raw Adam trajectory.")
    parser.add_argument("--run_name_adam_matched", type=str, required=True, help="Run name for the step-0 budget-matched Adam trajectory.")
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



def _assert_compatible(cfg_a: Dict[str, object], cfg_b: Dict[str, object], tag_a: str, tag_b: str) -> None:
    keys = ["dataset", "shots", "backbone", "position", "encoder", "ref_mode", "v_estimator"]
    mismatches = []
    for k in keys:
        if cfg_a.get(k) != cfg_b.get(k):
            mismatches.append((k, cfg_a.get(k), cfg_b.get(k)))
    if mismatches:
        msg = f"Experiment 5 requires comparable runs. Mismatches found between {tag_a} and {tag_b}:\n"
        msg += "\n".join(f"  {k}: {tag_a}={a!r}, {tag_b}={b!r}" for k, a, b in mismatches)
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
    required = [
        "scope", "encoder", "global_step", "few_loss", "ref_loss",
        "rel_update_norm_global", "rel_update_norm_scope",
        "adam_cos_fewref_vs_refref", "adam_cos_fewfew_vs_fewref",
        "optimizer_lr", "requested_lr", "lr_match_scale",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "This plotting script requires Experiment 5 logs with update-budget columns. Missing columns: "
            + ", ".join(missing)
        )
    df = df.copy()
    df = df[df["scope"] == "encoder"].copy().sort_values(["encoder", "global_step"])
    return df



def _prepare_block_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["scope", "encoder", "block", "global_step", "rel_update_norm_scope"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "This plotting script requires Experiment 5 block logs with update-budget columns. Missing columns: "
            + ", ".join(missing)
        )
    return df[(df["scope"] == "block")].copy().sort_values(["encoder", "block", "global_step"])



def _global_budget_df(df: pd.DataFrame) -> pd.DataFrame:
    # rel_update_norm_global is duplicated for vision/text, so keep vision rows.
    sub = df[df["encoder"] == "vision"].copy().sort_values("global_step")
    return sub[["global_step", "rel_update_norm_global", "few_loss", "ref_loss", "optimizer_lr", "requested_lr", "lr_match_scale"]]


# ============================================================================
# Plot A: global update-budget comparison
# ============================================================================

def plot_A_global_update_budget_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, df_match: pd.DataFrame, outdir: Path) -> None:
    g_sgd = _global_budget_df(df_sgd)
    g_adam = _global_budget_df(df_adam)
    g_match = _global_budget_df(df_match)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.8))
    ax.plot(g_sgd["global_step"], g_sgd["rel_update_norm_global"], marker="o", color=COL_SGD, label="SGD")
    ax.plot(g_adam["global_step"], g_adam["rel_update_norm_global"], marker="o", color=COL_ADAM, label="Adam")
    ax.plot(g_match["global_step"], g_match["rel_update_norm_global"], marker="o", color=COL_MATCH, label="Adam (step-0 matched)")
    ax.set_yscale("log")
    ax.set_xlabel("")
    _style_axis(ax, which="both")
    ax.legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"Global relative update norm $\|\Delta\Theta_t\|_2 / \|\Theta_t\|_2$")
    _save_png_pdf(fig, outdir / "exp5_global_update_budget_compare.png")
    plt.close(fig)


# ============================================================================
# Plot B: encoder-level update-budget comparison
# ============================================================================

def plot_B_encoder_update_budget_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, df_match: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        s_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        s_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        s_match = df_match[df_match["encoder"] == enc].sort_values("global_step")
        ax.plot(s_sgd["global_step"], s_sgd["rel_update_norm_scope"], marker="o", color=COL_SGD, label="SGD")
        ax.plot(s_adam["global_step"], s_adam["rel_update_norm_scope"], marker="o", color=COL_ADAM, label="Adam")
        ax.plot(s_match["global_step"], s_match["rel_update_norm_scope"], marker="o", color=COL_MATCH, label="Adam (step-0 matched)")
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"Encoder relative update norm")
    _save_png_pdf(fig, outdir / "exp5_encoder_update_budget_compare.png")
    plt.close(fig)


# ============================================================================
# Plot C: first-step blockwise update-budget comparison
# ============================================================================

def plot_C_block_step0_update_budget_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, df_match: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        s_sgd = df_sgd[(df_sgd["encoder"] == enc) & (df_sgd["global_step"] == 0)].sort_values("block")
        s_adam = df_adam[(df_adam["encoder"] == enc) & (df_adam["global_step"] == 0)].sort_values("block")
        s_match = df_match[(df_match["encoder"] == enc) & (df_match["global_step"] == 0)].sort_values("block")
        ax.plot(s_sgd["block"], s_sgd["rel_update_norm_scope"], marker="o", color=COL_SGD, label="SGD")
        ax.plot(s_adam["block"], s_adam["rel_update_norm_scope"], marker="o", color=COL_ADAM, label="Adam")
        ax.plot(s_match["block"], s_match["rel_update_norm_scope"], marker="o", color=COL_MATCH, label="Adam (step-0 matched)")
        ax.set_yscale("log")
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=r"Step-0 blockwise relative update norm")
    _save_png_pdf(fig, outdir / "exp5_block_step0_update_budget_compare.png")
    plt.close(fig)


# ============================================================================
# Plot D: geometry comparison
# ============================================================================

def plot_D_geometry_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, df_match: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        s_sgd = df_sgd[df_sgd["encoder"] == enc].sort_values("global_step")
        s_adam = df_adam[df_adam["encoder"] == enc].sort_values("global_step")
        s_match = df_match[df_match["encoder"] == enc].sort_values("global_step")
        ax.plot(s_sgd["global_step"], s_sgd["adam_cos_fewref_vs_refref"], marker="o", color=COL_SGD, label=r"SGD: $\cos(u_{fr},u_{rr})$")
        ax.plot(s_adam["global_step"], s_adam["adam_cos_fewref_vs_refref"], marker="o", color=COL_ADAM, label=r"Adam: $\cos(u_{fr},u_{rr})$")
        ax.plot(s_match["global_step"], s_match["adam_cos_fewref_vs_refref"], marker="o", color=COL_MATCH, label=r"Adam-matched: $\cos(u_{fr},u_{rr})$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"Geometry: $\cos(u_{fr},u_{rr})$")
    _save_png_pdf(fig, outdir / "exp5_geometry_compare.png")
    plt.close(fig)


# ============================================================================
# Plot E: support / reference loss comparison
# ============================================================================

def plot_E_loss_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, df_match: pd.DataFrame, outdir: Path) -> None:
    g_sgd = _global_budget_df(df_sgd)
    g_adam = _global_budget_df(df_adam)
    g_match = _global_budget_df(df_match)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=False)
    axes[0].plot(g_sgd["global_step"], g_sgd["few_loss"], marker="o", color=COL_SGD, label="SGD")
    axes[0].plot(g_adam["global_step"], g_adam["few_loss"], marker="o", color=COL_ADAM, label="Adam")
    axes[0].plot(g_match["global_step"], g_match["few_loss"], marker="o", color=COL_MATCH, label="Adam (step-0 matched)")
    axes[0].set_title("Support loss")
    axes[0].set_xlabel("")
    _style_axis(axes[0])

    axes[1].plot(g_sgd["global_step"], g_sgd["ref_loss"], marker="o", color=COL_SGD, label="SGD")
    axes[1].plot(g_adam["global_step"], g_adam["ref_loss"], marker="o", color=COL_ADAM, label="Adam")
    axes[1].plot(g_match["global_step"], g_match["ref_loss"], marker="o", color=COL_MATCH, label="Adam (step-0 matched)")
    axes[1].set_title("Reference loss")
    axes[1].set_xlabel("")
    _style_axis(axes[1])

    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Loss")
    _save_png_pdf(fig, outdir / "exp5_loss_compare.png")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg_sgd, df_sgd_iter, df_sgd_block = _load_run(args.logdir, args.run_name_sgd)
    cfg_adam, df_adam_iter, df_adam_block = _load_run(args.logdir, args.run_name_adam)
    cfg_match, df_match_iter, df_match_block = _load_run(args.logdir, args.run_name_adam_matched)

    _assert_compatible(cfg_sgd, cfg_adam, "SGD", "Adam")
    _assert_compatible(cfg_sgd, cfg_match, "SGD", "Adam-matched")

    df_sgd_iter = _prepare_iter_df(df_sgd_iter)
    df_adam_iter = _prepare_iter_df(df_adam_iter)
    df_match_iter = _prepare_iter_df(df_match_iter)
    df_sgd_block = _prepare_block_df(df_sgd_block)
    df_adam_block = _prepare_block_df(df_adam_block)
    df_match_block = _prepare_block_df(df_match_block)

    plot_A_global_update_budget_compare(df_sgd_iter, df_adam_iter, df_match_iter, args.outdir)
    plot_B_encoder_update_budget_compare(df_sgd_iter, df_adam_iter, df_match_iter, args.outdir)
    plot_C_block_step0_update_budget_compare(df_sgd_block, df_adam_block, df_match_block, args.outdir)
    plot_D_geometry_compare(df_sgd_iter, df_adam_iter, df_match_iter, args.outdir)
    plot_E_loss_compare(df_sgd_iter, df_adam_iter, df_match_iter, args.outdir)

    print(f"Saved Experiment 5 plots to: {args.outdir}")


if __name__ == "__main__":
    main()

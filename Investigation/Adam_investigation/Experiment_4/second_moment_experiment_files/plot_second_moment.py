#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style plotting utilities for Experiment 3 (second moment).

This script intentionally mirrors the presentation style used in the ECCV
sign-correctness plots:

  - transparent background
  - 1x2 layout (vision / text)
  - single global x/y labels
  - paper-consistent notation for K-shot vs reference quantities

Expected log files (produced by measure_second_moment.py):
  - <run_name>_config.json
  - <run_name>_per_iteration.csv
  - <run_name>_per_block.csv
  - <run_name>_hists_and_curves.npz

Figures emitted when data is available:
  A) hist_log10_v_ratio.(png|pdf)
  B) v_ratio_per_iteration.(png|pdf)
  C) v_ratio_per_block.(png|pdf)
  D) precond_cos_per_iteration.(png|pdf)
  E) precond_cos_per_block.(png|pdf)
  F) topk_v_overlap_curve.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Style (matched to plot_sign_correctness.py)
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

COL_BLUE = "#1f77b4"
COL_RED = "#d62728"
COL_GREEN = "#2ca02c"
COL_ORANGE = "#ff7f0e"
COL_GREY = "#7f7f7f"


# ============================================================================
# CLI / I/O
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing the second-moment log bundle.")
    parser.add_argument("--run_name", type=str, required=True, help="Run name used by measure_second_moment.py.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory where the plots will be saved.")
    return parser.parse_args()



def load_bundle(logdir: Path, run_name: str):
    cfg_path = logdir / f"{run_name}_config.json"
    iter_csv = logdir / f"{run_name}_per_iteration.csv"
    block_csv = logdir / f"{run_name}_per_block.csv"
    npz_path = logdir / f"{run_name}_hists_and_curves.npz"

    missing = [p for p in [cfg_path, iter_csv, block_csv, npz_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required log files:\n" + "\n".join(str(p) for p in missing))

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    df_iter = pd.read_csv(iter_csv)
    df_block = pd.read_csv(block_csv)
    npz = np.load(npz_path)
    return cfg, df_iter, df_block, npz


# ============================================================================
# Notation helpers
# ============================================================================

def _make_labels(cfg: Dict[str, object]) -> Dict[str, str]:
    k_noisy = int(cfg["shots"])
    ref_mode = str(cfg["ref_mode"])
    ref_shots = int(cfg.get("ref_shots", 16))

    v_few = rf"$v_t^{{(K={k_noisy})}}$"
    m_few = rf"$m_t^{{(K={k_noisy})}}$"

    if ref_mode == "kshot":
        v_ref = rf"$v_t^{{(K={ref_shots})}}$"
        m_ref = rf"$m_t^{{(K={ref_shots})}}$"
    else:
        v_ref = r"$v_t^{(\mathrm{full})}$"
        m_ref = r"$m_t^{(\mathrm{full})}$"

    return {
        "v_few": v_few,
        "v_ref": v_ref,
        "m_few": m_few,
        "m_ref": m_ref,
        "u_ff": r"$u_{ff}$",
        "u_fr": r"$u_{fr}$",
        "u_rr": r"$u_{rr}$",
    }


# ============================================================================
# Common helpers
# ============================================================================

def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])



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



def _plot_line_with_iqr(ax, x, arr, *, color, label):
    if arr is None or getattr(arr, "size", 0) == 0:
        return False
    med = np.median(arr, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    ax.plot(x, med, marker="o", color=color, label=label)
    ax.fill_between(x, p25, p75, alpha=0.18, color=color)
    return True



def _group_curve(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grp = (
        df.groupby("global_step", sort=True)[value_col]
        .agg(
            med="median",
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )
    return grp


# ============================================================================
# Plot A: histogram of log10(v_few / v_ref)
# ============================================================================

def plot_A_hist_log10_v_ratio(npz, outdir: Path, lab: Dict[str, str]) -> None:
    edges = npz["logratio_edges"]
    x = _centers(edges)

    def _pmf(counts: np.ndarray) -> np.ndarray:
        s = counts.sum()
        return counts / (s if s > 0 else 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        key = f"logratio_hist_{enc}"
        if key not in npz:
            continue
        counts = npz[key]
        ax.plot(x, _pmf(counts) + 1e-12, color=COL_BLUE)
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_yscale("log")
        ax.set_xlabel("")
        _style_axis(ax, which="both")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(
        fig,
        xlabel=rf"$\log_{{10}}\!\left({lab['v_few'][1:-1]}/{lab['v_ref'][1:-1]}\right)$",
        ylabel="Probability mass (log)",
    )
    _save_png_pdf(fig, outdir / "hist_log10_v_ratio.png")
    plt.close(fig)


# ============================================================================
# Plot B/C: magnitude mismatch over time / blocks
# ============================================================================

def plot_B_v_ratio_per_iteration(df_iter: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_iter[df_iter["encoder"] == enc].sort_values("global_step")
        if sub.empty:
            continue
        grp = _group_curve(sub, "v_mean_abs_log10_ratio")
        ax.plot(grp["global_step"], grp["med"], marker="o", color=COL_RED)
        ax.fill_between(grp["global_step"], grp["p25"], grp["p75"], alpha=0.18, color=COL_RED)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel=r"mean $|\log_{10}(v_{few}/v_{ref})|$")
    _save_png_pdf(fig, outdir / "v_ratio_per_iteration.png")
    plt.close(fig)



def plot_C_v_ratio_per_block(df_block: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_block[df_block["encoder"] == enc].copy()
        if sub.empty:
            continue
        grp = (
            sub.groupby("block")["v_mean_abs_log10_ratio"]
            .agg(
                med="median",
                p25=lambda x: np.percentile(x, 25),
                p75=lambda x: np.percentile(x, 75),
            )
            .reset_index()
            .sort_values("block")
        )
        x = grp["block"].to_numpy(dtype=float)
        med = grp["med"].to_numpy(dtype=float)
        p25 = grp["p25"].to_numpy(dtype=float)
        p75 = grp["p75"].to_numpy(dtype=float)
        ax.plot(x, med, marker="o", color=COL_RED)
        ax.fill_between(x, p25, p75, alpha=0.18, color=COL_RED)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=r"mean $|\log_{10}(v_{few}/v_{ref})|$")
    _save_png_pdf(fig, outdir / "v_ratio_per_block.png")
    plt.close(fig)


# ============================================================================
# Plot D/E: preconditioned-direction cosines over time / blocks
# ============================================================================

def plot_D_precond_cos_per_iteration(df_iter: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_iter[df_iter["encoder"] == enc].sort_values("global_step")
        if sub.empty:
            continue
        grp_1 = _group_curve(sub, "adam_cos_fewfew_vs_fewref")
        grp_2 = _group_curve(sub, "adam_cos_fewref_vs_refref")
        ax.plot(grp_1["global_step"], grp_1["med"], marker="o", color=COL_BLUE, label=r"$\cos(u_{ff},u_{fr})$")
        ax.fill_between(grp_1["global_step"], grp_1["p25"], grp_1["p75"], alpha=0.18, color=COL_BLUE)
        ax.plot(grp_2["global_step"], grp_2["med"], marker="o", color=COL_ORANGE, label=r"$\cos(u_{fr},u_{rr})$")
        ax.fill_between(grp_2["global_step"], grp_2["p25"], grp_2["p75"], alpha=0.18, color=COL_ORANGE)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Iteration", ylabel="Cosine")
    _save_png_pdf(fig, outdir / "precond_cos_per_iteration.png")
    plt.close(fig)



def plot_E_precond_cos_per_block(df_block: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub = df_block[df_block["encoder"] == enc].copy()
        if sub.empty:
            continue

        grp_1 = (
            sub.groupby("block")["adam_cos_fewfew_vs_fewref"]
            .agg(
                med="median",
                p25=lambda x: np.percentile(x, 25),
                p75=lambda x: np.percentile(x, 75),
            )
            .reset_index()
            .sort_values("block")
        )
        grp_2 = (
            sub.groupby("block")["adam_cos_fewref_vs_refref"]
            .agg(
                med="median",
                p25=lambda x: np.percentile(x, 25),
                p75=lambda x: np.percentile(x, 75),
            )
            .reset_index()
            .sort_values("block")
        )

        x1 = grp_1["block"].to_numpy(dtype=float)
        ax.plot(x1, grp_1["med"].to_numpy(dtype=float), marker="o", color=COL_BLUE, label=r"$\cos(u_{ff},u_{fr})$")
        ax.fill_between(x1, grp_1["p25"].to_numpy(dtype=float), grp_1["p75"].to_numpy(dtype=float), alpha=0.18, color=COL_BLUE)

        x2 = grp_2["block"].to_numpy(dtype=float)
        ax.plot(x2, grp_2["med"].to_numpy(dtype=float), marker="o", color=COL_ORANGE, label=r"$\cos(u_{fr},u_{rr})$")
        ax.fill_between(x2, grp_2["p25"].to_numpy(dtype=float), grp_2["p75"].to_numpy(dtype=float), alpha=0.18, color=COL_ORANGE)

        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel="Cosine")
    _save_png_pdf(fig, outdir / "precond_cos_per_block.png")
    plt.close(fig)


# ============================================================================
# Plot F: top-k overlap of v_few vs v_ref
# ============================================================================

def plot_F_topk_v_overlap(npz, outdir: Path) -> None:
    fracs = npz["topk_fracs"].astype(np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)

    for ax, enc in zip(axes, ["vision", "text"]):
        key = f"topk_overlap_curve_{enc}"
        if key not in npz:
            continue
        curves = npz[key]
        if getattr(curves, "size", 0) == 0:
            continue
        med = np.median(curves, axis=0)
        p25 = np.percentile(curves, 25, axis=0)
        p75 = np.percentile(curves, 75, axis=0)
        ax.plot(fracs, med, marker="o", color=COL_GREEN)
        ax.fill_between(fracs, np.clip(p25, 0, 1), np.clip(p75, 0, 1), alpha=0.18, color=COL_GREEN)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax, which="both")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Density $\rho = k/d$", ylabel=r"$|\mathrm{top}_k(v_{few})\cap\mathrm{top}_k(v_{ref})|/k$")
    _save_png_pdf(fig, outdir / "topk_v_overlap_curve.png")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    cfg, df_iter, df_block, npz = load_bundle(args.logdir, args.run_name)
    lab = _make_labels(cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_A_hist_log10_v_ratio(npz, args.outdir, lab)
    plot_B_v_ratio_per_iteration(df_iter, args.outdir)
    plot_C_v_ratio_per_block(df_block, args.outdir)
    plot_D_precond_cos_per_iteration(df_iter, args.outdir)
    plot_E_precond_cos_per_block(df_block, args.outdir)
    plot_F_topk_v_overlap(npz, args.outdir)


if __name__ == "__main__":
    main()

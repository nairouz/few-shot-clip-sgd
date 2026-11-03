#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Plot Experiment J — Noise-scale / batch-size landscape study.

Required inputs (from measure_experimentJ_noise_scale.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_variant_final.csv
  - <run_name>_connectivity.csv
  - <run_name>_connectivity_summary.json

Figures emitted:
  A) expJ_training_flatness.png
  B) expJ_noise_drift.png
  C) expJ_connectivity_curves.png
  D) expJ_budget_accounting.png
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
# Style (matched to Experiment A / user's Experiment 2 plotting style)
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

COLORS = {
    "fullbatch_nomom": "#1f77b4",
    "div2_nomom": "#ff7f0e",
    "div4_nomom": "#2ca02c",
    "fullbatch_noise": "#d62728",
    "fullbatch_momentum": "#9467bd",
}

DISPLAY = {
    "fullbatch_nomom": "full batch",
    "div2_nomom": "half batch",
    "div4_nomom": "quarter batch",
    "fullbatch_noise": "full batch + noise",
    "fullbatch_momentum": "full batch + momentum",
}

PROTOCOL_DISPLAY = {
    "epoch_matched": "epoch-matched",
    "update_matched": "update-matched",
    "example_matched": "example-matched",
    "update_example_matched": "update+example-matched",
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
    final = pd.read_csv(logdir / f"{run_name}_variant_final.csv")
    conn = pd.read_csv(logdir / f"{run_name}_connectivity.csv")
    conn_summary = json.load(open(logdir / f"{run_name}_connectivity_summary.json", "r"))
    return cfg, traj, final, conn, conn_summary


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


def _variant_iter(df: pd.DataFrame):
    order = ["fullbatch_nomom", "div2_nomom", "div4_nomom", "fullbatch_noise", "fullbatch_momentum"]
    available = [x for x in order if x in set(df["variant"])]
    for name in available:
        if "global_update" in df.columns:
            sort_cols = ["global_update", "epoch"]
        else:
            sort_cols = ["epoch"]
        yield name, df[df["variant"] == name].sort_values(sort_cols)


def _xcol_and_label(cfg: dict, traj: pd.DataFrame):
    protocol = cfg.get("budget_protocol", traj.get("budget_protocol", pd.Series(["epoch_matched"])).iloc[0] if len(traj) else "epoch_matched")
    if protocol in {"update_matched", "update_example_matched"} and "global_update" in traj.columns:
        return "global_update", "Optimizer updates", PROTOCOL_DISPLAY[protocol]
    if protocol == "example_matched" and "examples_seen" in traj.columns:
        return "examples_seen", "Seen support examples", PROTOCOL_DISPLAY[protocol]
    return "epoch", "Checkpoint epoch t", PROTOCOL_DISPLAY.get(protocol, protocol)


def plot_A_training_flatness(cfg: dict, traj: pd.DataFrame, outdir: Path) -> None:
    xcol, xlabel, protocol_label = _xcol_and_label(cfg, traj)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("full_loss", f"Full-train loss ({protocol_label})", False),
        ("test_acc", f"Test accuracy ({protocol_label})", False),
        ("full_curv_grad", f"Full-loss curvature along ∇L_full ({protocol_label})", True),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        for name, sub in _variant_iter(traj):
            ax.plot(sub[xcol], sub[col], marker="o", color=COLORS.get(name, None), label=DISPLAY.get(name, name))
        if logy:
            positive = traj[col].to_numpy()
            if np.all(positive > 0):
                ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax, which="both" if logy else "major")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=xlabel, ylabel="Value")
    _save_png_pdf(fig, outdir / "expJ_training_flatness.png")
    plt.close(fig)


def plot_B_noise_drift(cfg: dict, traj: pd.DataFrame, outdir: Path) -> None:
    xcol, xlabel, protocol_label = _xcol_and_label(cfg, traj)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("total_noise_ratio", f"Total stochasticity ratio ({protocol_label})", False),
        ("full_grad_norm", f"Full-train gradient norm ({protocol_label})", False),
        ("drift_rel", f"Relative drift from initialization ({protocol_label})", False),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        for name, sub in _variant_iter(traj):
            ax.plot(sub[xcol], sub[col], marker="o", color=COLORS.get(name, None), label=DISPLAY.get(name, name))
        if logy:
            positive = traj[col].to_numpy()
            if np.all(positive > 0):
                ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax, which="both" if logy else "major")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=xlabel, ylabel="Value")
    _save_png_pdf(fig, outdir / "expJ_noise_drift.png")
    plt.close(fig)


def plot_C_connectivity_curves(cfg: dict, conn: pd.DataFrame, conn_summary: dict, outdir: Path) -> None:
    protocol = cfg.get("budget_protocol", conn.get("budget_protocol", pd.Series(["epoch_matched"])).iloc[0] if len(conn) else "epoch_matched")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharex=True)
    panels = [
        ("full_loss", f"Full-train loss interpolation ({PROTOCOL_DISPLAY.get(protocol, protocol)})", "full_loss_curve"),
        ("test_loss", f"Test loss interpolation ({PROTOCOL_DISPLAY.get(protocol, protocol)})", "test_loss_curve"),
    ]
    for ax, (col, title, key) in zip(axes, panels):
        for pair, sub in conn.groupby("pair"):
            target = sub["target_variant"].iloc[0]
            label = DISPLAY.get(target, target)
            barrier = conn_summary[pair][key]["barrier"]
            ax.plot(
                sub["alpha"],
                sub[col],
                color=COLORS.get(target, None),
                label=f"{label} (bar={barrier:.3f})",
            )
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Interpolation coefficient $\alpha$", ylabel="Cross-entropy loss")
    _save_png_pdf(fig, outdir / "expJ_connectivity_curves.png")
    plt.close(fig)


def plot_D_budget_accounting(cfg: dict, traj: pd.DataFrame, outdir: Path) -> None:
    xcol, xlabel, protocol_label = _xcol_and_label(cfg, traj)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), sharex=True)
    panels = [
        ("epochs_completed", f"Actual epochs completed ({protocol_label})", False),
        ("example_budget_ratio", f"Seen-example budget ratio ({protocol_label})", False),
        ("update_budget_ratio", f"Update-budget ratio ({protocol_label})", False),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        if col not in traj.columns:
            ax.set_visible(False)
            continue
        for name, sub in _variant_iter(traj):
            ax.plot(sub[xcol], sub[col], marker="o", color=COLORS.get(name, None), label=DISPLAY.get(name, name))
        if logy:
            positive = traj[col].to_numpy()
            if np.all(positive > 0):
                ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        _style_axis(ax, which="both" if logy else "major")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=xlabel, ylabel="Value")
    _save_png_pdf(fig, outdir / "expJ_budget_accounting.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, final, conn, conn_summary = _load_bundle(args.logdir, args.run_name)
    plot_A_training_flatness(cfg, traj, args.outdir)
    plot_B_noise_drift(cfg, traj, args.outdir)
    plot_C_connectivity_curves(cfg, conn, conn_summary, args.outdir)
    plot_D_budget_accounting(cfg, traj, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

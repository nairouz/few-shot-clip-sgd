#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Experiment H — Support-vs-full landscape mismatch.

Required inputs (from measure_experimentH_support_full_mismatch.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_encoder_alignment.csv
  - <run_name>_block_alignment.csv
  - <run_name>_surface.csv
  - <run_name>_summary.json

Figures emitted:
  A) expH_gradient_alignment.(png|pdf)
  B) expH_hessian_alignment.(png|pdf)
  C) expH_block_gradient_alignment.(png|pdf)
  D) expH_surface_tXX.(png|pdf) for each analysed surface checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Style (matched to Experiment 2 / Experiment A plotting style)
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

COL_MAIN = "#1f77b4"
COL_MID = "#2ca02c"
COL_ACCENT = "#d62728"
COL_AUX = "#7f7f7f"
SURF_CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    enc = pd.read_csv(logdir / f"{run_name}_encoder_alignment.csv")
    blk = pd.read_csv(logdir / f"{run_name}_block_alignment.csv")
    surf_path = logdir / f"{run_name}_surface.csv"
    surf = pd.read_csv(surf_path) if surf_path.is_file() else pd.DataFrame()
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, enc, blk, surf, summary


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


def _encoder_order(df: pd.DataFrame) -> List[str]:
    order = [enc for enc in ["vision", "text"] if enc in df["encoder"].unique().tolist()]
    return order if order else sorted(df["encoder"].unique().tolist())


def plot_A_gradient_alignment(enc: pd.DataFrame, outdir: Path) -> None:
    encoders = _encoder_order(enc)
    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2 * len(encoders), 3.8), sharex=True)
    if len(encoders) == 1:
        axes = [axes]
    for ax, encoder in zip(axes, encoders):
        sub = enc[enc["encoder"] == encoder].sort_values("epoch")
        ax.plot(sub["epoch"], sub["grad_cosine"], marker="o", color=COL_MAIN, label=r"$\cos(g_{few}, g_{full})$")
        ax.plot(sub["epoch"], sub["grad_sign_agreement"], marker="o", color=COL_MID, label="sign agreement")
        ax.axhline(0.0, color=COL_AUX, linestyle=":", linewidth=1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{encoder.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

        ax_r = ax.twinx()
        ratio = np.maximum(sub["grad_norm_ratio"].to_numpy(dtype=float), 1e-8)
        ax_r.plot(sub["epoch"], ratio, marker="s", linestyle="--", color=COL_ACCENT, label=r"$\|g_{few}\|_2 / \|g_{full}\|_2$")
        ax_r.set_yscale("log")
        ax_r.set_ylabel(r"Norm ratio", color=COL_ACCENT)
        ax_r.tick_params(axis="y", colors=COL_ACCENT)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch $t$", ylabel="Alignment / agreement")
    _save_png_pdf(fig, outdir / "expH_gradient_alignment.png")
    plt.close(fig)


def plot_B_hessian_alignment(enc: pd.DataFrame, outdir: Path) -> None:
    encoders = _encoder_order(enc)
    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2 * len(encoders), 3.8), sharex=True)
    if len(encoders) == 1:
        axes = [axes]
    for ax, encoder in zip(axes, encoders):
        sub = enc[enc["encoder"] == encoder].sort_values("epoch")
        ax.plot(sub["epoch"], sub["hessian_eigvec_abs_cosine"], marker="o", color=COL_MAIN, label=r"$|\cos(v_1^{few}, v_1^{full})|$")
        ax.axhline(0.0, color=COL_AUX, linestyle=":", linewidth=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{encoder.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)

        ax_r = ax.twinx()
        ratio = np.log10((np.abs(sub["hessian_top_eig_full"].to_numpy(dtype=float)) + 1e-12) / (np.abs(sub["hessian_top_eig_few"].to_numpy(dtype=float)) + 1e-12))
        ax_r.plot(sub["epoch"], ratio, marker="s", linestyle="--", color=COL_ACCENT, label=r"$\log_{10}(|\lambda^{full}_1|/|\lambda^{few}_1|)$")
        ax_r.set_ylabel(r"Eigenvalue log-ratio", color=COL_ACCENT)
        ax_r.tick_params(axis="y", colors=COL_ACCENT)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Checkpoint epoch $t$", ylabel="Dominant eigenspace alignment")
    _save_png_pdf(fig, outdir / "expH_hessian_alignment.png")
    plt.close(fig)


def plot_C_block_gradient_alignment(blk: pd.DataFrame, outdir: Path) -> None:
    encoders = _encoder_order(blk)
    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2 * len(encoders), 3.8), sharey=True)
    if len(encoders) == 1:
        axes = [axes]
    for ax, encoder in zip(axes, encoders):
        sub = blk[blk["encoder"] == encoder].copy()
        grp = sub.groupby("block").agg(
            grad_cos_med=("grad_cosine", "median"),
            grad_cos_p25=("grad_cosine", lambda x: np.percentile(x, 25)),
            grad_cos_p75=("grad_cosine", lambda x: np.percentile(x, 75)),
            sign_med=("grad_sign_agreement", "median"),
        ).reset_index().sort_values("block")
        x = grp["block"].to_numpy(dtype=float)
        ax.plot(x, grp["grad_cos_med"], marker="o", color=COL_MAIN, label=r"median $\cos(g_{few}, g_{full})$")
        ax.fill_between(x, grp["grad_cos_p25"], grp["grad_cos_p75"], color=COL_MAIN, alpha=0.18)
        ax.plot(x, grp["sign_med"], marker="s", linestyle="--", color=COL_MID, label="median sign agreement")
        ax.axhline(0.0, color=COL_AUX, linestyle=":", linewidth=1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{encoder.capitalize()} encoder")
        ax.set_xlabel("")
        _style_axis(ax)
        ax.legend(frameon=False, loc="lower left")

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Transformer block $\ell$", ylabel="Alignment / agreement")
    _save_png_pdf(fig, outdir / "expH_block_gradient_alignment.png")
    plt.close(fig)


def _pivot_surface(sub: pd.DataFrame, col: str):
    piv = sub.pivot(index="beta_units", columns="alpha_units", values=col).sort_index().sort_index(axis=1)
    X = piv.columns.to_numpy(dtype=float)
    Y = piv.index.to_numpy(dtype=float)
    Z = piv.to_numpy(dtype=float)
    return X, Y, Z


def _plot_single_surface(ax: plt.Axes, sub: pd.DataFrame, value_col: str, title: str) -> None:
    X, Y, Z = _pivot_surface(sub, value_col)
    XX, YY = np.meshgrid(X, Y)
    cs = ax.contourf(XX, YY, Z, levels=21, cmap=SURF_CMAP)
    ax.contour(XX, YY, Z, levels=10, colors="white", linewidths=0.5, alpha=0.45)
    # Origin + arrows that indicate the support-gradient basis axis and the projected full gradient.
    arrow_scale = 0.55 * max(np.max(np.abs(X)), np.max(np.abs(Y)))
    meta = sub.iloc[0]
    ax.annotate("", xy=(arrow_scale, 0.0), xytext=(0.0, 0.0), arrowprops=dict(color="black", width=1.2, headwidth=7))
    ax.annotate("", xy=(arrow_scale * float(meta["full_arrow_x"]), arrow_scale * float(meta["full_arrow_y"])), xytext=(0.0, 0.0), arrowprops=dict(color=COL_ACCENT, width=1.2, headwidth=7))
    ax.scatter([0.0], [0.0], color="white", s=24, edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _style_axis(ax)
    return cs


def plot_D_surface_grid(surf: pd.DataFrame, enc: pd.DataFrame, outdir: Path) -> None:
    if surf.empty:
        return
    encoders = _encoder_order(enc)
    for epoch in sorted(surf["epoch"].unique().tolist()):
        surf_t = surf[surf["epoch"] == epoch]
        fig, axes = plt.subplots(len(encoders), 2, figsize=(10.6, 3.8 * len(encoders)), sharex=True, sharey=True)
        if len(encoders) == 1:
            axes = np.array([axes])
        mappables = []
        for i, encoder in enumerate(encoders):
            surf_enc = surf_t[surf_t["encoder"] == encoder]
            if surf_enc.empty:
                continue
            sub_enc = enc[(enc["epoch"] == epoch) & (enc["encoder"] == encoder)]
            grad_cos = float(sub_enc["grad_cosine"].iloc[0]) if not sub_enc.empty else float("nan")
            eig_cos = float(sub_enc["hessian_eigvec_abs_cosine"].iloc[0]) if not sub_enc.empty else float("nan")
            cs = _plot_single_surface(
                axes[i, 0],
                surf_enc,
                "few_loss",
                f"{encoder.capitalize()} • support loss\n$\\cos(g_{{few}},g_{{full}})$={grad_cos:.2f}",
            )
            mappables.append(cs)
            cs = _plot_single_surface(
                axes[i, 1],
                surf_enc,
                "full_loss",
                f"{encoder.capitalize()} • full-train loss\n$|\\cos(v_1^{{few}},v_1^{{full}})|$={eig_cos:.2f}",
            )
            mappables.append(cs)
        # Shared colorbars per column.
        for j, label in enumerate(["Support loss", "Full-train loss"]):
            col_axes = axes[:, j].ravel().tolist()
            vals = [m for idx, m in enumerate(mappables) if idx % 2 == j]
            if vals:
                cbar = fig.colorbar(vals[0], ax=col_axes, shrink=0.9)
                cbar.set_label(label)
        plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
        _add_global_xy_labels(fig, xlabel=r"$\alpha$ (support-gradient axis, in step units)", ylabel=r"$\beta$ (full-gradient orthogonal axis, in step units)")
        _save_png_pdf(fig, outdir / f"expH_surface_t{int(epoch):02d}.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, enc, blk, surf, summary = _load_bundle(args.logdir, args.run_name)
    plot_A_gradient_alignment(enc, args.outdir)
    plot_B_hessian_alignment(enc, args.outdir)
    plot_C_block_gradient_alignment(blk, args.outdir)
    plot_D_surface_grid(surf, enc, args.outdir)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

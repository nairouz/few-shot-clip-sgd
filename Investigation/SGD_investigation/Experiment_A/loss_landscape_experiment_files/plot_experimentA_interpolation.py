#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Experiment A — 1D interpolation from initialization to final SGD solution.

Required inputs (from measure_experimentA_interpolation.py):
  - <run_name>_config.json
  - <run_name>_trajectory.csv
  - <run_name>_checkpoint_projection.csv
  - <run_name>_interpolation.csv
  - <run_name>_summary.json

Figures emitted:
  A) expA_interpolation.pdf/png           (three subplots: few/full/test)
  B) expA_checkpoint_projection.pdf/png   (optional auxiliary figure)

The figure notation matches the NeurIPS paper:
  Theta(alpha) = (1-alpha) Theta_0 + alpha Theta_T
  varphi_ell(alpha)
  alpha_ell^*, B_ell, W_ell^(0.05)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Style (matched to the revised Experiment B / E figures)
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
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.6,
    "lines.markersize": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_CURVE = "#1f77b4"
COL_CHORD = "#7f7f7f"
COL_MIN = "#d62728"
COL_END = "#2ca02c"
COL_MILE = "#6e6e6e"

LOSS_KEYS = [
    ("few_loss", "few_loss_curve", r"\mathrm{few}", r"$\varphi_{\mathrm{few}}(\alpha)$"),
    ("full_loss", "full_loss_curve", r"\mathrm{full}", r"$\varphi_{\mathrm{full}}(\alpha)$"),
    ("test_loss", "test_loss_curve", r"\mathrm{test}", r"$\varphi_{\mathrm{test}}(\alpha)$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--paper_prefix",
        type=str,
        default=None,
        help=(
            "Optional basename prefix for paper figures. If set to 'expA_eurosat', the script also writes "
            "expA_eurosat_interpolation.pdf/png and expA_eurosat_projection.pdf/png."
        ),
    )
    parser.add_argument(
        "--add_suptitle",
        action="store_true",
        help="Add a compact dataset suptitle for standalone browsing.",
    )
    return parser.parse_args()


def _load_bundle(logdir: Path, run_name: str):
    cfg = json.load(open(logdir / f"{run_name}_config.json", "r"))
    traj = pd.read_csv(logdir / f"{run_name}_trajectory.csv")
    proj = pd.read_csv(logdir / f"{run_name}_checkpoint_projection.csv")
    interp = pd.read_csv(logdir / f"{run_name}_interpolation.csv")
    summary = json.load(open(logdir / f"{run_name}_summary.json", "r"))
    return cfg, traj, proj, interp, summary


def _infer_dataset_name(run_name: str, cfg: dict) -> str:
    ds = str(cfg.get("dataset", ""))
    if not ds:
        m = re.search(r"dataset_([A-Za-z0-9_]+)_shots", run_name)
        ds = m.group(1) if m else run_name
    mapping = {
        "eurosat": "EuroSAT",
        "oxford_pets": "Oxford-Pets",
        "dtd": "DTD",
    }
    return mapping.get(ds, ds.replace("_", " ").title())


def _add_global_xy_labels(fig: plt.Figure, xlabel: str, ylabel: str) -> None:
    fs = mpl.rcParams.get("axes.labelsize", BASE_SIZE + 2)
    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=fs)
    fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=fs)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.tick_params(axis="both", pad=4)


def _save_png_pdf(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() not in {".png", ".pdf"}:
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    alt = out.with_suffix(".pdf") if out.suffix.lower() == ".png" else out.with_suffix(".png")
    fig.savefig(alt, dpi=300, bbox_inches="tight", transparent=True)


def _save_standard_and_paper(fig: plt.Figure, outdir: Path, stem: str, paper_prefix: str | None, paper_suffix: str) -> None:
    _save_png_pdf(fig, outdir / f"{stem}.png")
    if paper_prefix:
        _save_png_pdf(fig, outdir / f"{paper_prefix}_{paper_suffix}.png")


def _maybe_add_dataset_suptitle(fig: plt.Figure, dataset_name: str, add_suptitle: bool) -> None:
    if add_suptitle:
        fig.suptitle(dataset_name, y=1.03, fontsize=BASE_SIZE + 3)


def _interp_chord(alpha: np.ndarray, y0: float, y1: float) -> np.ndarray:
    return (1.0 - alpha) * y0 + alpha * y1


def plot_A_interpolation(interp: pd.DataFrame, summary: dict, outdir: Path, *, dataset_name: str, add_suptitle: bool, paper_prefix: str | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 3.9), sharex=True)

    for ax, (col, key, ell_tex, title) in zip(axes, LOSS_KEYS):
        alpha = interp["alpha"].to_numpy(dtype=float)
        y = interp[col].to_numpy(dtype=float)
        y0, y1 = float(y[0]), float(y[-1])
        chord = _interp_chord(alpha, y0, y1)
        info = summary[key]
        alpha_star = float(info["alpha_min"])
        y_star = float(info["loss_min"])
        # curve and chord
        ax.plot(alpha, y, color=COL_CURVE)
        ax.plot(alpha, chord, color=COL_CHORD, linestyle='--', linewidth=1.6)
        # endpoints and minimizer
        ax.scatter([0.0, 1.0], [y0, y1], color=COL_END, s=26, zorder=3)
        ax.axvline(alpha_star, color=COL_MIN, linestyle=':', linewidth=1.5)
        ax.scatter([alpha_star], [y_star], color=COL_MIN, s=28, zorder=4)
        ax.set_title(title)
        _style_axis(ax)
        stats_text = "\n".join([
            rf"$\alpha_{{{ell_tex}}}^{{\star}}={alpha_star:.2f}$",
            rf"$B_{{{ell_tex}}}={float(info['barrier']):.3f}$",
            rf"$W_{{{ell_tex}}}^{{(0.05)}}={float(info['width_5pct']):.2f}$",
        ])
        ax.text(
            0.97, 0.93, stats_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=BASE_SIZE,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.92),
            zorder=5,
        )
        ax.set_xlabel("")

    _maybe_add_dataset_suptitle(fig, dataset_name, add_suptitle)
    plt.tight_layout(rect=[0.04, 0.08, 1.0, 0.98])
    _add_global_xy_labels(
        fig,
        xlabel=r"Interpolation coefficient $\alpha$ in $\Theta(\alpha)=(1-\alpha)\Theta_0+\alpha\Theta_T$",
        ylabel=r"$\varphi_{\ell}(\alpha)$",
    )
    _save_standard_and_paper(fig, outdir, "expA_interpolation", paper_prefix, "interpolation")
    plt.close(fig)


def _format_decimal_tick(y: float, _pos=None) -> str:
    if not np.isfinite(y):
        return ""
    if abs(y) < 1e-12:
        return "0"
    if y >= 0.1:
        s = f"{float(y):.1f}"
    else:
        s = f"{float(y):.2f}"
    return s.rstrip("0").rstrip(".")


def _orth_deviation_ticks(values: np.ndarray) -> list[float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return []
    ymin = float(vals.min())
    ymax = float(vals.max())
    candidates = []
    lo = int(np.floor(np.log10(ymin))) - 1
    hi = int(np.ceil(np.log10(ymax))) + 1
    mantissas = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0]
    for k in range(lo, hi + 1):
        scale = 10.0 ** k
        for m in mantissas:
            t = m * scale
            if ymin <= t <= ymax:
                candidates.append(float(t))
    def _nice_endpoint(v: float) -> float:
        if v >= 0.1:
            return round(v, 1)
        return round(v, 2)
    ticks = sorted(set(candidates + [_nice_endpoint(ymin), _nice_endpoint(ymax)]))
    return ticks


def plot_B_projection(proj: pd.DataFrame, outdir: Path, *, dataset_name: str, add_suptitle: bool, paper_prefix: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), sharex=True)
    axes[0].plot(proj["epoch"], proj["alpha_proj_clipped"], marker="o", color=COL_CURVE)
    axes[0].axhline(0.0, color=COL_MILE, linestyle=":", linewidth=1.0)
    axes[0].axhline(1.0, color=COL_MILE, linestyle=":", linewidth=1.0)
    axes[0].set_title(r"Projection on $[\Theta_0,\Theta_T]$")
    axes[0].set_ylabel(r"$\alpha_{\mathrm{proj}}(t)$")
    _style_axis(axes[0])

    axes[1].plot(proj["epoch"], proj["orth_rel"], marker="o", color=COL_MIN)
    axes[1].set_yscale("log")
    axes[1].set_title(r"Relative orthogonal deviation")
    axes[1].set_ylabel(r"$\|r_t\|_2 / \|\Theta_T-\Theta_0\|_2$")
    yticks = _orth_deviation_ticks(proj["orth_rel"].to_numpy())
    if yticks:
        axes[1].set_yticks(yticks)
        axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_format_decimal_tick))
    _style_axis(axes[1])

    _maybe_add_dataset_suptitle(fig, dataset_name, add_suptitle)
    plt.tight_layout(rect=[0.05, 0.08, 1.0, 0.98])
    _add_global_xy_labels(fig, xlabel=r"Checkpoint epoch $t$", ylabel="")
    _save_standard_and_paper(fig, outdir, "expA_checkpoint_projection", paper_prefix, "projection")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, traj, proj, interp, summary = _load_bundle(args.logdir, args.run_name)
    dataset_name = _infer_dataset_name(args.run_name, cfg)
    plot_A_interpolation(interp, summary, args.outdir, dataset_name=dataset_name, add_suptitle=args.add_suptitle, paper_prefix=args.paper_prefix)
    plot_B_projection(proj, args.outdir, dataset_name=dataset_name, add_suptitle=args.add_suptitle, paper_prefix=args.paper_prefix)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

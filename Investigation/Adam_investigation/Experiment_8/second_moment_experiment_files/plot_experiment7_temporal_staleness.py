#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 7: temporal staleness of the few-shot gradient.

This script compares an SGD trajectory and an Adam trajectory using the CSV
bundles produced by ``measure_experiment7_temporal_staleness.py``.

Required inputs:
  - <run_name_sgd>_config.json
  - <run_name_sgd>_temporal_staleness.csv
  - <run_name_adam>_config.json
  - <run_name_adam>_temporal_staleness.csv

Figures emitted:
  A) exp7_temporal_autocorr_fewfew_compare.(png|pdf)
  B) exp7_temporal_autocorr_fewfuturefull_compare.(png|pdf)
  C) exp7_temporal_lag_summary_compare.(png|pdf)
  D) exp7_temporal_per_block_futurefull_compare.(png|pdf)
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
# Style (matched to Experiment 2 / second-moment plotting)
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
COL_GREY = "#7f7f7f"
LAG_STYLE = {1: "-", 2: "--", 5: ":"}


# ============================================================================
# CLI / I/O
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing the Experiment 7 CSV bundles.")
    parser.add_argument("--run_name_sgd", type=str, required=True, help="Run name for the SGD trajectory.")
    parser.add_argument("--run_name_adam", type=str, required=True, help="Run name for the Adam trajectory.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory where the plots will be saved.")
    return parser.parse_args()



def _load_run(logdir: Path, run_name: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    cfg_path = logdir / f"{run_name}_config.json"
    csv_path = logdir / f"{run_name}_temporal_staleness.csv"
    missing = [p for p in [cfg_path, csv_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Experiment 7 files:\n" + "\n".join(str(p) for p in missing))
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    df = pd.read_csv(csv_path)
    return cfg, df



def _assert_compatible(cfg_sgd: Dict[str, object], cfg_adam: Dict[str, object]) -> None:
    keys = ["dataset", "shots", "backbone", "position", "encoder", "ref_mode", "lags"]
    mismatches = []
    for k in keys:
        if cfg_sgd.get(k) != cfg_adam.get(k):
            mismatches.append((k, cfg_sgd.get(k), cfg_adam.get(k)))
    if mismatches:
        msg = "Experiment 7 requires comparable SGD/Adam runs. Mismatches found:\n"
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



def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["scope", "encoder", "lag", "step_t"])
    return df



def _lags_from_cfg(cfg: Dict[str, object]) -> list[int]:
    vals = cfg.get("lags", [1, 2, 5])
    return [int(x) for x in vals]


# ============================================================================
# Plot A: cos(m_t^few, m_{t+Δ}^few)
# ============================================================================

def plot_A_temporal_autocorr_fewfew_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, lags: list[int], outdir: Path) -> None:
    df_sgd = df_sgd[df_sgd["scope"] == "encoder"].copy()
    df_adam = df_adam[df_adam["scope"] == "encoder"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc]
        sub_adam = df_adam[df_adam["encoder"] == enc]
        for lag in lags:
            ss = sub_sgd[sub_sgd["lag"] == lag].sort_values("step_t")
            sa = sub_adam[sub_adam["lag"] == lag].sort_values("step_t")
            ax.plot(ss["step_t"], ss["corr_few_t_to_few_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_SGD, label=fr"SGD, $\Delta={lag}$")
            ax.plot(sa["step_t"], sa["corr_few_t_to_few_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_ADAM, label=fr"Adam, $\Delta={lag}$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Anchor step $t$", ylabel=r"$\cos(m_t^{few}, m_{t+\Delta}^{few})$")
    _save_png_pdf(fig, outdir / "exp7_temporal_autocorr_fewfew_compare.png")
    plt.close(fig)


# ============================================================================
# Plot B: cos(m_t^few, m_{t+Δ}^full)
# ============================================================================

def plot_B_temporal_autocorr_fewfuturefull_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, lags: list[int], outdir: Path) -> None:
    df_sgd = df_sgd[df_sgd["scope"] == "encoder"].copy()
    df_adam = df_adam[df_adam["scope"] == "encoder"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc]
        sub_adam = df_adam[df_adam["encoder"] == enc]
        for lag in lags:
            ss = sub_sgd[sub_sgd["lag"] == lag].sort_values("step_t")
            sa = sub_adam[sub_adam["lag"] == lag].sort_values("step_t")
            ax.plot(ss["step_t"], ss["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_SGD, label=fr"SGD, $\Delta={lag}$")
            ax.plot(sa["step_t"], sa["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_ADAM, label=fr"Adam, $\Delta={lag}$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Anchor step $t$", ylabel=r"$\cos(m_t^{few}, m_{t+\Delta}^{full})$")
    _save_png_pdf(fig, outdir / "exp7_temporal_autocorr_fewfuturefull_compare.png")
    plt.close(fig)


# ============================================================================
# Plot C: lag summary across time
# ============================================================================

def plot_C_temporal_lag_summary_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, lags: list[int], outdir: Path) -> None:
    df_sgd = df_sgd[df_sgd["scope"] == "encoder"].copy()
    df_adam = df_adam[df_adam["scope"] == "encoder"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        ss = df_sgd[df_sgd["encoder"] == enc].groupby("lag", sort=True)[["corr_few_t_to_few_t_plus_lag", "corr_few_t_to_full_t_plus_lag"]].median().reset_index()
        sa = df_adam[df_adam["encoder"] == enc].groupby("lag", sort=True)[["corr_few_t_to_few_t_plus_lag", "corr_few_t_to_full_t_plus_lag"]].median().reset_index()
        ax.plot(ss["lag"], ss["corr_few_t_to_few_t_plus_lag"], marker="o", linestyle="--", color=COL_SGD, label=r"SGD: $\cos(m_t^{few}, m_{t+\Delta}^{few})$")
        ax.plot(ss["lag"], ss["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle="-", color=COL_SGD, label=r"SGD: $\cos(m_t^{few}, m_{t+\Delta}^{full})$")
        ax.plot(sa["lag"], sa["corr_few_t_to_few_t_plus_lag"], marker="o", linestyle="--", color=COL_ADAM, label=r"Adam: $\cos(m_t^{few}, m_{t+\Delta}^{few})$")
        ax.plot(sa["lag"], sa["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle="-", color=COL_ADAM, label=r"Adam: $\cos(m_t^{few}, m_{t+\Delta}^{full})$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_xticks(lags)
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=1)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel=r"Temporal lag $\Delta$", ylabel="Median cosine over time")
    _save_png_pdf(fig, outdir / "exp7_temporal_lag_summary_compare.png")
    plt.close(fig)


# ============================================================================
# Plot D: per-block depth summary for cos(m_t^few, m_{t+Δ}^full)
# ============================================================================

def plot_D_temporal_per_block_futurefull_compare(df_sgd: pd.DataFrame, df_adam: pd.DataFrame, lags: list[int], outdir: Path) -> None:
    df_sgd = df_sgd[df_sgd["scope"] == "block"].copy()
    df_adam = df_adam[df_adam["scope"] == "block"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    for ax, enc in zip(axes, ["vision", "text"]):
        sub_sgd = df_sgd[df_sgd["encoder"] == enc]
        sub_adam = df_adam[df_adam["encoder"] == enc]
        for lag in lags:
            ss = sub_sgd[sub_sgd["lag"] == lag].groupby("block", sort=True)["corr_few_t_to_full_t_plus_lag"].median().reset_index()
            sa = sub_adam[sub_adam["lag"] == lag].groupby("block", sort=True)["corr_few_t_to_full_t_plus_lag"].median().reset_index()
            ax.plot(ss["block"], ss["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_SGD, label=fr"SGD, $\Delta={lag}$")
            ax.plot(sa["block"], sa["corr_few_t_to_full_t_plus_lag"], marker="o", linestyle=LAG_STYLE.get(lag, "-"), color=COL_ADAM, label=fr"Adam, $\Delta={lag}$")
        ax.axhline(0.0, color=COL_GREY, linestyle=":", linewidth=1.0)
        ax.set_title(f"{enc.capitalize()} encoder")
        ax.set_xlabel("")
        ax.set_xticks(sorted(sub_sgd["block"].dropna().unique()))
        ax.set_ylim(-1.0, 1.0)
        _style_axis(ax)
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    _add_global_xy_labels(fig, xlabel="Transformer block", ylabel=r"Median $\cos(m_t^{few}, m_{t+\Delta}^{full})$")
    _save_png_pdf(fig, outdir / "exp7_temporal_per_block_futurefull_compare.png")
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

    lags = _lags_from_cfg(cfg_sgd)
    df_sgd = _prepare_df(df_sgd)
    df_adam = _prepare_df(df_adam)

    plot_A_temporal_autocorr_fewfew_compare(df_sgd, df_adam, lags, args.outdir)
    plot_B_temporal_autocorr_fewfuturefull_compare(df_sgd, df_adam, lags, args.outdir)
    plot_C_temporal_lag_summary_compare(df_sgd, df_adam, lags, args.outdir)
    plot_D_temporal_per_block_futurefull_compare(df_sgd, df_adam, lags, args.outdir)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Consistent plotting helpers for P5 figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "zero-shot": "#555555",
    "sgd": "#1f77b4",
    "adam": "#d62728",
    "adam-m0": "#2ca02c",
    "vision": "#9467bd",
    "text": "#ff7f0e",
}
MARKERS = {"sgd": "o", "adam": "s", "adam-m0": "^"}


def rows(path):
    with Path(path).open(newline="") as stream:
        output = []
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value) if value != "" else None
                except (TypeError, ValueError):
                    parsed[key] = value
            output.append(parsed)
        return output


def finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

"""Shared plotting style for dissertation-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "blue": "#2F6F9F",
    "green": "#3B7F5C",
    "red": "#B85C5C",
    "gold": "#B38B2E",
    "purple": "#6B5B95",
    "gray": "#5B6470",
    "dark": "#1F2933",
}

LIGHT_COLORS = {
    "blue": "#E7F0F7",
    "green": "#E8F3ED",
    "red": "#F6E7E7",
    "gold": "#F8F1DF",
    "purple": "#EEEAF6",
    "gray": "#F3F5F7",
}


def apply_dissertation_style() -> None:
    """Apply a compact academic plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["dark"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 1.8,
            "patch.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    """Save a high-resolution figure without clipped labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

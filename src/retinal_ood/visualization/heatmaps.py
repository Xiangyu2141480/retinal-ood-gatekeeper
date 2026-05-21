"""Heatmap visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_map(anomaly_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize anomaly map to [0, 1]."""
    arr = np.asarray(anomaly_map, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + eps)


def save_heatmap_overlay(image_path: str | Path, anomaly_map: np.ndarray, out_path: str | Path) -> None:
    """Save a simple side-by-side original/heatmap overlay figure."""
    image = Image.open(image_path).convert("RGB")
    heat = Image.fromarray((normalize_map(anomaly_map) * 255).astype(np.uint8)).resize(image.size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heat, alpha=0.45)
    plt.axis("off")
    plt.title("Anomaly heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

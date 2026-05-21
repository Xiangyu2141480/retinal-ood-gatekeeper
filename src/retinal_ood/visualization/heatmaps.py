"""Heatmap visualization utilities for patch-level anomaly maps."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_OUTCOME_ORDER = ("fp", "fn", "tp", "tn")
_BILINEAR = Image.Resampling.BILINEAR


@dataclass(frozen=True)
class HeatmapNormalization:
    """Run-level color scale used to make heatmaps comparable within one evaluation."""

    vmin: float
    vmax: float


def normalize_map(
    anomaly_map: np.ndarray,
    *,
    normalization: HeatmapNormalization | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize an anomaly map to ``[0, 1]`` using per-map or run-level limits."""
    arr = _validate_anomaly_map(anomaly_map)
    if normalization is None:
        vmin = float(arr.min())
        vmax = float(arr.max())
    else:
        vmin = normalization.vmin
        vmax = normalization.vmax
    denom = max(vmax - vmin, eps)
    return np.clip((arr - vmin) / denom, 0.0, 1.0)


def resize_anomaly_map(anomaly_map: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D patch anomaly map to a PIL-style ``(width, height)`` image size."""
    arr = _validate_anomaly_map(anomaly_map)
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size must contain positive width and height")
    resized = Image.fromarray(arr.astype(np.float32), mode="F").resize(output_size, resample=_BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def compute_heatmap_normalization(patch_maps: list[np.ndarray]) -> HeatmapNormalization:
    """Compute one color scale for all patch maps in an evaluation run."""
    if not patch_maps:
        raise ValueError("patch_maps must not be empty")
    validated = [_validate_anomaly_map(patch_map) for patch_map in patch_maps]
    vmin = min(float(patch_map.min()) for patch_map in validated)
    vmax = max(float(patch_map.max()) for patch_map in validated)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return HeatmapNormalization(vmin=vmin, vmax=vmax)


def colorize_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    normalization: HeatmapNormalization,
    cmap: str = "magma",
) -> Image.Image:
    """Convert a normalized anomaly map to an RGB heatmap image."""
    normalized = normalize_map(anomaly_map, normalization=normalization)
    color_map = plt.get_cmap(cmap)
    rgb = (color_map(normalized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(
    image: Image.Image,
    anomaly_map: np.ndarray,
    *,
    normalization: HeatmapNormalization,
    alpha: float = 0.45,
    cmap: str = "magma",
) -> tuple[Image.Image, Image.Image]:
    """Return RGB heatmap and overlay images aligned to the input image."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    image = image.convert("RGB")
    resized_map = resize_anomaly_map(anomaly_map, image.size)
    heatmap = colorize_anomaly_map(resized_map, normalization=normalization, cmap=cmap)
    overlay = Image.blend(image, heatmap, alpha=alpha)
    return heatmap, overlay


def save_heatmap_artifacts(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    out_dir: str | Path,
    *,
    stem: str = "sample",
    normalization: HeatmapNormalization | None = None,
    alpha: float = 0.45,
    cmap: str = "magma",
) -> dict[str, Path]:
    """Save original, heatmap, and overlay PNGs for one image."""
    image = Image.open(image_path).convert("RGB")
    normalization = normalization or compute_heatmap_normalization([anomaly_map])
    heatmap, overlay = overlay_heatmap(
        image,
        anomaly_map,
        normalization=normalization,
        alpha=alpha,
        cmap=cmap,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_stem(stem)
    paths = {
        "original_file": out_dir / f"{safe_stem}_original.png",
        "heatmap_file": out_dir / f"{safe_stem}_heatmap.png",
        "overlay_file": out_dir / f"{safe_stem}_overlay.png",
    }
    image.save(paths["original_file"])
    heatmap.save(paths["heatmap_file"])
    overlay.save(paths["overlay_file"])
    return paths


def save_heatmap_overlay(image_path: str | Path, anomaly_map: np.ndarray, out_path: str | Path) -> None:
    """Save only an overlay PNG, kept for simple ad-hoc visual checks."""
    image = Image.open(image_path).convert("RGB")
    normalization = compute_heatmap_normalization([anomaly_map])
    _, overlay = overlay_heatmap(image, anomaly_map, normalization=normalization)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def select_top_k_outcomes(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    top_k: int,
) -> dict[str, list[int]]:
    """Select top-K false positives, false negatives, true positives, and true negatives."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must be one-dimensional arrays with the same length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    predictions = scores >= threshold
    masks = {
        "fp": (labels == 0) & predictions,
        "fn": (labels == 1) & ~predictions,
        "tp": (labels == 1) & predictions,
        "tn": (labels == 0) & ~predictions,
    }
    selected: dict[str, list[int]] = {}
    for outcome in _OUTCOME_ORDER:
        indices = np.flatnonzero(masks[outcome]).tolist()
        descending = outcome in {"fp", "tp"}
        selected[outcome] = _rank_indices(indices, scores, descending=descending)[:top_k]
    return selected


def save_topk_heatmaps(
    metadata_rows: list[dict[str, Any]],
    patch_maps: list[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    out_dir: str | Path,
    top_k: int = 5,
    alpha: float = 0.45,
    cmap: str = "magma",
    image_path_key: str = "resolved_image_path",
) -> list[dict[str, Any]]:
    """Save top-K diagnostic heatmaps and a privacy-conscious manifest."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(metadata_rows) != len(patch_maps) or len(patch_maps) != labels.shape[0]:
        raise ValueError("metadata_rows, patch_maps, labels, and scores must have the same length")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for outcome in _OUTCOME_ORDER:
        (out_dir / outcome).mkdir(parents=True, exist_ok=True)

    normalization = compute_heatmap_normalization(patch_maps)
    save_heatmap_colorbar(out_dir / "colorbar.png", normalization=normalization, cmap=cmap)

    selected = select_top_k_outcomes(labels, scores, threshold=threshold, top_k=top_k)
    rows: list[dict[str, Any]] = []
    for outcome in _OUTCOME_ORDER:
        for rank, sample_index in enumerate(selected[outcome], start=1):
            metadata = metadata_rows[sample_index]
            image_path = metadata.get(image_path_key)
            if not image_path:
                raise ValueError(f"metadata row {sample_index} missing {image_path_key}")
            stem = f"{outcome}_{rank:03d}_sample_{sample_index:05d}"
            paths = save_heatmap_artifacts(
                image_path,
                patch_maps[sample_index],
                out_dir / outcome,
                stem=stem,
                normalization=normalization,
                alpha=alpha,
                cmap=cmap,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "rank": rank,
                    "sample_index": sample_index,
                    "image_path": metadata.get("image_path", ""),
                    "label": int(labels[sample_index]),
                    "score": float(scores[sample_index]),
                    "prediction": int(scores[sample_index] >= threshold),
                    "threshold": float(threshold),
                    **_relative_path_fields(paths, out_dir),
                }
            )

    _write_heatmap_manifest(out_dir / "heatmap_manifest.csv", rows)
    (out_dir / "heatmap_normalization.json").write_text(
        json.dumps(
            {
                **asdict(normalization),
                "alpha": alpha,
                "cmap": cmap,
                "top_k": top_k,
                "selected_count": len(rows),
                "colorbar_file": "colorbar.png",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def save_heatmap_colorbar(
    out_path: str | Path,
    *,
    normalization: HeatmapNormalization,
    cmap: str = "magma",
) -> None:
    """Save the run-level heatmap colorbar used by all overlay images."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.2, 3.2))
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=normalization.vmin, vmax=normalization.vmax),
        cmap=cmap,
    )
    fig.colorbar(mappable, cax=ax)
    ax.set_ylabel("Anomaly score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _validate_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("anomaly_map must be a 2D array")
    if arr.size == 0:
        raise ValueError("anomaly_map must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("anomaly_map must contain only finite values")
    return arr


def _rank_indices(indices: list[int], scores: np.ndarray, *, descending: bool) -> list[int]:
    if descending:
        return sorted(indices, key=lambda idx: (-float(scores[idx]), idx))
    return sorted(indices, key=lambda idx: (float(scores[idx]), idx))


def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip())
    return cleaned or "sample"


def _relative_path_fields(paths: dict[str, Path], root: Path) -> dict[str, str]:
    return {key: path.relative_to(root).as_posix() for key, path in paths.items()}


def _write_heatmap_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "outcome",
        "rank",
        "sample_index",
        "image_path",
        "label",
        "score",
        "prediction",
        "threshold",
        "original_file",
        "heatmap_file",
        "overlay_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

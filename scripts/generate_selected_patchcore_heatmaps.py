#!/usr/bin/env python
"""Generate category-targeted PatchCore heatmaps from a completed evaluation run."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.transforms import build_transforms
from retinal_ood.data.dataset import collate_manifest_batch
from retinal_ood.models.patchcore import PatchCoreDetector
from retinal_ood.utils.io import read_yaml
from retinal_ood.visualization.heatmaps import (
    compute_heatmap_normalization,
    save_heatmap_artifacts,
    save_heatmap_colorbar,
)


class SelectedImageDataset(Dataset):
    """Small in-memory dataset for selected score rows."""

    def __init__(self, rows: pd.DataFrame, *, root_dir: str | Path, transform: Any) -> None:
        self.rows = rows.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, Any]]:
        row = self.rows.iloc[index].to_dict()
        image_path = self.root_dir / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row


def generate_selected_patchcore_heatmaps(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    scores_csv: str | Path,
    out_dir: str | Path,
    categories: Sequence[str],
    per_category: int,
    alpha: float = 0.45,
    cmap: str = "magma",
) -> Path:
    """Generate selected heatmaps and return the manifest path."""
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    config = read_yaml(config_path)
    data_cfg = config.get("data", {})
    root_dir = Path(data_cfg.get("root_dir", "data"))
    scores = pd.read_csv(scores_csv)
    selected = _select_category_rows(scores, categories=categories, per_category=per_category)
    if selected.empty:
        raise ValueError("No selected OOD rows matched the requested categories")

    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        normalize=data_cfg.get("normalize", "imagenet"),
    )
    dataset = SelectedImageDataset(selected, root_dir=root_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        collate_fn=collate_manifest_batch,
    )

    detector = _load_detector(config, checkpoint)
    result = detector.predict_scores(loader, return_patch_maps=True)
    if not isinstance(result, tuple):
        raise ValueError("PatchCore detector did not return patch maps")
    detector_scores, patch_maps = result
    patch_maps = list(patch_maps)
    if len(patch_maps) != len(selected):
        raise ValueError("PatchCore returned a different number of patch maps than selected rows")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalization = compute_heatmap_normalization(patch_maps)
    save_heatmap_colorbar(out_dir / "colorbar.png", normalization=normalization, cmap=cmap)

    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(selected.iterrows()):
        ood_type = str(row.get("ood_type", "ood"))
        subtype = str(row.get("ood_subtype", ood_type))
        stem = f"{ood_type}_{index + 1:03d}_{_safe_stem(subtype)}"
        image_path = root_dir / str(row["image_path"])
        paths = save_heatmap_artifacts(
            image_path,
            patch_maps[index],
            out_dir / ood_type,
            stem=stem,
            normalization=normalization,
            alpha=alpha,
            cmap=cmap,
        )
        score = float(detector_scores[index]) if index < len(detector_scores) else float(row["score"])
        threshold = _row_threshold(row)
        rows.append(
            {
                "outcome": "selected",
                "rank": int(row.get("_category_rank", index + 1)),
                "sample_index": index,
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "score": score,
                "prediction": int(score >= threshold),
                "threshold": threshold,
                "original_file": paths["original_file"].relative_to(out_dir).as_posix(),
                "heatmap_file": paths["heatmap_file"].relative_to(out_dir).as_posix(),
                "overlay_file": paths["overlay_file"].relative_to(out_dir).as_posix(),
            }
        )

    manifest = out_dir / "heatmap_manifest.csv"
    _write_manifest(manifest, rows)
    return manifest


def _select_category_rows(
    scores: pd.DataFrame,
    *,
    categories: Sequence[str],
    per_category: int,
) -> pd.DataFrame:
    required = ["image_path", "label", "ood_type", "score"]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"scores.csv missing required columns: {missing}")
    if per_category <= 0:
        raise ValueError("per_category must be positive")

    table = scores.copy()
    table["label"] = table["label"].astype(int)
    table["score"] = table["score"].astype(float)
    selected: list[pd.DataFrame] = []
    for category in categories:
        rows = table[(table["label"] == 1) & (table["ood_type"].astype(str) == category)]
        rows = rows.sort_values(["score", "image_path"], ascending=[False, True], kind="stable")
        rows = rows.head(per_category).copy()
        rows["_category_rank"] = np.arange(1, len(rows) + 1)
        selected.append(rows)
    if not selected:
        return pd.DataFrame(columns=table.columns)
    return pd.concat(selected, ignore_index=True)


def _load_detector(config: dict[str, Any], checkpoint: str | Path) -> PatchCoreDetector:
    detector = PatchCoreDetector.load(checkpoint)
    model_cfg = config.get("model", {})
    device = _resolve_device(str(model_cfg.get("device", detector.config.device)))
    detector.config.device = device
    detector.device = torch.device(device)
    if detector.feature_extractor is not None:
        detector.feature_extractor.to(detector.device)
        detector.feature_extractor.eval()
    return detector


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _row_threshold(row: pd.Series) -> float:
    if "threshold" not in row or pd.isna(row["threshold"]):
        return float("inf")
    return float(row["threshold"])


def _safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("_") or "example"


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _parse_categories(values: list[str] | None) -> list[str]:
    categories: list[str] = []
    for value in values or []:
        categories.extend(part.strip() for part in value.split(",") if part.strip())
    return categories or ["sensory_artifact", "modality_shift", "semantic_outlier"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved PatchCore YAML config")
    parser.add_argument("--checkpoint", required=True, help="PatchCore memory bank .npz")
    parser.add_argument("--scores-csv", required=True, help="Evaluation scores.csv")
    parser.add_argument("--out-dir", required=True, help="Heatmap output directory")
    parser.add_argument("--category", action="append", help="OOD category to select")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", default="magma")
    args = parser.parse_args()

    manifest = generate_selected_patchcore_heatmaps(
        config_path=args.config,
        checkpoint=args.checkpoint,
        scores_csv=args.scores_csv,
        out_dir=args.out_dir,
        categories=_parse_categories(args.category),
        per_category=args.per_category,
        alpha=args.alpha,
        cmap=args.cmap,
    )
    print(f"Saved selected heatmap manifest to {manifest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train lightweight unsupervised baselines on ID FAF images only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.baselines.feature_distance import FeatureDistanceDetector
from retinal_ood.baselines.image_statistics import ImageStatisticsDetector
from retinal_ood.data.dataset import ManifestImageDataset, collate_manifest_batch
from retinal_ood.data.transforms import build_transforms
from retinal_ood.utils.io import read_yaml, write_json
from retinal_ood.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _require_config_value(config: dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {}).get(key)
    if value in (None, ""):
        raise ValueError(f"{section}.{key} must be set")
    return value


def train_from_config(config: dict[str, Any]) -> Path:
    """Train a lightweight baseline from config and return checkpoint path."""
    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    seed = int(project_cfg.get("seed", 42))
    set_seed(seed)
    model_name = str(model_cfg.get("name", "")).lower()
    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", model_name != "image_statistics")),
        normalize=data_cfg.get("normalize", "minmax" if model_name == "image_statistics" else "imagenet"),
    )
    dataset = ManifestImageDataset(
        _require_config_value(config, "data", "train_manifest"),
        root_dir=data_cfg.get("root_dir"),
        transform=transform,
    ).id_subset()
    if len(dataset) == 0:
        raise ValueError("Baseline training requires at least one ID image with label=0")
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", model_cfg.get("batch_size", 32))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_manifest_batch,
    )
    detector = _build_detector(model_cfg)
    detector.fit(loader)

    run_name = str(project_cfg.get("run_name", model_name or "baseline"))
    run_dir = Path(output_cfg.get("runs_dir", "runs")) / run_name
    checkpoint_path = run_dir / str(output_cfg.get("checkpoint_name", "baseline_model.npz"))
    detector.save(checkpoint_path)
    write_json(run_dir / "resolved_config.json", config)
    write_json(
        run_dir / "training_metrics.json",
        {
            "model": model_name,
            "train_id_images": len(dataset),
            "feature_dim": getattr(detector, "feature_dim", None),
            "device": str(getattr(detector, "device", "cpu")),
            "checkpoint": checkpoint_path.name,
        },
    )
    return checkpoint_path


def _build_detector(model_cfg: dict[str, Any]) -> ImageStatisticsDetector | FeatureDistanceDetector:
    model_name = str(model_cfg.get("name", "")).lower()
    if model_name == "image_statistics":
        return ImageStatisticsDetector()
    if model_name == "global_feature_knn":
        return FeatureDistanceDetector(
            mode="knn",
            backbone=str(model_cfg.get("backbone", "resnet50")),
            layers=tuple(model_cfg.get("layers", ["layer4"])),
            nearest_neighbors=int(model_cfg.get("nearest_neighbors", 1)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            feature_backend=str(model_cfg.get("feature_backend", "torchvision")),
            device=_resolve_device(str(model_cfg.get("device", "auto"))),
        )
    if model_name == "mahalanobis_feature":
        return FeatureDistanceDetector(
            mode="mahalanobis",
            backbone=str(model_cfg.get("backbone", "resnet50")),
            layers=tuple(model_cfg.get("layers", ["layer4"])),
            covariance_regularization=float(model_cfg.get("covariance_regularization", 1e-3)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            feature_backend=str(model_cfg.get("feature_backend", "torchvision")),
            device=_resolve_device(str(model_cfg.get("device", "auto"))),
        )
    raise ValueError(f"Unsupported baseline model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to baseline YAML config")
    args = parser.parse_args()
    checkpoint_path = train_from_config(read_yaml(args.config))
    print(f"Saved baseline checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

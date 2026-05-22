#!/usr/bin/env python
"""Train a PatchCore detector on ID/valid FAF images only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.dataset import ManifestImageDataset
from retinal_ood.data.transforms import build_transforms
from retinal_ood.models.patchcore import PatchCoreConfig, PatchCoreDetector
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


def _build_patchcore_config(config: dict[str, Any]) -> PatchCoreConfig:
    model_cfg = config.get("model", {})
    project_cfg = config.get("project", {})
    return PatchCoreConfig(
        backbone=str(model_cfg.get("backbone", "resnet50")),
        layers=tuple(model_cfg.get("layers", ["layer2", "layer3"])),
        coreset_ratio=float(model_cfg.get("coreset_ratio", 0.1)),
        nearest_neighbors=int(model_cfg.get("nearest_neighbors", 1)),
        max_train_patches=model_cfg.get("max_train_patches"),
        random_seed=int(project_cfg.get("seed", model_cfg.get("random_seed", 42))),
        pretrained=bool(model_cfg.get("pretrained", True)),
        feature_backend=str(model_cfg.get("feature_backend", "torchvision")),
        device=_resolve_device(str(model_cfg.get("device", "auto"))),
    )


def _build_train_dataset(config: dict[str, Any]) -> ManifestImageDataset:
    data_cfg = config.get("data", {})
    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        normalize=data_cfg.get("normalize", "imagenet"),
    )
    dataset = ManifestImageDataset(
        _require_config_value(config, "data", "train_manifest"),
        root_dir=data_cfg.get("root_dir"),
        transform=transform,
    )
    id_dataset = dataset.id_subset()
    if len(id_dataset) == 0:
        raise ValueError("PatchCore training requires at least one ID image with label=0")
    return id_dataset


def _build_dataloader(config: dict[str, Any], dataset: ManifestImageDataset) -> DataLoader:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", model_cfg.get("batch_size", 32))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )


def train_from_config(
    config: dict[str, Any],
    *,
    feature_extractor: nn.Module | None = None,
) -> Path:
    """Fit PatchCore from config and return the saved memory-bank checkpoint path."""
    project_cfg = config.get("project", {})
    output_cfg = config.get("output", {})

    seed = int(project_cfg.get("seed", 42))
    set_seed(seed)
    train_dataset = _build_train_dataset(config)
    dataloader = _build_dataloader(config, train_dataset)
    detector = PatchCoreDetector(
        _build_patchcore_config(config),
        feature_extractor=feature_extractor,
    )
    detector.fit(dataloader)

    run_name = str(project_cfg.get("run_name", "patchcore_run"))
    run_dir = Path(output_cfg.get("runs_dir", "runs")) / run_name
    checkpoint_name = str(output_cfg.get("checkpoint_name", "patchcore_memory.npz"))
    checkpoint_path = run_dir / checkpoint_name
    detector.save(checkpoint_path)
    write_json(run_dir / "resolved_config.json", config)
    write_json(
        run_dir / "training_metrics.json",
        {
            "model": "patchcore",
            "backbone": detector.config.backbone,
            "layers": list(detector.config.layers),
            "train_id_images": len(train_dataset),
            "memory_bank_patches": int(detector.memory_bank.shape[0]) if detector.memory_bank is not None else 0,
            "feature_dim": detector.feature_dim,
            "coreset_ratio": detector.config.coreset_ratio,
            "max_train_patches": detector.config.max_train_patches,
            "nearest_neighbors": detector.config.nearest_neighbors,
            "device": str(detector.device),
            "checkpoint": checkpoint_path.name,
        },
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    checkpoint_path = train_from_config(read_yaml(args.config))
    print(f"Saved PatchCore memory bank to {checkpoint_path}")


if __name__ == "__main__":
    main()

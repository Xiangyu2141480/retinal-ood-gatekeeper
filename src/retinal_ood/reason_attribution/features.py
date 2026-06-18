"""Image-derived features for post-rejection reason attribution.

These features are used only by the optional Stage 2 explanation module. They are not
used to train or modify the Stage 1 ID-only OOD gatekeeper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

FeatureMode = Literal["image_statistics", "cnn"]

STATISTICS_FEATURE_NAMES = (
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "entropy",
    "edge_density",
    "foreground_ratio",
    "black_border_ratio",
    *[f"hist_bin_{index:02d}" for index in range(16)],
)


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for image-only Stage 2 feature extraction."""

    feature_mode: FeatureMode = "image_statistics"
    image_size: int = 224
    backbone: str = "resnet50"
    layers: tuple[str, ...] = ("layer4",)
    pretrained: bool = True
    feature_backend: str = "torchvision"
    grayscale_to_rgb: bool = True
    normalize: str | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class FeatureTable:
    """Feature matrix with companion manifest metadata kept out of the model input."""

    features: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]


def extract_features_from_manifest(
    manifest_path: str | Path,
    *,
    root_dir: str | Path,
    config: FeatureExtractionConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> FeatureTable:
    """Extract image-derived features for all rows in a manifest.

    Manifest metadata is returned for targets/grouping only. It is never concatenated into
    the feature matrix.
    """
    cfg = config or FeatureExtractionConfig()
    metadata = pd.read_csv(manifest_path)
    _validate_manifest_for_feature_extraction(metadata, Path(manifest_path))
    if cfg.feature_mode == "image_statistics":
        return _extract_statistics(metadata, root_dir=Path(root_dir), config=cfg)
    if cfg.feature_mode == "cnn":
        return _extract_cnn(
            manifest_path,
            root_dir=Path(root_dir),
            metadata=metadata,
            config=cfg,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    raise ValueError(f"Unsupported feature_mode: {cfg.feature_mode}")


def _validate_manifest_for_feature_extraction(metadata: pd.DataFrame, manifest_path: Path) -> None:
    if "image_path" not in metadata.columns:
        raise ValueError(f"Manifest missing image_path column: {manifest_path}")
    if metadata.empty:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")


def _extract_statistics(
    metadata: pd.DataFrame,
    *,
    root_dir: Path,
    config: FeatureExtractionConfig,
) -> FeatureTable:
    features = np.vstack(
        [
            _statistics_from_image(
                _resolve_image_path(path, root_dir=root_dir),
                image_size=config.image_size,
                grayscale_to_rgb=config.grayscale_to_rgb,
            )
            for path in metadata["image_path"].tolist()
        ]
    ).astype(np.float32)
    return FeatureTable(
        features=features,
        metadata=metadata.reset_index(drop=True).copy(),
        feature_names=STATISTICS_FEATURE_NAMES,
    )


def _extract_cnn(
    manifest_path: str | Path,
    *,
    root_dir: Path,
    metadata: pd.DataFrame,
    config: FeatureExtractionConfig,
    batch_size: int,
    num_workers: int,
) -> FeatureTable:
    import torch
    from torch.utils.data import DataLoader

    from retinal_ood.data.dataset import ManifestImageDataset, collate_manifest_batch
    from retinal_ood.data.transforms import build_transforms
    from retinal_ood.models.feature_extractor import build_feature_extractor

    transform = build_transforms(
        image_size=config.image_size,
        grayscale_to_rgb=config.grayscale_to_rgb,
        normalize=config.normalize or "imagenet",
    )
    dataset = ManifestImageDataset(
        manifest_path,
        root_dir=root_dir,
        transform=transform,
        require_files=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_manifest_batch,
    )
    device = _resolve_device(config.device)
    extractor = build_feature_extractor(
        backbone=config.backbone,
        layers=config.layers,
        pretrained=config.pretrained,
        backend=config.feature_backend,  # type: ignore[arg-type]
    )
    extractor.to(device)
    extractor.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device=device, dtype=torch.float32)
            outputs = extractor(images)
            batches.append(pool_feature_outputs(outputs, selected_layers=config.layers))
    features = _concat_features(batches)
    names = tuple(f"{layer}_feature_{index:04d}" for layer in config.layers for index in range(features.shape[1]))
    return FeatureTable(
        features=features,
        metadata=metadata.reset_index(drop=True).copy(),
        feature_names=names[: features.shape[1]],
    )


def pool_feature_outputs(outputs: object, *, selected_layers: tuple[str, ...] | list[str]) -> np.ndarray:
    """Global-pool frozen CNN outputs into a 2D feature matrix."""
    import torch

    if torch.is_tensor(outputs):
        features = outputs
    elif isinstance(outputs, dict):
        tensors: list[torch.Tensor] = []
        for layer in selected_layers:
            if layer not in outputs:
                continue
            tensor = outputs[layer]
            if tensor.ndim == 4:
                tensor = tensor.mean(dim=(2, 3))
            elif tensor.ndim > 2:
                tensor = tensor.flatten(start_dim=1)
            tensors.append(tensor)
        if not tensors:
            raise ValueError("Feature extractor returned no requested layers")
        features = torch.cat(tensors, dim=1)
    else:
        raise TypeError("Feature extractor must return a tensor or mapping of tensors")
    if features.ndim > 2:
        features = features.flatten(start_dim=1)
    return features.detach().cpu().numpy().astype(np.float32)


def _resolve_device(requested: str) -> object:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _concat_features(batches: list[np.ndarray]) -> np.ndarray:
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def _statistics_from_image(path: Path, *, image_size: int, grayscale_to_rgb: bool) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")
    image = Image.open(path)
    if grayscale_to_rgb:
        image = image.convert("L").convert("RGB")
    else:
        image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.moveaxis(array, 2, 0)
    flat = chw.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std())
    min_value = float(flat.min())
    max_value = float(flat.max())
    entropy = _entropy(flat)
    edge_density = _edge_density(chw)
    foreground_ratio = float((flat > (mean + 0.25 * std)).mean())
    black_border_ratio = _black_border_ratio(chw)
    hist = _histogram(flat)
    numeric = np.asarray(
        [
            mean,
            std,
            min_value,
            max_value,
            entropy,
            edge_density,
            foreground_ratio,
            black_border_ratio,
        ],
        dtype=np.float32,
    )
    return np.concatenate([numeric, hist]).astype(np.float32)


def _entropy(flat: np.ndarray) -> float:
    hist = np.clip(_histogram(flat), 1e-8, None)
    return float(-(hist * np.log2(hist)).sum())


def _histogram(flat: np.ndarray, bins: int = 16) -> np.ndarray:
    min_value = float(flat.min())
    max_value = float(flat.max())
    normalized = (flat - min_value) / max(max_value - min_value, 1e-6)
    bucket = np.clip((normalized * bins).astype(int), 0, bins - 1)
    hist = np.bincount(bucket, minlength=bins).astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def _edge_density(chw: np.ndarray) -> float:
    dx = np.abs(chw[:, :, 1:] - chw[:, :, :-1]).mean()
    dy = np.abs(chw[:, 1:, :] - chw[:, :-1, :]).mean()
    return float(dx + dy)


def _black_border_ratio(chw: np.ndarray) -> float:
    top = chw[:, :1, :].reshape(-1)
    bottom = chw[:, -1:, :].reshape(-1)
    left = chw[:, :, :1].reshape(-1)
    right = chw[:, :, -1:].reshape(-1)
    border = np.concatenate([top, bottom, left, right])
    return float((border <= 0.05).mean())


def _resolve_image_path(image_path: str | Path, *, root_dir: Path) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return root_dir / path

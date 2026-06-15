"""Simple image-statistics OOD baseline fitted on ID images only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

FEATURE_NAMES = (
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


class ImageStatisticsDetector:
    """Distance-to-ID-summary detector using low-level image statistics."""

    def __init__(self, *, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.train_id_images: int = 0

    def fit(self, dataloader: DataLoader) -> None:
        features: list[np.ndarray] = []
        total_id = 0
        for batch in dataloader:
            images, labels = _split_batch(batch)
            id_images = _filter_id_images(images, labels)
            if id_images.numel() == 0:
                continue
            total_id += int(id_images.shape[0])
            features.append(extract_image_statistics_features(id_images))
        if total_id == 0 or not features:
            raise ValueError("Image statistics baseline requires at least one ID image")
        matrix = np.concatenate(features, axis=0).astype(np.float32)
        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < self.eps] = 1.0
        self.scale_ = scale.astype(np.float32)
        self.train_id_images = total_id

    def predict_scores(self, dataloader: DataLoader, **kwargs: Any) -> np.ndarray:
        if kwargs.get("return_patch_maps"):
            raise TypeError("Image statistics baseline does not provide patch heatmaps")
        self._ensure_fitted()
        scores: list[np.ndarray] = []
        for batch in dataloader:
            images, _ = _split_batch(batch)
            features = extract_image_statistics_features(images)
            z = (features - self.mean_) / self.scale_
            scores.append(np.sqrt(np.mean(z * z, axis=1)).astype(np.float32))
        if not scores:
            return np.asarray([], dtype=float)
        return np.concatenate(scores, axis=0).astype(float)

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean_,
            scale=self.scale_,
            train_id_images=np.array(self.train_id_images),
            feature_names=np.array(FEATURE_NAMES),
            baseline_type=np.array("image_statistics"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ImageStatisticsDetector":
        data = np.load(path, allow_pickle=False)
        detector = cls()
        detector.mean_ = data["mean"].astype(np.float32)
        detector.scale_ = data["scale"].astype(np.float32)
        detector.train_id_images = int(data["train_id_images"])
        return detector

    def _ensure_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("ImageStatisticsDetector must be fitted before scoring")


def extract_image_statistics_features(images: torch.Tensor) -> np.ndarray:
    """Extract deterministic low-level image features from a tensor batch."""
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape B,C,H,W, got {tuple(images.shape)}")
    images = images.detach().to(dtype=torch.float32, device="cpu")
    batch_size = int(images.shape[0])
    flat = images.reshape(batch_size, -1)
    means = flat.mean(dim=1)
    stds = flat.std(dim=1, unbiased=False)
    mins = flat.min(dim=1).values
    maxes = flat.max(dim=1).values
    entropy = _entropy(flat)
    edge_density = _edge_density(images)
    foreground_ratio = (flat > (means[:, None] + 0.25 * stds[:, None])).float().mean(dim=1)
    black_border_ratio = _black_border_ratio(images)
    hist = _histograms(flat)
    numeric = torch.stack(
        [means, stds, mins, maxes, entropy, edge_density, foreground_ratio, black_border_ratio],
        dim=1,
    )
    return torch.cat([numeric, hist], dim=1).numpy().astype(np.float32)


def _entropy(flat: torch.Tensor) -> torch.Tensor:
    hist = _histograms(flat).clamp_min(1e-8)
    return -(hist * torch.log2(hist)).sum(dim=1)


def _histograms(flat: torch.Tensor, bins: int = 16) -> torch.Tensor:
    min_values = flat.min(dim=1).values[:, None]
    max_values = flat.max(dim=1).values[:, None]
    normalized = (flat - min_values) / (max_values - min_values).clamp_min(1e-6)
    bucket = torch.clamp((normalized * bins).long(), min=0, max=bins - 1)
    hist = torch.zeros((flat.shape[0], bins), dtype=torch.float32)
    hist.scatter_add_(1, bucket, torch.ones_like(normalized, dtype=torch.float32))
    return hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)


def _edge_density(images: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean(dim=(1, 2, 3))
    dy = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean(dim=(1, 2, 3))
    return dx + dy


def _black_border_ratio(images: torch.Tensor) -> torch.Tensor:
    top = images[:, :, :1, :].reshape(images.shape[0], -1)
    bottom = images[:, :, -1:, :].reshape(images.shape[0], -1)
    left = images[:, :, :, :1].reshape(images.shape[0], -1)
    right = images[:, :, :, -1:].reshape(images.shape[0], -1)
    border = torch.cat([top, bottom, left, right], dim=1)
    return (border <= 0.05).float().mean(dim=1)


def _split_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_tensor(batch):
        return batch, None
    if isinstance(batch, (tuple, list)) and batch:
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        if not torch.is_tensor(images):
            raise TypeError("Expected batch images to be a torch.Tensor")
        if labels is not None and not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        return images, labels
    raise TypeError("Expected a tensor batch or tuple/list batch")


def _filter_id_images(images: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    if labels is None:
        return images
    labels = labels.to(device=images.device)
    return images[labels == 0]

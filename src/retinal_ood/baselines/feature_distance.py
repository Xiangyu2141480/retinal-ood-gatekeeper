"""Global feature-distance OOD baselines fitted on ID images only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from retinal_ood.models.feature_extractor import build_feature_extractor

SUPPORTED_MODES = {"knn", "mahalanobis"}


class FeatureDistanceDetector:
    """Global pooled feature kNN or Mahalanobis OOD detector."""

    def __init__(
        self,
        *,
        mode: str,
        backbone: str = "resnet50",
        layers: tuple[str, ...] = ("layer4",),
        nearest_neighbors: int = 1,
        covariance_regularization: float = 1e-3,
        pretrained: bool = True,
        feature_backend: str = "torchvision",
        device: str = "cpu",
        feature_extractor: nn.Module | None = None,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported feature-distance mode: {mode}")
        if nearest_neighbors <= 0:
            raise ValueError("nearest_neighbors must be positive")
        if covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be positive")
        self.mode = mode
        self.backbone = backbone
        self.layers = tuple(layers)
        self.nearest_neighbors = int(nearest_neighbors)
        self.covariance_regularization = float(covariance_regularization)
        self.pretrained = bool(pretrained)
        self.feature_backend = feature_backend
        self.device = torch.device(device)
        self.feature_extractor = feature_extractor or build_feature_extractor(
            backbone=backbone,
            layers=self.layers,
            pretrained=pretrained,
            backend=feature_backend,  # type: ignore[arg-type]
        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.memory_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None
        self.train_id_images: int = 0
        self.feature_dim: int | None = None

    def fit(self, dataloader: DataLoader) -> None:
        feature_batches: list[np.ndarray] = []
        total_id = 0
        with torch.no_grad():
            for batch in dataloader:
                images, labels = _split_batch(batch)
                images = images.to(device=self.device, dtype=torch.float32)
                id_images = _filter_id_images(images, labels)
                if id_images.numel() == 0:
                    continue
                total_id += int(id_images.shape[0])
                feature_batches.append(self._extract_features(id_images))
        if total_id == 0 or not feature_batches:
            raise ValueError("Feature-distance baseline requires at least one ID image")
        features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        self.memory_ = features
        self.mean_ = features.mean(axis=0).astype(np.float32)
        self.feature_dim = int(features.shape[1])
        self.train_id_images = total_id
        if self.mode == "mahalanobis":
            self.inv_cov_ = _regularized_inverse_covariance(features, self.covariance_regularization)

    def predict_scores(self, dataloader: DataLoader, **kwargs: Any) -> np.ndarray:
        if kwargs.get("return_patch_maps"):
            raise TypeError("Feature-distance baseline does not provide patch heatmaps")
        self._ensure_fitted()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                images, _ = _split_batch(batch)
                images = images.to(device=self.device, dtype=torch.float32)
                features = self._extract_features(images)
                if self.mode == "knn":
                    scores.append(_knn_scores(features, self.memory_, self.nearest_neighbors))
                else:
                    scores.append(_mahalanobis_scores(features, self.mean_, self.inv_cov_))
        if not scores:
            return np.asarray([], dtype=float)
        return np.concatenate(scores, axis=0).astype(float)

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mode=np.array(self.mode),
            backbone=np.array(self.backbone),
            layers=np.array(self.layers),
            nearest_neighbors=np.array(self.nearest_neighbors),
            covariance_regularization=np.array(self.covariance_regularization),
            pretrained=np.array(self.pretrained),
            feature_backend=np.array(self.feature_backend),
            memory=self.memory_,
            mean=self.mean_,
            inv_cov=np.empty((0, 0), dtype=np.float32) if self.inv_cov_ is None else self.inv_cov_,
            train_id_images=np.array(self.train_id_images),
            feature_dim=np.array(-1 if self.feature_dim is None else self.feature_dim),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_extractor: nn.Module | None = None,
        build_extractor: bool = True,
        device: str = "cpu",
    ) -> "FeatureDistanceDetector":
        data = np.load(path, allow_pickle=False)
        layers = tuple(str(value) for value in data["layers"].tolist())
        extractor = feature_extractor
        if not build_extractor and extractor is None:
            detector = object.__new__(cls)
            detector.feature_extractor = None
        else:
            detector = cls(
                mode=str(data["mode"]),
                backbone=str(data["backbone"]),
                layers=layers,
                nearest_neighbors=int(data["nearest_neighbors"]),
                covariance_regularization=float(data["covariance_regularization"]),
                pretrained=bool(data["pretrained"]),
                feature_backend=str(data["feature_backend"]),
                device=device,
                feature_extractor=extractor,
            )
        detector.mode = str(data["mode"])
        detector.backbone = str(data["backbone"])
        detector.layers = layers
        detector.nearest_neighbors = int(data["nearest_neighbors"])
        detector.covariance_regularization = float(data["covariance_regularization"])
        detector.pretrained = bool(data["pretrained"])
        detector.feature_backend = str(data["feature_backend"])
        detector.device = torch.device(device)
        if detector.feature_extractor is not None:
            detector.feature_extractor.to(detector.device)
            detector.feature_extractor.eval()
        detector.memory_ = data["memory"].astype(np.float32)
        detector.mean_ = data["mean"].astype(np.float32)
        inv_cov = data["inv_cov"]
        detector.inv_cov_ = None if inv_cov.size == 0 else inv_cov.astype(np.float32)
        detector.train_id_images = int(data["train_id_images"])
        detector.feature_dim = int(data["feature_dim"]) if int(data["feature_dim"]) >= 0 else None
        return detector

    def _extract_features(self, images: torch.Tensor) -> np.ndarray:
        if self.feature_extractor is None:
            raise ValueError("FeatureDistanceDetector needs a feature_extractor for scoring")
        outputs = self.feature_extractor(images)
        if torch.is_tensor(outputs):
            features = outputs
        elif isinstance(outputs, dict):
            tensors = []
            selected_layers = [layer for layer in self.layers if layer in outputs] or list(outputs)
            for layer in selected_layers:
                tensor = outputs[layer]
                if tensor.ndim == 4:
                    tensor = tensor.mean(dim=(2, 3))
                elif tensor.ndim > 2:
                    tensor = tensor.flatten(start_dim=1)
                tensors.append(tensor)
            features = torch.cat(tensors, dim=1)
        else:
            raise TypeError("Feature extractor must return a tensor or mapping of tensors")
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features.detach().cpu().numpy().astype(np.float32)

    def _ensure_fitted(self) -> None:
        if self.memory_ is None or self.mean_ is None:
            raise ValueError("FeatureDistanceDetector must be fitted before scoring")
        if self.mode == "mahalanobis" and self.inv_cov_ is None:
            raise ValueError("Mahalanobis detector is missing inverse covariance")


def _regularized_inverse_covariance(features: np.ndarray, regularization: float) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    if features.shape[0] <= 1:
        covariance = np.zeros((features.shape[1], features.shape[1]), dtype=np.float32)
    else:
        covariance = np.cov(centered, rowvar=False).astype(np.float32)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0], dtype=np.float32) * regularization
    return np.linalg.pinv(covariance).astype(np.float32)


def _knn_scores(features: np.ndarray, memory: np.ndarray, nearest_neighbors: int) -> np.ndarray:
    distances = _pairwise_euclidean_distances(features, memory)
    if nearest_neighbors == 1:
        return distances.min(axis=1).astype(np.float32)
    nearest = np.partition(distances, kth=nearest_neighbors - 1, axis=1)[:, :nearest_neighbors]
    return nearest.mean(axis=1).astype(np.float32)


def _mahalanobis_scores(features: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    centered = features - mean[None, :]
    squared = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)


def _pairwise_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.sqrt(squared).astype(np.float32)


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

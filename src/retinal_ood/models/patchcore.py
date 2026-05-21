"""PatchCore-style feature memory bank and scoring.

PatchCore learns a bank of local CNN patch features from ID/normal images only. Inference
scores incoming images by the distance between their patch features and that normal memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from retinal_ood.models.feature_extractor import build_feature_extractor


@dataclass
class PatchCoreConfig:
    backbone: str = "resnet50"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_ratio: float = 0.1
    nearest_neighbors: int = 1
    max_train_patches: int | None = None
    random_seed: int = 42
    pretrained: bool = True
    feature_backend: str = "torchvision"
    device: str = "cpu"


class PatchCoreDetector:
    """PatchCore detector that can fit and persist a normal patch-feature memory bank.

    ``fit`` filters labelled batches to ID samples (`label == 0`) so OOD examples are never
    used to build the unsupervised reference memory.
    """

    def __init__(
        self,
        config: PatchCoreConfig,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self.feature_extractor = feature_extractor or build_feature_extractor(
            backbone=config.backbone,
            layers=config.layers,
            pretrained=config.pretrained,
            backend=config.feature_backend,  # type: ignore[arg-type]
        )
        self.device = torch.device(config.device)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.memory_bank: np.ndarray | None = None
        self.feature_dim: int | None = None

    def fit(self, dataloader: DataLoader) -> None:
        """Build the normal patch-feature memory bank from ID training images."""
        if self.feature_extractor is None:
            raise ValueError("PatchCoreDetector.load() returns a fitted detector; create a new detector to refit")
        patch_batches: list[np.ndarray] = []
        total_id_images = 0
        with torch.no_grad():
            for batch in dataloader:
                images, labels = _split_batch(batch)
                images = images.to(device=self.device, dtype=torch.float32)
                id_images = _filter_id_images(images, labels)
                if id_images.numel() == 0:
                    continue
                total_id_images += int(id_images.shape[0])
                features = self.feature_extractor(id_images)
                embeddings = patch_embeddings_from_feature_maps(features, self.config.layers)
                patch_batches.append(embeddings.detach().cpu().numpy().astype(np.float32))

        if total_id_images == 0:
            raise ValueError("PatchCore memory bank requires at least one ID image with label=0")
        if not patch_batches:
            raise ValueError("PatchCore feature extractor returned no patch embeddings")

        patches = np.concatenate(patch_batches, axis=0)
        self.feature_dim = int(patches.shape[1])
        self.memory_bank = deterministic_coreset(
            patches,
            coreset_ratio=self.config.coreset_ratio,
            max_patches=self.config.max_train_patches,
            seed=self.config.random_seed,
        )
        if self.memory_bank.size == 0:
            raise ValueError("PatchCore memory bank is empty after coreset selection")

    def predict_scores(
        self,
        dataloader: DataLoader,
        *,
        return_patch_maps: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """Score images by nearest-neighbor distance to the fitted memory bank.

        Higher scores mean the image is farther from the ID feature memory and therefore more
        anomalous. The image score is the maximum patch anomaly score. Optional patch maps are
        returned at the first selected feature layer's spatial resolution; image-size overlays
        are handled by the visualization task.
        """
        self._ensure_ready_for_inference()
        image_scores: list[float] = []
        patch_maps: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                images, _ = _split_batch(batch)
                images = images.to(device=self.device, dtype=torch.float32)
                features = self.feature_extractor(images)
                embeddings, grid_shape = patch_embeddings_and_grid(features, self.config.layers)
                patch_scores = nearest_neighbor_patch_scores(
                    embeddings.detach().cpu().numpy().astype(np.float32),
                    self.memory_bank,
                    nearest_neighbors=self.config.nearest_neighbors,
                )
                batch_size, height, width = grid_shape
                maps = patch_scores.reshape(batch_size, height, width)
                image_scores.extend(maps.reshape(batch_size, -1).max(axis=1).astype(float).tolist())
                if return_patch_maps:
                    patch_maps.extend(maps.astype(np.float32))

        scores = np.asarray(image_scores, dtype=float)
        if return_patch_maps:
            return scores, patch_maps
        return scores

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None:
            raise ValueError("Cannot save PatchCoreDetector before fit() builds a memory bank")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank,
            backbone=np.array(self.config.backbone),
            layers=np.array(self.config.layers),
            coreset_ratio=np.array(self.config.coreset_ratio),
            nearest_neighbors=np.array(self.config.nearest_neighbors),
            max_train_patches=np.array(-1 if self.config.max_train_patches is None else self.config.max_train_patches),
            random_seed=np.array(self.config.random_seed),
            pretrained=np.array(self.config.pretrained),
            feature_backend=np.array(self.config.feature_backend),
            feature_dim=np.array(-1 if self.feature_dim is None else self.feature_dim),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_extractor: nn.Module | None = None,
        build_feature_extractor: bool = True,
    ) -> "PatchCoreDetector":
        data = np.load(path, allow_pickle=False)
        layers = tuple(str(x) for x in data["layers"].tolist())
        max_train_patches = int(data["max_train_patches"]) if "max_train_patches" in data else -1
        config = PatchCoreConfig(
            backbone=str(data["backbone"]) if "backbone" in data else "resnet50",
            layers=layers,
            coreset_ratio=float(data["coreset_ratio"]) if "coreset_ratio" in data else 0.1,
            nearest_neighbors=int(data["nearest_neighbors"]) if "nearest_neighbors" in data else 1,
            max_train_patches=None if max_train_patches < 0 else max_train_patches,
            random_seed=int(data["random_seed"]) if "random_seed" in data else 42,
            pretrained=bool(data["pretrained"]) if "pretrained" in data else True,
            feature_backend=str(data["feature_backend"]) if "feature_backend" in data else "torchvision",
        )
        if build_feature_extractor:
            detector = cls(config, feature_extractor=feature_extractor)
        else:
            detector = object.__new__(cls)
            detector.config = config
            detector.feature_extractor = feature_extractor
            detector.device = torch.device(config.device)
            if detector.feature_extractor is not None:
                detector.feature_extractor.to(detector.device)
                detector.feature_extractor.eval()
        detector.memory_bank = data["memory_bank"]
        detector.feature_dim = int(data["feature_dim"]) if "feature_dim" in data and int(data["feature_dim"]) >= 0 else (
            int(detector.memory_bank.shape[1]) if detector.memory_bank.ndim == 2 else None
        )
        return detector

    def _ensure_ready_for_inference(self) -> None:
        if self.memory_bank is None:
            raise ValueError("PatchCoreDetector must be fitted or loaded before predict_scores()")
        if self.memory_bank.ndim != 2 or self.memory_bank.shape[0] == 0:
            raise ValueError("PatchCore memory bank must have shape num_patches, feature_dim")
        if self.feature_extractor is None:
            raise ValueError("PatchCoreDetector needs a feature_extractor for predict_scores()")
        if self.config.nearest_neighbors > self.memory_bank.shape[0]:
            raise ValueError("nearest_neighbors cannot exceed memory bank size")


def patch_embeddings_from_feature_maps(
    features: Mapping[str, torch.Tensor],
    layers: tuple[str, ...],
) -> torch.Tensor:
    """Convert selected CNN feature maps to concatenated per-patch embeddings."""
    patches, _ = patch_embeddings_and_grid(features, layers)
    return patches


def patch_embeddings_and_grid(
    features: Mapping[str, torch.Tensor],
    layers: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Return patch embeddings and their batch/spatial grid shape."""
    if not layers:
        raise ValueError("At least one feature layer is required")
    missing = [layer for layer in layers if layer not in features]
    if missing:
        raise ValueError(f"Feature maps missing requested layers: {missing}")

    first = features[layers[0]]
    if first.ndim != 4:
        raise ValueError(f"Expected feature map shape B,C,H,W, got {tuple(first.shape)}")
    target_size = first.shape[-2:]
    resized: list[torch.Tensor] = []
    batch_size = int(first.shape[0])
    for layer in layers:
        feature = features[layer]
        if feature.ndim != 4:
            raise ValueError(f"Expected feature map shape B,C,H,W for {layer}, got {tuple(feature.shape)}")
        if int(feature.shape[0]) != batch_size:
            raise ValueError("All feature maps must have the same batch dimension")
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        resized.append(feature)

    concatenated = torch.cat(resized, dim=1)
    patches = concatenated.permute(0, 2, 3, 1).reshape(-1, concatenated.shape[1])
    batch_size, _, height, width = concatenated.shape
    return patches.contiguous(), (int(batch_size), int(height), int(width))


def nearest_neighbor_patch_scores(
    patches: np.ndarray,
    memory_bank: np.ndarray,
    *,
    nearest_neighbors: int = 1,
) -> np.ndarray:
    """Return per-patch anomaly scores from kNN distance to the memory bank."""
    patches = np.asarray(patches, dtype=np.float32)
    memory_bank = np.asarray(memory_bank, dtype=np.float32)
    if patches.ndim != 2 or memory_bank.ndim != 2:
        raise ValueError("patches and memory_bank must be 2D arrays")
    if patches.shape[0] == 0:
        raise ValueError("patches must not be empty")
    if memory_bank.shape[0] == 0:
        raise ValueError("memory_bank must not be empty")
    if patches.shape[1] != memory_bank.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: patches have {patches.shape[1]}, "
            f"memory bank has {memory_bank.shape[1]}"
        )
    if nearest_neighbors <= 0:
        raise ValueError("nearest_neighbors must be positive")
    if nearest_neighbors > memory_bank.shape[0]:
        raise ValueError("nearest_neighbors cannot exceed memory bank size")

    distances = _pairwise_euclidean_distances(patches, memory_bank)
    if nearest_neighbors == 1:
        return distances.min(axis=1).astype(np.float32)
    nearest = np.partition(distances, kth=nearest_neighbors - 1, axis=1)[:, :nearest_neighbors]
    return nearest.mean(axis=1).astype(np.float32)


def _pairwise_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.sqrt(squared).astype(np.float32)


def deterministic_coreset(
    patches: np.ndarray,
    *,
    coreset_ratio: float,
    max_patches: int | None,
    seed: int,
) -> np.ndarray:
    """Select a deterministic random coreset from patch embeddings."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError("patches must be a 2D array with shape num_patches, feature_dim")
    if patches.shape[0] == 0:
        raise ValueError("patches must not be empty")
    _validate_coreset_args(coreset_ratio, max_patches)

    target = int(np.ceil(patches.shape[0] * coreset_ratio))
    if max_patches is not None:
        target = min(target, max_patches)
    target = max(1, min(target, patches.shape[0]))
    if target == patches.shape[0]:
        return patches.copy()

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(patches.shape[0], size=target, replace=False))
    return patches[selected].copy()


def _validate_config(config: PatchCoreConfig) -> None:
    _validate_coreset_args(config.coreset_ratio, config.max_train_patches)
    if config.nearest_neighbors <= 0:
        raise ValueError("nearest_neighbors must be positive")


def _validate_coreset_args(coreset_ratio: float, max_patches: int | None) -> None:
    if not 0 < coreset_ratio <= 1:
        raise ValueError("coreset_ratio must be in (0, 1]")
    if max_patches is not None and max_patches <= 0:
        raise ValueError("max_train_patches must be positive when set")


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
    raise TypeError("Expected a tensor batch or a tuple/list whose first item is a tensor")


def _filter_id_images(images: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    if labels is None:
        return images
    labels = labels.to(device=images.device)
    return images[labels == 0]

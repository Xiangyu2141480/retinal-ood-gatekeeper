from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from retinal_ood.models.patchcore import (
    PatchCoreConfig,
    PatchCoreDetector,
    deterministic_coreset,
    patch_embeddings_from_feature_maps,
)


class ToyFeatureExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        layer2 = torch.arange(batch_size * 2 * 4 * 4, dtype=x.dtype, device=x.device).reshape(
            batch_size, 2, 4, 4
        )
        layer3 = torch.ones(batch_size, 3, 2, 2, dtype=x.dtype, device=x.device)
        return {"layer2": layer2, "layer3": layer3}


def test_patch_embeddings_concatenate_layers_at_first_layer_resolution():
    features = {
        "layer2": torch.zeros(2, 2, 4, 4),
        "layer3": torch.ones(2, 3, 2, 2),
    }

    patches = patch_embeddings_from_feature_maps(features, ("layer2", "layer3"))

    assert patches.shape == (2 * 4 * 4, 5)
    assert torch.all(patches[:, :2] == 0)
    assert torch.all(patches[:, 2:] == 1)


def test_patch_embeddings_reject_missing_layer():
    with pytest.raises(ValueError, match="missing requested layers"):
        patch_embeddings_from_feature_maps({"layer2": torch.zeros(1, 2, 4, 4)}, ("layer3",))


def test_deterministic_coreset_is_reproducible_and_bounded():
    patches = np.arange(100, dtype=np.float32).reshape(20, 5)

    first = deterministic_coreset(patches, coreset_ratio=0.5, max_patches=6, seed=123)
    second = deterministic_coreset(patches, coreset_ratio=0.5, max_patches=6, seed=123)

    assert first.shape == (6, 5)
    np.testing.assert_array_equal(first, second)


def test_deterministic_coreset_rejects_invalid_inputs():
    patches = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="coreset_ratio"):
        deterministic_coreset(patches, coreset_ratio=0.0, max_patches=None, seed=1)
    with pytest.raises(ValueError, match="2D array"):
        deterministic_coreset(np.ones(4, dtype=np.float32), coreset_ratio=1.0, max_patches=None, seed=1)


def test_patchcore_fit_builds_memory_bank_from_id_images_only():
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)
    detector = PatchCoreDetector(
        PatchCoreConfig(
            layers=("layer2", "layer3"),
            coreset_ratio=0.5,
            max_train_patches=5,
            pretrained=False,
        ),
        feature_extractor=ToyFeatureExtractor(),
    )

    detector.fit(dataloader)

    assert detector.memory_bank is not None
    assert detector.memory_bank.shape == (5, 5)
    assert detector.feature_dim == 5


def test_patchcore_fit_rejects_dataset_without_id_images():
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)
    detector = PatchCoreDetector(
        PatchCoreConfig(layers=("layer2",), pretrained=False),
        feature_extractor=ToyFeatureExtractor(),
    )

    with pytest.raises(ValueError, match="at least one ID image"):
        detector.fit(dataloader)


def test_patchcore_save_load_roundtrip(tmp_path: Path):
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 0])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)
    detector = PatchCoreDetector(
        PatchCoreConfig(
            layers=("layer2", "layer3"),
            coreset_ratio=1.0,
            max_train_patches=None,
            pretrained=False,
            random_seed=9,
        ),
        feature_extractor=ToyFeatureExtractor(),
    )
    detector.fit(dataloader)
    checkpoint = tmp_path / "patchcore_memory.npz"

    detector.save(checkpoint)
    loaded = PatchCoreDetector.load(checkpoint)

    assert loaded.memory_bank is not None
    assert detector.memory_bank is not None
    np.testing.assert_array_equal(loaded.memory_bank, detector.memory_bank)
    assert loaded.config.layers == ("layer2", "layer3")
    assert loaded.config.random_seed == 9
    assert loaded.feature_dim == detector.feature_dim


def test_patchcore_save_rejects_unfitted_detector(tmp_path: Path):
    detector = PatchCoreDetector(
        PatchCoreConfig(layers=("layer2",), pretrained=False),
        feature_extractor=ToyFeatureExtractor(),
    )

    with pytest.raises(ValueError, match="before fit"):
        detector.save(tmp_path / "unfitted.npz")

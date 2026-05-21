import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from retinal_ood.models.patchcore import PatchCoreDetector


class MeanFeatureExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        means = x.mean(dim=(1, 2, 3), keepdim=True)
        return {"layer2": means.expand(x.shape[0], 1, 2, 2)}


def _load_train_patchcore_module():
    script_path = Path("scripts/train_patchcore.py")
    spec = importlib.util.spec_from_file_location("train_patchcore_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_toy_image(path: Path, value: int) -> None:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)


def test_train_patchcore_from_config_saves_memory_bank_and_metadata(tmp_path: Path):
    train_patchcore = _load_train_patchcore_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_toy_image(image_dir / "id.png", 0)
    _write_toy_image(image_dir / "ood.png", 255)
    manifest_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/id.png",
                "label": 0,
                "split": "train",
                "source": "toy",
                "ood_type": "id",
            },
            {
                "image_path": "images/ood.png",
                "label": 1,
                "split": "train",
                "source": "toy",
                "ood_type": "modality_shift",
            },
        ]
    ).to_csv(manifest_path, index=False)
    config = {
        "project": {"run_name": "toy_patchcore", "seed": 7},
        "data": {
            "root_dir": str(tmp_path),
            "train_manifest": str(manifest_path),
            "image_size": 8,
            "grayscale_to_rgb": False,
            "normalize": "minmax",
            "batch_size": 2,
        },
        "model": {
            "backbone": "resnet50",
            "layers": ["layer2"],
            "coreset_ratio": 1.0,
            "nearest_neighbors": 1,
            "max_train_patches": None,
            "pretrained": False,
            "device": "cpu",
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    checkpoint_path = train_patchcore.train_from_config(
        config,
        feature_extractor=MeanFeatureExtractor(),
    )

    assert checkpoint_path == tmp_path / "runs" / "toy_patchcore" / "patchcore_memory.npz"
    assert checkpoint_path.exists()
    run_dir = checkpoint_path.parent
    assert (run_dir / "resolved_config.json").exists()
    training_metrics = json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    assert training_metrics["train_id_images"] == 1
    assert training_metrics["memory_bank_patches"] == 4
    assert training_metrics["layers"] == ["layer2"]
    assert training_metrics["checkpoint"] == "patchcore_memory.npz"

    loaded = PatchCoreDetector.load(checkpoint_path, build_feature_extractor=False)
    assert loaded.memory_bank is not None
    assert loaded.memory_bank.shape == (4, 1)


def test_train_patchcore_requires_train_manifest():
    train_patchcore = _load_train_patchcore_module()

    with pytest.raises(ValueError, match="data.train_manifest must be set"):
        train_patchcore.train_from_config({"data": {}, "model": {"pretrained": False}})

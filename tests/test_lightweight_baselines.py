from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MeanFeatureExtractor(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = images.mean(dim=(2, 3), keepdim=True)
        return {"layer4": pooled}


def test_image_statistics_baseline_fits_id_only_and_scores_anomalies():
    from retinal_ood.baselines.image_statistics import ImageStatisticsDetector

    id_image = torch.full((1, 16, 16), 0.5)
    near_id = torch.full((1, 16, 16), 0.52)
    ood_image = torch.zeros((1, 16, 16))
    images = torch.stack([id_image, id_image + 0.02, ood_image])
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)

    detector = ImageStatisticsDetector()
    detector.fit(loader)

    assert detector.train_id_images == 2
    scores = detector.predict_scores(DataLoader(TensorDataset(torch.stack([near_id, ood_image])), batch_size=2))
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_global_feature_knn_and_mahalanobis_fit_id_only_and_score_distances():
    from retinal_ood.baselines.feature_distance import FeatureDistanceDetector

    id_a = torch.full((3, 8, 8), 0.1)
    id_b = torch.full((3, 8, 8), 0.2)
    ood = torch.full((3, 8, 8), 0.9)
    images = torch.stack([id_a, id_b, ood])
    labels = torch.tensor([0, 0, 1])
    train_loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    eval_loader = DataLoader(TensorDataset(torch.stack([id_a, ood])), batch_size=2)

    knn = FeatureDistanceDetector(
        mode="knn",
        feature_extractor=MeanFeatureExtractor(),
        layers=("layer4",),
    )
    knn.fit(train_loader)
    knn_scores = knn.predict_scores(eval_loader)

    maha = FeatureDistanceDetector(
        mode="mahalanobis",
        feature_extractor=MeanFeatureExtractor(),
        layers=("layer4",),
        covariance_regularization=1e-3,
    )
    maha.fit(train_loader)
    maha_scores = maha.predict_scores(eval_loader)

    assert knn.train_id_images == 2
    assert maha.train_id_images == 2
    assert knn_scores[1] > knn_scores[0]
    assert maha_scores[1] > maha_scores[0]


def test_train_and_evaluate_image_statistics_baseline_writes_outputs(tmp_path: Path):
    train_script = _load_script("scripts/train_baseline.py")
    eval_script = _load_script("scripts/evaluate_baseline.py")
    root = tmp_path / "data"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    for name, value in [("train_a.png", 120), ("train_b.png", 126), ("id.png", 124), ("ood.png", 10)]:
        Image.new("L", (12, 12), color=value).save(image_dir / name)

    def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    train_manifest = root / "manifests" / "train.csv"
    val_manifest = root / "manifests" / "val.csv"
    test_id_manifest = root / "manifests" / "test_id.csv"
    test_ood_manifest = root / "manifests" / "test_ood.csv"
    base = {"split": "test", "source": "toy", "ood_type": "id"}
    write_manifest(
        train_manifest,
        [
            {"image_path": "images/train_a.png", "label": 0, "split": "train", "source": "toy", "ood_type": "id"},
            {"image_path": "images/train_b.png", "label": 0, "split": "train", "source": "toy", "ood_type": "id"},
            {
                "image_path": "images/ood.png",
                "label": 1,
                "split": "train",
                "source": "toy",
                "ood_type": "sensory_artifact",
            },
        ],
    )
    write_manifest(
        val_manifest,
        [{"image_path": "images/id.png", "label": 0, "split": "val", "source": "toy", "ood_type": "id"}],
    )
    write_manifest(test_id_manifest, [{"image_path": "images/id.png", "label": 0, **base}])
    write_manifest(
        test_ood_manifest,
        [
            {
                "image_path": "images/ood.png",
                "label": 1,
                "split": "test",
                "source": "toy",
                "ood_type": "sensory_artifact",
                "ood_subtype": "dark_image",
            }
        ],
    )

    config = {
        "project": {"run_name": "image_statistics_smoke", "seed": 7},
        "data": {
            "root_dir": str(root),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_id_manifest": str(test_id_manifest),
            "test_ood_manifest": str(test_ood_manifest),
            "image_size": 12,
            "grayscale_to_rgb": False,
            "normalize": "minmax",
        },
        "model": {"name": "image_statistics"},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    checkpoint = train_script.train_from_config(config)
    out_dir = eval_script.evaluate_from_config(config, checkpoint)

    assert checkpoint.exists()
    assert (out_dir / "metrics.json").exists()
    scores = pd.read_csv(out_dir / "scores.csv")
    assert list(scores["ood_subtype"].dropna()) == ["dark_image"]
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "global" in metrics

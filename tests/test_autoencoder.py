import importlib.util
import json
from pathlib import Path

import torch
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from retinal_ood.models.autoencoder import (
    ConvAutoEncoder,
    load_autoencoder_checkpoint,
    predict_reconstruction_scores,
    reconstruction_error,
    save_autoencoder_checkpoint,
    train_autoencoder,
)


def test_autoencoder_forward_and_reconstruction_scores():
    model = ConvAutoEncoder(in_channels=1)
    images = torch.rand(2, 1, 32, 32)

    recon = model(images)
    assert recon.shape == images.shape

    errors = reconstruction_error(images, recon)
    assert errors.shape == (2,)
    assert torch.all(errors >= 0)

    dataloader = DataLoader(TensorDataset(images), batch_size=1)
    scores = predict_reconstruction_scores(model, dataloader)
    assert scores.shape == (2,)
    assert (scores >= 0).all()


def test_train_autoencoder_uses_only_id_labels():
    model = ConvAutoEncoder(in_channels=1)
    images = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    losses = train_autoencoder(model, dataloader, epochs=1, learning_rate=1e-3)

    assert len(losses) == 1
    assert losses[0] >= 0


def test_train_autoencoder_rejects_dataset_without_id_images():
    model = ConvAutoEncoder(in_channels=1)
    images = torch.rand(2, 1, 32, 32)
    labels = torch.tensor([1, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)

    with pytest.raises(ValueError, match="at least one ID image"):
        train_autoencoder(model, dataloader, epochs=1, learning_rate=1e-3)


def test_autoencoder_checkpoint_roundtrip(tmp_path: Path):
    model = ConvAutoEncoder(in_channels=1)
    checkpoint_path = tmp_path / "model.pt"

    save_autoencoder_checkpoint(
        checkpoint_path,
        model,
        config={"model": {"name": "conv_autoencoder"}},
        train_losses=[0.1],
    )
    loaded = load_autoencoder_checkpoint(checkpoint_path, in_channels=1)

    images = torch.rand(1, 1, 32, 32)
    assert loaded(images).shape == images.shape

    loaded_inferred = load_autoencoder_checkpoint(checkpoint_path)
    assert loaded_inferred(images).shape == images.shape


def test_train_autoencoder_script_smoke_test_with_toy_manifest(tmp_path: Path):
    script_path = Path("scripts/train_autoencoder.py")
    spec = importlib.util.spec_from_file_location("train_autoencoder_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    train_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_script)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for idx, label in enumerate([0, 0, 1]):
        image_path = image_dir / f"toy_{idx}.png"
        Image.new("RGB", (32, 32), color=(idx * 40, idx * 40, idx * 40)).save(image_path)
        rows.append(
            {
                "image_path": f"images/{image_path.name}",
                "label": label,
                "split": "train",
                "source": "toy",
                "ood_type": "id" if label == 0 else "semantic_outlier",
                "category": "id" if label == 0 else "semantic_outlier",
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    config = {
        "project": {"run_name": "autoencoder_smoke", "seed": 7},
        "data": {
            "train_manifest": str(manifest_path),
            "root_dir": str(tmp_path),
            "image_size": 32,
            "grayscale_to_rgb": False,
            "normalize": "minmax",
        },
        "model": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "device": "cpu",
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    checkpoint_path = train_script.train_from_config(config)

    run_dir = tmp_path / "runs" / "autoencoder_smoke"
    assert checkpoint_path == run_dir / "model.pt"
    assert checkpoint_path.exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "training_metrics.json").exists()


def test_evaluate_autoencoder_script_writes_metrics(tmp_path: Path):
    train_script_path = Path("scripts/train_autoencoder.py")
    train_spec = importlib.util.spec_from_file_location("train_autoencoder_script", train_script_path)
    assert train_spec is not None
    assert train_spec.loader is not None
    train_script = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_script)

    eval_script_path = Path("scripts/evaluate_autoencoder.py")
    eval_spec = importlib.util.spec_from_file_location("evaluate_autoencoder_script", eval_script_path)
    assert eval_spec is not None
    assert eval_spec.loader is not None
    eval_script = importlib.util.module_from_spec(eval_spec)
    eval_spec.loader.exec_module(eval_script)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in {
        "train_id.png": 40,
        "val_id.png": 60,
        "test_id.png": 80,
        "test_ood.png": 220,
    }.items():
        Image.new("RGB", (32, 32), color=(color, color, color)).save(image_dir / name)

    def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    write_manifest(
        manifest_dir / "train.csv",
        [
            {
                "image_path": "images/train_id.png",
                "label": 0,
                "split": "train",
                "source": "toy",
                "ood_type": "id",
            }
        ],
    )
    write_manifest(
        manifest_dir / "val.csv",
        [
            {
                "image_path": "images/val_id.png",
                "label": 0,
                "split": "val",
                "source": "toy",
                "ood_type": "id",
            }
        ],
    )
    write_manifest(
        manifest_dir / "test_id.csv",
        [
            {
                "image_path": "images/test_id.png",
                "label": 0,
                "split": "test",
                "source": "toy",
                "ood_type": "id",
            }
        ],
    )
    write_manifest(
        manifest_dir / "test_ood.csv",
        [
            {
                "image_path": "images/test_ood.png",
                "label": 1,
                "split": "test",
                "source": "toy",
                "ood_type": "semantic_outlier",
            }
        ],
    )
    config = {
        "project": {"run_name": "autoencoder_eval_smoke", "seed": 7},
        "data": {
            "root_dir": str(tmp_path),
            "train_manifest": str(manifest_dir / "train.csv"),
            "val_manifest": str(manifest_dir / "val.csv"),
            "test_id_manifest": str(manifest_dir / "test_id.csv"),
            "test_ood_manifest": str(manifest_dir / "test_ood.csv"),
            "image_size": 32,
            "grayscale_to_rgb": False,
            "normalize": "minmax",
            "batch_size": 1,
        },
        "model": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "device": "cpu",
            "name": "conv_autoencoder",
        },
        "evaluation": {"save_plots": False, "save_heatmaps": True},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    checkpoint = train_script.train_from_config(config)
    out_dir = eval_script.evaluate_from_config(config, checkpoint)

    assert out_dir == tmp_path / "runs" / "autoencoder_eval_smoke" / "evaluation"
    assert (out_dir / "scores.csv").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["counts"] == {"total": 2, "id": 1, "ood": 1}
    assert "semantic_outlier" in metrics["per_ood_type"]
    assert not (out_dir / "heatmaps").exists()

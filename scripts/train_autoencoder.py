#!/usr/bin/env python
"""Train the convolutional autoencoder baseline on ID FAF images only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.dataset import ManifestImageDataset
from retinal_ood.data.transforms import build_transforms
from retinal_ood.models.autoencoder import (
    ConvAutoEncoder,
    save_autoencoder_checkpoint,
    train_autoencoder,
)
from retinal_ood.utils.io import read_yaml, write_json
from retinal_ood.utils.seed import set_seed


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _infer_in_channels(dataset: ManifestImageDataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Autoencoder training requires at least one ID image with label=0")
    image, _, _ = dataset[0]
    if not torch.is_tensor(image):
        raise TypeError("Training transform must return a torch.Tensor")
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape C,H,W, got {tuple(image.shape)}")
    return int(image.shape[0])


def train_from_config(config: dict[str, Any]) -> Path:
    """Train from a YAML-like config and return the checkpoint path."""
    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    seed = int(project_cfg.get("seed", 42))
    set_seed(seed)

    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", False)),
        normalize=data_cfg.get("normalize", "minmax"),
    )
    train_manifest = data_cfg.get("train_manifest")
    if not train_manifest:
        raise ValueError("data.train_manifest must be set")

    dataset = ManifestImageDataset(
        train_manifest,
        root_dir=data_cfg.get("root_dir"),
        transform=transform,
    ).id_subset()
    if len(dataset) == 0:
        raise ValueError("Autoencoder training requires at least one ID image with label=0")

    batch_size = int(model_cfg.get("batch_size", 32))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    in_channels = int(model_cfg.get("in_channels", _infer_in_channels(dataset)))
    model = ConvAutoEncoder(in_channels=in_channels)
    device = _resolve_device(str(model_cfg.get("device", "auto")))
    train_losses = train_autoencoder(
        model,
        dataloader,
        epochs=int(model_cfg.get("epochs", 50)),
        learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
        device=device,
    )

    run_name = str(project_cfg.get("run_name", "autoencoder_baseline"))
    run_dir = Path(output_cfg.get("runs_dir", "runs")) / run_name
    checkpoint_path = run_dir / "model.pt"
    save_autoencoder_checkpoint(
        checkpoint_path,
        model,
        config=config,
        train_losses=train_losses,
    )
    write_json(run_dir / "resolved_config.json", config)
    write_json(
        run_dir / "training_metrics.json",
        {
            "model": "conv_autoencoder",
            "train_images": len(dataset),
            "epochs": len(train_losses),
            "final_loss": train_losses[-1],
            "in_channels": in_channels,
            "device": str(device),
        },
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    checkpoint_path = train_from_config(read_yaml(args.config))
    print(f"Saved autoencoder checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

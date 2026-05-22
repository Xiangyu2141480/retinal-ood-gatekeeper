"""Simple convolutional autoencoder baseline.

The autoencoder is a pixel-space unsupervised baseline: it learns to reconstruct valid
FAF-like images and uses reconstruction error as an anomaly score. It must be trained only
on ID/normal samples so OOD categories do not leak into the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


class ConvAutoEncoder(nn.Module):
    """Small autoencoder baseline for reconstruction-error anomaly scoring."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-image mean squared reconstruction error."""
    if x.shape != x_hat.shape:
        raise ValueError(f"Shape mismatch: {tuple(x.shape)} vs {tuple(x_hat.shape)}")
    return ((x - x_hat) ** 2).flatten(start_dim=1).mean(dim=1)


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


def _id_only(images: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    if labels is None:
        return images
    labels = labels.to(device=images.device)
    mask = labels == 0
    return images[mask]


def train_autoencoder(
    model: ConvAutoEncoder,
    dataloader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Train the reconstruction baseline on ID samples only.

    If a batch includes labels, only rows with label 0 are used for the loss. This keeps the
    baseline unsupervised with respect to OOD examples even when a mixed toy manifest is used
    for smoke tests.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    device = torch.device(device)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epoch_losses: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        total_images = 0
        for batch in dataloader:
            images, labels = _split_batch(batch)
            images = images.to(device=device, dtype=torch.float32)
            id_images = _id_only(images, labels)
            if id_images.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            recon = model(id_images)
            loss = loss_fn(recon, id_images)
            loss.backward()
            optimizer.step()

            batch_size = int(id_images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size

        if total_images == 0:
            raise ValueError("Autoencoder training requires at least one ID image with label=0")
        epoch_losses.append(total_loss / total_images)

    return epoch_losses


@torch.no_grad()
def predict_reconstruction_scores(
    model: ConvAutoEncoder,
    dataloader: DataLoader,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return image-level anomaly scores from reconstruction error.

    Higher values mean the image is harder for the ID-trained autoencoder to reconstruct and
    therefore more anomalous.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    scores: list[np.ndarray] = []
    for batch in dataloader:
        images, _ = _split_batch(batch)
        images = images.to(device=device, dtype=torch.float32)
        recon = model(images)
        batch_scores = reconstruction_error(images, recon).detach().cpu().numpy()
        scores.append(batch_scores)
    if not scores:
        return np.array([], dtype=float)
    return np.concatenate(scores).astype(float)


def save_autoencoder_checkpoint(
    path: str | Path,
    model: ConvAutoEncoder,
    *,
    config: dict[str, Any],
    train_losses: list[float],
) -> None:
    """Save model weights and reproducibility metadata, never raw images."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "train_losses": train_losses,
        },
        path,
    )


def load_autoencoder_checkpoint(
    path: str | Path,
    *,
    in_channels: int | None = None,
    map_location: str | torch.device = "cpu",
) -> ConvAutoEncoder:
    """Load a saved autoencoder checkpoint for scoring."""
    checkpoint = torch.load(path, map_location=map_location)
    if in_channels is None:
        first_weight = checkpoint["model_state_dict"]["encoder.0.weight"]
        in_channels = int(first_weight.shape[1])
    model = ConvAutoEncoder(in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

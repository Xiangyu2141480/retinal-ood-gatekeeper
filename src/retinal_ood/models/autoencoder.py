"""Simple convolutional autoencoder baseline skeleton."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoEncoder(nn.Module):
    """Small autoencoder baseline for reconstruction-error anomaly scoring."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
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

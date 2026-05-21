"""Intermediate CNN feature extraction.

This module is intentionally lightweight. Codex should complete/extend this implementation
when adding the PatchCore training pipeline.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetFeatureExtractor(nn.Module):
    """Extract selected intermediate feature maps from ResNet-50."""

    SUPPORTED_LAYERS = {"layer1", "layer2", "layer3", "layer4"}

    def __init__(self, layers: list[str], pretrained: bool = True) -> None:
        super().__init__()
        unknown = set(layers) - self.SUPPORTED_LAYERS
        if unknown:
            raise ValueError(f"Unsupported layers: {sorted(unknown)}")
        self.layers = layers
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        x = self.stem(x)
        x = self.layer1(x)
        if "layer1" in self.layers:
            outputs["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.layers:
            outputs["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.layers:
            outputs["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.layers:
            outputs["layer4"] = x
        return dict(outputs)

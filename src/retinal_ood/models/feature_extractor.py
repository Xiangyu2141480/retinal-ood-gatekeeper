"""Intermediate CNN feature extraction for PatchCore-style OOD detection.

PatchCore uses frozen pretrained CNN feature maps rather than disease labels. This module
exposes ResNet layer outputs as deterministic feature dictionaries so later tasks can build
patch memories from valid FAF images only.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

LayerName = Literal["layer1", "layer2", "layer3", "layer4"]
FeatureBackend = Literal["torchvision", "timm"]
SUPPORTED_LAYERS: tuple[LayerName, ...] = ("layer1", "layer2", "layer3", "layer4")
_TIMM_OUT_INDICES: dict[LayerName, int] = {
    "layer1": 1,
    "layer2": 2,
    "layer3": 3,
    "layer4": 4,
}


def _validate_layers(layers: Sequence[str]) -> tuple[LayerName, ...]:
    if not layers:
        raise ValueError("At least one feature layer must be requested")
    unknown = set(layers) - set(SUPPORTED_LAYERS)
    if unknown:
        raise ValueError(f"Unsupported layers: {sorted(unknown)}")
    ordered = tuple(layer for layer in SUPPORTED_LAYERS if layer in layers)
    return ordered


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected input tensor with shape B,C,H,W, got {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB tensor, got {x.shape[1]} channels")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


class ResNetFeatureExtractor(nn.Module):
    """Extract selected intermediate feature maps from ResNet-50."""

    def __init__(self, layers: Sequence[str], pretrained: bool = True) -> None:
        super().__init__()
        self.layers = _validate_layers(layers)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        _freeze(self)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_input(x)
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


class TimmFeatureExtractor(nn.Module):
    """Extract selected ResNet-compatible feature maps using timm's features_only API."""

    def __init__(
        self,
        backbone: str = "resnet50",
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = _validate_layers(layers)
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for backend='timm' feature extraction") from exc
        out_indices = tuple(_TIMM_OUT_INDICES[layer] for layer in self.layers)
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        _freeze(self)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_input(x)
        features = self.model(x)
        return dict(zip(self.layers, features))


def build_feature_extractor(
    backbone: str = "resnet50",
    layers: Sequence[str] = ("layer2", "layer3"),
    *,
    pretrained: bool = True,
    backend: FeatureBackend = "torchvision",
) -> nn.Module:
    """Build a frozen feature extractor for supported ResNet backbones."""
    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")
    if backend == "torchvision":
        return ResNetFeatureExtractor(layers=layers, pretrained=pretrained)
    if backend == "timm":
        return TimmFeatureExtractor(backbone=backbone, layers=layers, pretrained=pretrained)
    raise ValueError(f"Unsupported feature extractor backend: {backend}")

import pytest
import torch

from retinal_ood.models.feature_extractor import (
    ResNetFeatureExtractor,
    TimmFeatureExtractor,
    build_feature_extractor,
)


def test_torchvision_resnet_feature_shapes_on_cpu():
    model = ResNetFeatureExtractor(layers=["layer1", "layer2", "layer3", "layer4"], pretrained=False)
    x = torch.randn(2, 3, 64, 64)

    outputs = model(x)

    assert list(outputs) == ["layer1", "layer2", "layer3", "layer4"]
    assert outputs["layer1"].shape == (2, 256, 16, 16)
    assert outputs["layer2"].shape == (2, 512, 8, 8)
    assert outputs["layer3"].shape == (2, 1024, 4, 4)
    assert outputs["layer4"].shape == (2, 2048, 2, 2)
    assert all(not feature.requires_grad for feature in outputs.values())


def test_torchvision_resnet_returns_requested_layers_in_backbone_order():
    model = build_feature_extractor(
        backbone="resnet50",
        layers=["layer3", "layer1"],
        pretrained=False,
        backend="torchvision",
    )
    outputs = model(torch.randn(1, 3, 64, 64))

    assert list(outputs) == ["layer1", "layer3"]


def test_feature_extractor_is_frozen_and_in_eval_mode():
    model = ResNetFeatureExtractor(layers=["layer2"], pretrained=False)

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_timm_resnet_feature_shapes_on_cpu():
    model = TimmFeatureExtractor(backbone="resnet50", layers=["layer2", "layer3"], pretrained=False)
    x = torch.randn(1, 3, 64, 64)

    outputs = model(x)

    assert list(outputs) == ["layer2", "layer3"]
    assert outputs["layer2"].shape == (1, 512, 8, 8)
    assert outputs["layer3"].shape == (1, 1024, 4, 4)
    assert all(not feature.requires_grad for feature in outputs.values())


def test_feature_extractor_rejects_unknown_layer():
    with pytest.raises(ValueError, match="Unsupported layers"):
        ResNetFeatureExtractor(layers=["layer5"], pretrained=False)


def test_feature_extractor_rejects_invalid_input_shape():
    model = ResNetFeatureExtractor(layers=["layer1"], pretrained=False)
    with pytest.raises(ValueError, match="B,C,H,W"):
        model(torch.randn(3, 64, 64))


def test_feature_extractor_rejects_non_rgb_input():
    model = ResNetFeatureExtractor(layers=["layer1"], pretrained=False)
    with pytest.raises(ValueError, match="3-channel RGB"):
        model(torch.randn(1, 1, 64, 64))


def test_feature_extractor_rejects_unknown_backbone_or_backend():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_feature_extractor(backbone="vgg16", layers=["layer2"], pretrained=False)
    with pytest.raises(ValueError, match="Unsupported feature extractor backend"):
        build_feature_extractor(
            backbone="resnet50",
            layers=["layer2"],
            pretrained=False,
            backend="unknown",  # type: ignore[arg-type]
        )

"""Image transform builders."""

from __future__ import annotations

from typing import Literal

import torch
from torchvision import transforms

NormalizeMode = Literal["imagenet", "minmax", "none"]


def build_transforms(
    image_size: int = 224,
    grayscale_to_rgb: bool = True,
    normalize: NormalizeMode = "imagenet",
) -> transforms.Compose:
    """Build deterministic preprocessing transforms for FAF images."""
    ops: list[object] = [transforms.Resize((image_size, image_size))]
    if grayscale_to_rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())
    if normalize == "imagenet":
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalize == "minmax":
        # ToTensor already maps uint8 images to [0, 1].
        pass
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")
    return transforms.Compose(ops)  # type: ignore[arg-type]


def ensure_batched_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor has batch dimension."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(x.shape)}")

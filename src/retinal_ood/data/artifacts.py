"""Synthetic sensory artifact generation for OOD stress tests."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from retinal_ood.data.dataset import ManifestImageDataset

ArtifactType = Literal[
    "text_watermark",
    "rectangle_annotation",
    "arrow_annotation",
    "composite_layout",
    "blur_artifact",
    "border_crop",
    "gaussian_noise",
    "jpeg_compression",
]

ARTIFACT_TYPES: tuple[ArtifactType, ...] = (
    "text_watermark",
    "rectangle_annotation",
    "arrow_annotation",
    "composite_layout",
    "blur_artifact",
    "border_crop",
    "gaussian_noise",
    "jpeg_compression",
)

OUTPUT_COLUMNS = ["image_path", "label", "split", "source", "ood_type", "patient_id", "scanner", "notes"]
VALID_SPLITS = {"train", "val", "test"}
DEFAULT_SOURCE_SPLITS = ("val", "test")


def generate_artifact_dataset(
    input_manifest: str | Path,
    out_dir: str | Path,
    out_manifest: str | Path,
    *,
    root_dir: str | Path | None = None,
    artifact_types: Sequence[ArtifactType] = ARTIFACT_TYPES,
    split: str = "test",
    source_splits: Sequence[str] = DEFAULT_SOURCE_SPLITS,
    seed: int = 42,
    limit: int | None = None,
) -> pd.DataFrame:
    """Create deterministic synthetic sensory artifacts from ID rows in a manifest."""
    if split not in VALID_SPLITS:
        raise ValueError("split must be one of train, val, test")
    selected_source_splits = _validate_source_splits(source_splits)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    selected_artifacts = _validate_artifact_types(artifact_types)
    dataset = ManifestImageDataset(input_manifest, root_dir=root_dir, require_files=True).id_subset()
    dataset.df = dataset.df[dataset.df["split"].isin(selected_source_splits)].reset_index(drop=True)
    if len(dataset) == 0:
        raise ValueError(
            "input_manifest must contain at least one ID row with label=0 in held-out source_splits "
            f"{selected_source_splits}; use source_splits explicitly only for controlled smoke tests"
        )
    if limit is not None:
        dataset.df = dataset.df.head(limit).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    root_path = Path(root_dir).resolve() if root_dir is not None else None
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset.df.to_dict(orient="records")):
        source_path = dataset._resolve_path(row["image_path"])
        image = Image.open(source_path).convert("RGB")
        for artifact_idx, artifact_type in enumerate(selected_artifacts):
            corrupted = apply_artifact(image, artifact_type, rng=rng)
            filename = f"artifact_{row_idx:05d}_{artifact_idx:02d}_{artifact_type}.png"
            output_path = out_dir / filename
            corrupted.save(output_path)
            rows.append(
                {
                    "image_path": _manifest_image_path(output_path, root_path),
                    "label": 1,
                    "split": split,
                    "source": "synthetic_artifact",
                    "ood_type": "sensory_artifact",
                    "patient_id": "",
                    "scanner": row.get("scanner", ""),
                    "notes": f"{artifact_type} generated from held-out valid image",
                }
            )

    manifest_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    manifest_df.to_csv(out_manifest, index=False)
    return manifest_df


def apply_artifact(image: Image.Image, artifact_type: ArtifactType, *, rng: np.random.Generator) -> Image.Image:
    """Return a corrupted copy of ``image`` without modifying the original."""
    artifact_type = _validate_artifact_types([artifact_type])[0]
    if artifact_type == "text_watermark":
        return _text_watermark(image, rng)
    if artifact_type == "rectangle_annotation":
        return _rectangle_annotation(image, rng)
    if artifact_type == "arrow_annotation":
        return _arrow_annotation(image, rng)
    if artifact_type == "composite_layout":
        return _composite_layout(image)
    if artifact_type == "blur_artifact":
        return _blur_artifact(image, rng)
    if artifact_type == "border_crop":
        return _border_crop(image, rng)
    if artifact_type == "gaussian_noise":
        return _gaussian_noise(image, rng)
    if artifact_type == "jpeg_compression":
        return _jpeg_compression(image)
    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def _text_watermark(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    text = "INVALID FAF"
    x = int(rng.integers(4, max(5, width // 4)))
    y = int(rng.integers(4, max(5, height // 4)))
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return out


def _rectangle_annotation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    x0 = int(rng.integers(0, max(1, width // 3)))
    y0 = int(rng.integers(0, max(1, height // 3)))
    x1 = int(rng.integers(max(x0 + 1, width // 2), width))
    y1 = int(rng.integers(max(y0 + 1, height // 2), height))
    line_width = max(2, min(width, height) // 40)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=line_width)
    draw.line((x0, y0, x1, y1), fill=(255, 255, 255), width=max(1, line_width // 2))
    return out


def _arrow_annotation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    start = (int(rng.integers(0, max(1, width // 3))), int(rng.integers(0, height)))
    end = (
        int(rng.integers(max(1, width // 2), width)),
        int(rng.integers(0, height)),
    )
    line_width = max(2, min(width, height) // 35)
    draw.line((*start, *end), fill=(255, 255, 255), width=line_width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, float((dx * dx + dy * dy) ** 0.5))
    unit = (dx / length, dy / length)
    normal = (-unit[1], unit[0])
    head_len = max(6, min(width, height) // 8)
    head_w = max(4, min(width, height) // 12)
    base = (end[0] - unit[0] * head_len, end[1] - unit[1] * head_len)
    points = [
        end,
        (base[0] + normal[0] * head_w, base[1] + normal[1] * head_w),
        (base[0] - normal[0] * head_w, base[1] - normal[1] * head_w),
    ]
    draw.polygon(points, fill=(255, 0, 0))
    return out


def _composite_layout(image: Image.Image) -> Image.Image:
    width, height = image.size
    out = Image.new("RGB", (width, height), color=(255, 255, 255))
    tile_w = max(1, (width - 3) // 2)
    tile_h = max(1, (height - 3) // 2)
    resized = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
    positions = [(0, 0), (tile_w + 3, 0), (0, tile_h + 3), (tile_w + 3, tile_h + 3)]
    for idx, position in enumerate(positions):
        tile = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if idx % 2 else resized
        out.paste(tile, position)
    draw = ImageDraw.Draw(out)
    draw.line((tile_w + 1, 0, tile_w + 1, height), fill=(0, 0, 0), width=2)
    draw.line((0, tile_h + 1, width, tile_h + 1), fill=(0, 0, 0), width=2)
    return out


def _blur_artifact(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    radius = float(rng.uniform(1.2, 2.8))
    return image.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")


def _border_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    crop_frac = float(rng.uniform(0.08, 0.18))
    dx = max(1, int(width * crop_frac))
    dy = max(1, int(height * crop_frac))
    cropped = image.crop((dx, dy, width, height))
    resized = cropped.resize((max(1, width - dx), max(1, height - dy)), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, height), color=(0, 0, 0))
    out.paste(resized, (0, 0))
    return out


def _gaussian_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32)
    sigma = float(rng.uniform(18.0, 36.0))
    noisy = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _jpeg_compression(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=12, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _validate_artifact_types(artifact_types: Sequence[str]) -> list[ArtifactType]:
    if not artifact_types:
        raise ValueError("At least one artifact type is required")
    invalid = sorted(set(artifact_types) - set(ARTIFACT_TYPES))
    if invalid:
        raise ValueError(f"Unsupported artifact types: {invalid}; supported: {list(ARTIFACT_TYPES)}")
    return [artifact_type for artifact_type in artifact_types]  # type: ignore[list-item]


def _validate_source_splits(source_splits: Sequence[str]) -> list[str]:
    if not source_splits:
        raise ValueError("At least one source split is required")
    normalized = [str(split).strip() for split in source_splits]
    invalid = sorted(set(normalized) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"Invalid source_splits {invalid}; expected one of {sorted(VALID_SPLITS)}")
    return normalized


def _manifest_image_path(path: Path, root_dir: Path | None) -> str:
    resolved = path.resolve()
    if root_dir is not None:
        try:
            return resolved.relative_to(root_dir).as_posix()
        except ValueError:
            pass
    return str(path)

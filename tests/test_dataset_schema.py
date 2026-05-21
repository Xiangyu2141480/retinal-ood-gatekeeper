from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.dataset import ManifestImageDataset


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_manifest_dataset_loads_toy_image_with_relative_path(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img_path = image_dir / "img.png"
    Image.new("L", (16, 16), color=128).save(img_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_path": "images/img.png",
                "split": "train",
                "label": 0,
                "category": "id",
                "scanner": "toy_scanner",
                "notes": "synthetic test image",
            }
        ],
    )
    ds = ManifestImageDataset(manifest, root_dir=tmp_path)
    image, label, meta = ds[0]
    assert image.size == (16, 16)
    assert label == 0
    assert meta["category"] == "id"
    assert meta["patient_id"] == ""
    assert meta["resolved_image_path"] == str(img_path)


def test_manifest_dataset_does_not_require_patient_id_column(tmp_path: Path):
    img_path = tmp_path / "public_toy.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(img_path)
    manifest = tmp_path / "public_manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_path": str(img_path),
                "split": "test",
                "label": 1,
                "category": "semantic_outlier",
            }
        ],
    )
    ds = ManifestImageDataset(manifest)
    _, _, meta = ds[0]
    assert meta["patient_id"] == ""
    assert meta["scanner"] == ""
    assert meta["notes"] == ""


def test_manifest_dataset_validates_columns(tmp_path: Path):
    manifest = tmp_path / "bad.csv"
    _write_manifest(manifest, [{"image_path": "x.png", "split": "train", "label": 0}])
    with pytest.raises(ValueError, match="missing required columns"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_empty_manifest(tmp_path: Path):
    manifest = tmp_path / "empty.csv"
    pd.DataFrame(columns=["image_path", "split", "label", "category"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="contains no rows"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_invalid_split(tmp_path: Path):
    manifest = tmp_path / "bad_split.csv"
    _write_manifest(
        manifest,
        [{"image_path": "x.png", "split": "dev", "label": 0, "category": "id"}],
    )
    with pytest.raises(ValueError, match="invalid split values"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_invalid_label(tmp_path: Path):
    manifest = tmp_path / "bad_label.csv"
    _write_manifest(
        manifest,
        [{"image_path": "x.png", "split": "train", "label": 2, "category": "id"}],
    )
    with pytest.raises(ValueError, match="invalid label values"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_missing_files(tmp_path: Path):
    manifest = tmp_path / "missing_file.csv"
    _write_manifest(
        manifest,
        [{"image_path": "missing.png", "split": "train", "label": 0, "category": "id"}],
    )
    with pytest.raises(FileNotFoundError, match="missing image files"):
        ManifestImageDataset(manifest, root_dir=tmp_path)

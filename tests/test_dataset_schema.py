from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.dataset import ManifestImageDataset


def _manifest_row(image_path: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "image_path": image_path,
        "label": 0,
        "split": "train",
        "source": "synthetic_faf",
        "ood_type": "id",
    }
    row.update(overrides)
    return row


def test_manifest_dataset_loads_toy_image_without_patient_metadata(tmp_path: Path):
    img_path = tmp_path / "img.png"
    Image.new("L", (16, 16), color=128).save(img_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([_manifest_row(str(img_path))]).to_csv(manifest, index=False)
    ds = ManifestImageDataset(manifest)
    image, label, meta = ds[0]
    assert image.size == (16, 16)
    assert label == 0
    assert meta["ood_type"] == "id"


def test_manifest_dataset_validates_columns(tmp_path: Path):
    manifest = tmp_path / "bad.csv"
    pd.DataFrame([{"image_path": "x.png"}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_empty_dataset(tmp_path: Path):
    manifest = tmp_path / "empty.csv"
    pd.DataFrame(columns=["image_path", "label", "split", "source", "ood_type"]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="contains no rows"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_invalid_split(tmp_path: Path):
    manifest = tmp_path / "bad_split.csv"
    pd.DataFrame([_manifest_row("img.png", split="dev")]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="invalid splits"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_invalid_label(tmp_path: Path):
    manifest = tmp_path / "bad_label.csv"
    pd.DataFrame([_manifest_row("img.png", label=2)]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="invalid labels"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_empty_image_paths(tmp_path: Path):
    manifest = tmp_path / "empty_image_path.csv"
    pd.DataFrame([_manifest_row("")]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="empty image_path"):
        ManifestImageDataset(manifest, require_files=False)


def test_manifest_dataset_rejects_missing_files(tmp_path: Path):
    manifest = tmp_path / "missing_file.csv"
    pd.DataFrame([_manifest_row("missing.png")]).to_csv(manifest, index=False)

    with pytest.raises(FileNotFoundError, match="Missing 1 image files"):
        ManifestImageDataset(manifest, root_dir=tmp_path)

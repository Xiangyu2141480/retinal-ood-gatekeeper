from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.dataset import ManifestImageDataset


def test_manifest_dataset_loads_toy_image(tmp_path: Path):
    img_path = tmp_path / "img.png"
    Image.new("L", (16, 16), color=128).save(img_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(img_path),
                "label": 0,
                "split": "train",
                "source": "synthetic_faf",
                "ood_type": "id",
            }
        ]
    ).to_csv(manifest, index=False)
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

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from PIL import Image

from retinal_ood.data.artifacts import ARTIFACT_TYPES, apply_artifact, generate_artifact_dataset


def _write_image(path: Path, color: int) -> None:
    Image.new("RGB", (32, 24), color=(color, color, color)).save(path)


def _write_manifest(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images" / "valid_faf"
    image_dir.mkdir(parents=True)
    _write_image(image_dir / "id_0.png", 80)
    _write_image(image_dir / "id_1.png", 120)
    _write_image(image_dir / "ood_0.png", 200)
    manifest = tmp_path / "manifests" / "valid.csv"
    manifest.parent.mkdir()
    pd.DataFrame(
        [
            {
                "image_path": "images/valid_faf/id_0.png",
                "label": 0,
                "split": "test",
                "source": "toy",
                "ood_type": "id",
                "scanner": "toy_scanner",
            },
            {
                "image_path": "images/valid_faf/id_1.png",
                "label": 0,
                "split": "test",
                "source": "toy",
                "ood_type": "id",
                "scanner": "toy_scanner",
            },
            {
                "image_path": "images/valid_faf/ood_0.png",
                "label": 1,
                "split": "test",
                "source": "toy",
                "ood_type": "semantic_outlier",
            },
        ]
    ).to_csv(manifest, index=False)
    return manifest


def test_apply_artifact_keeps_size_and_does_not_modify_original():
    image = Image.new("RGB", (32, 24), color=(100, 100, 100))
    original_bytes = image.tobytes()

    for artifact_type in ARTIFACT_TYPES:
        corrupted = apply_artifact(image, artifact_type, rng=np.random.default_rng(7))
        assert corrupted.size == image.size
        assert corrupted.mode == "RGB"

    assert image.tobytes() == original_bytes


def test_generate_artifact_dataset_writes_images_and_manifest(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    out_dir = tmp_path / "images" / "ood_artifact"
    out_manifest = tmp_path / "manifests" / "artifact.csv"

    df = generate_artifact_dataset(
        manifest,
        out_dir,
        out_manifest,
        root_dir=tmp_path,
        artifact_types=["text_watermark", "gaussian_noise"],
        seed=11,
    )

    assert len(df) == 4
    assert out_manifest.exists()
    assert set(df["label"]) == {1}
    assert set(df["split"]) == {"test"}
    assert set(df["source"]) == {"synthetic_artifact"}
    assert set(df["ood_type"]) == {"sensory_artifact"}
    assert set(df["patient_id"]) == {""}
    assert all((tmp_path / path).exists() for path in df["image_path"])
    assert (tmp_path / "images" / "valid_faf" / "id_0.png").exists()


def test_generate_artifact_dataset_is_deterministic(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = tmp_path / "a.csv"
    manifest_b = tmp_path / "b.csv"

    df_a = generate_artifact_dataset(
        manifest,
        out_a,
        manifest_a,
        root_dir=tmp_path,
        artifact_types=["gaussian_noise"],
        seed=123,
        limit=1,
    )
    df_b = generate_artifact_dataset(
        manifest,
        out_b,
        manifest_b,
        root_dir=tmp_path,
        artifact_types=["gaussian_noise"],
        seed=123,
        limit=1,
    )

    bytes_a = (tmp_path / df_a.iloc[0]["image_path"]).read_bytes()
    bytes_b = (tmp_path / df_b.iloc[0]["image_path"]).read_bytes()
    assert hashlib.sha256(bytes_a).hexdigest() == hashlib.sha256(bytes_b).hexdigest()


def test_generate_artifact_dataset_rejects_invalid_artifact(tmp_path: Path):
    manifest = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="Unsupported artifact types"):
        generate_artifact_dataset(
            manifest,
            tmp_path / "out",
            tmp_path / "artifact.csv",
            root_dir=tmp_path,
            artifact_types=["unknown"],  # type: ignore[list-item]
        )


def test_generate_artifacts_script_help_loads():
    script_path = Path("scripts/generate_artifacts.py")
    spec = importlib.util.spec_from_file_location("generate_artifacts_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

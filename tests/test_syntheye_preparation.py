from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.syntheye import prepare_syntheye_dataset


def _write_syntheye_tree(root: Path, *, classes: int = 3, per_class: int = 20) -> None:
    for class_idx in range(classes):
        class_dir = root / f"CLASS{class_idx}"
        class_dir.mkdir(parents=True)
        for image_idx in range(per_class):
            value = 20 + class_idx * 20 + image_idx
            Image.new("L", (16, 16), color=value).save(class_dir / f"image{image_idx + 1}.png")


def test_prepare_syntheye_dataset_writes_stratified_manifests(tmp_path: Path):
    input_dir = tmp_path / "syntheye_onefold_10class_100perclass"
    out_dir = tmp_path / "data" / "images" / "synthetic_faf"
    manifest_dir = tmp_path / "data" / "manifests"
    _write_syntheye_tree(input_dir)

    result = prepare_syntheye_dataset(
        input_dir=input_dir,
        out_dir=out_dir,
        manifest_dir=manifest_dir,
        seed=7,
    )

    assert result.total_images == 60
    assert result.class_counts == {"CLASS0": 20, "CLASS1": 20, "CLASS2": 20}
    assert len(pd.read_csv(manifest_dir / "train_synthetic_faf.csv")) == 42
    assert len(pd.read_csv(manifest_dir / "val_synthetic_faf.csv")) == 9
    assert len(pd.read_csv(manifest_dir / "test_real_id.csv")) == 9

    for manifest_name, expected_split in [
        ("train_synthetic_faf.csv", "train"),
        ("val_synthetic_faf.csv", "val"),
        ("test_real_id.csv", "test"),
    ]:
        df = pd.read_csv(manifest_dir / manifest_name)
        assert list(df.columns) == [
            "image_path",
            "label",
            "split",
            "source",
            "ood_type",
            "patient_id",
            "scanner",
            "notes",
        ]
        assert set(df["label"]) == {0}
        assert set(df["split"]) == {expected_split}
        assert set(df["source"]) == {"synthetic_faf"}
        assert set(df["ood_type"]) == {"id"}
        assert set(df["scanner"]) == {"synthetic_heidelberg"}
        assert all(path.startswith("images/synthetic_faf/") for path in df["image_path"])
        assert not any("image1.png" in path for path in df["image_path"])
        assert all("syntheye_class=" in notes for notes in df["notes"])
        assert all((tmp_path / "data" / path).exists() for path in df["image_path"])


def test_prepare_syntheye_dataset_is_deterministic(tmp_path: Path):
    input_dir = tmp_path / "source"
    _write_syntheye_tree(input_dir, classes=2, per_class=10)

    first = prepare_syntheye_dataset(
        input_dir=input_dir,
        out_dir=tmp_path / "first" / "images" / "synthetic_faf",
        manifest_dir=tmp_path / "first" / "manifests",
        seed=42,
    )
    second = prepare_syntheye_dataset(
        input_dir=input_dir,
        out_dir=tmp_path / "second" / "images" / "synthetic_faf",
        manifest_dir=tmp_path / "second" / "manifests",
        seed=42,
    )

    assert first.total_images == second.total_images
    first_manifest = pd.read_csv(tmp_path / "first" / "manifests" / "train_synthetic_faf.csv")
    second_manifest = pd.read_csv(tmp_path / "second" / "manifests" / "train_synthetic_faf.csv")
    assert first_manifest["notes"].tolist() == second_manifest["notes"].tolist()
    assert first_manifest["image_path"].tolist() == second_manifest["image_path"].tolist()


def test_prepare_syntheye_dataset_validates_expected_counts(tmp_path: Path):
    input_dir = tmp_path / "source"
    _write_syntheye_tree(input_dir, classes=2, per_class=10)

    prepare_syntheye_dataset(
        input_dir=input_dir,
        out_dir=tmp_path / "out" / "images" / "synthetic_faf",
        manifest_dir=tmp_path / "out" / "manifests",
        expected_classes=2,
        expected_total=20,
        expected_per_class=10,
    )

    with pytest.raises(ValueError, match="Expected 10 SynthEye class folders"):
        prepare_syntheye_dataset(
            input_dir=input_dir,
            out_dir=tmp_path / "bad" / "images" / "synthetic_faf",
            manifest_dir=tmp_path / "bad" / "manifests",
            expected_classes=10,
        )


def test_prepare_syntheye_script_help_loads():
    import importlib.util

    script_path = Path("scripts/prepare_syntheye_dataset.py")
    spec = importlib.util.spec_from_file_location("prepare_syntheye_dataset_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

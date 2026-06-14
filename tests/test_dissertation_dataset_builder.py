from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.dissertation_dataset import (
    ARTIFACT_SUBTYPES,
    DissertationDatasetError,
    prepare_dissertation_dataset,
    validate_dissertation_manifest_frame,
)


def _write_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "dataset_contract_id: dissertation_faf_ood_v1",
                "seed: 42",
                "task_type: unsupervised_ood_gatekeeper",
                "target_modality: retinal_faf",
                "not_a_disease_classifier: true",
                "training_policy:",
                "  id_only: true",
                "synthetic_id_fallback_warning: true",
                "targets:",
                "  surrogate_modality_per_subtype: 2",
                "  surrogate_semantic: 3",
                "subtypes:",
                "  modality_shift:",
                "    - colour_fundus",
                "    - oct_screenshot",
                "  semantic_outlier:",
                "    - cifar10_natural",
                "  sensory_artifact:",
                "    - text_watermark",
                "    - rectangle_annotation",
                "    - arrow_annotation",
                "    - composite_layout",
                "    - blur_artifact",
                "    - border_crop",
                "    - gaussian_noise",
                "    - jpeg_compression",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_png(path: Path, *, value: int, size: tuple[int, int] = (24, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(value, (value * 3) % 255, (value * 7) % 255))
    image.save(path)


def _write_syntheye_tree(root: Path, *, classes: int = 3, per_class: int = 10) -> None:
    for class_idx in range(classes):
        class_dir = root / f"CLASS{class_idx}"
        for image_idx in range(per_class):
            value = 20 + class_idx * per_class + image_idx
            _write_png(class_dir / f"source_name_{image_idx}.png", value=value)


def _write_ood_sources(root: Path) -> tuple[Path, Path, Path]:
    colour_dir = root / "images" / "ood_modality" / "colour_fundus"
    oct_dir = root / "images" / "ood_modality" / "oct_screenshot"
    cifar_dir = root / "images" / "ood_semantic" / "natural"
    for idx in range(3):
        _write_png(colour_dir / f"colour_original_{idx}.png", value=100 + idx)
        _write_png(oct_dir / f"oct_original_{idx}.png", value=120 + idx)
    for idx, semantic_class in enumerate(["airplane", "cat", "ship", "truck"]):
        _write_png(cifar_dir / semantic_class / f"natural_original_{idx}.png", value=140 + idx)
    return colour_dir, oct_dir, cifar_dir


def test_builder_creates_safe_dissertation_package(tmp_path: Path):
    root = tmp_path / "data"
    syntheye_dir = tmp_path / "syntheye"
    _write_syntheye_tree(syntheye_dir)
    colour_dir, oct_dir, cifar_dir = _write_ood_sources(root)

    result = prepare_dissertation_dataset(
        config_path=_write_config(tmp_path / "config.yaml"),
        root_dir=root,
        prepared_syntheye_dir=syntheye_dir,
        prepared_colour_fundus_dir=colour_dir,
        prepared_oct_dir=oct_dir,
        prepared_cifar_dir=cifar_dir,
        out_image_dir=root / "images" / "dissertation_v1",
        local_manifest_dir=root / "manifests" / "generated" / "dissertation_v1",
        repo_dataset_dir=tmp_path / "datasets" / "dissertation_v1",
        audit_dir=tmp_path / "reports" / "local_audits" / "dissertation_v1",
        seed=7,
        commit_individual_lfs_images=True,
    )

    repo_manifests = result.repo_manifests
    assert set(repo_manifests) >= {
        "train_id",
        "val_id",
        "test_id_synthetic_fallback",
        "test_artifact",
        "test_ood_artifact",
        "test_modality",
        "test_ood_modality",
        "test_semantic",
        "test_ood_semantic",
        "test_ood",
        "test_ood_full",
        "test_ood_balanced_by_type",
        "test_ood_balanced_by_subtype",
        "test_ood_smoke",
    }
    assert "test_real_id" not in repo_manifests
    assert not (repo_manifests["test_id_synthetic_fallback"].parent / "test_real_id.csv").exists()

    train = pd.read_csv(repo_manifests["train_id"])
    val = pd.read_csv(repo_manifests["val_id"])
    test_id = pd.read_csv(repo_manifests["test_id_synthetic_fallback"])
    artifact = pd.read_csv(repo_manifests["test_artifact"])
    modality = pd.read_csv(repo_manifests["test_modality"])
    semantic = pd.read_csv(repo_manifests["test_semantic"])
    full_ood = pd.read_csv(repo_manifests["test_ood"])
    balanced_subtype = pd.read_csv(repo_manifests["test_ood_balanced_by_subtype"])
    smoke = pd.read_csv(repo_manifests["test_ood_smoke"])

    assert len(train) == 21
    assert len(val) == 6
    assert len(test_id) == 3
    assert set(train["label"]) == {0}
    assert set(val["ood_type"]) == {"id"}
    assert "synthetic fallback only" in " ".join(test_id["notes"])
    assert all(train["image_path"].str.startswith("images/dissertation_v1/id/train/"))
    assert all(test_id["image_path"].str.startswith("images/dissertation_v1/id/test_synthetic_fallback/"))
    assert not any("source_name_" in path for path in train["image_path"])

    assert len(artifact) == len(test_id) * len(ARTIFACT_SUBTYPES)
    assert set(artifact["label"]) == {1}
    assert set(artifact["ood_type"]) == {"sensory_artifact"}
    assert set(artifact["ood_subtype"]) == set(ARTIFACT_SUBTYPES)
    assert set(artifact["source_split"]) == {"test"}

    assert modality["ood_type"].value_counts().to_dict() == {"modality_shift": 6}
    assert modality["ood_subtype"].value_counts().to_dict() == {
        "colour_fundus": 3,
        "oct_screenshot": 3,
    }
    assert semantic["ood_subtype"].value_counts().to_dict() == {"cifar10_natural": 4}
    assert set(semantic["semantic_class"]) == {"airplane", "cat", "ship", "truck"}
    assert len(full_ood) == len(artifact) + len(modality) + len(semantic)
    assert set(balanced_subtype["ood_subtype"]) == {
        *ARTIFACT_SUBTYPES,
        "colour_fundus",
        "oct_screenshot",
        "cifar10_natural",
    }
    assert smoke["ood_subtype"].nunique() == 11
    assert smoke["ood_subtype"].value_counts().min() == 2
    assert smoke["ood_subtype"].value_counts().max() == 2

    for manifest_path in repo_manifests.values():
        df = pd.read_csv(manifest_path)
        validate_dissertation_manifest_frame(df, manifest_path=manifest_path, official_package=True)
        assert not any(Path(value).is_absolute() for value in df["image_path"])
        assert not df.astype(str).apply(lambda col: col.str.contains("Users", case=False)).any().any()
        assert "Eye_ID" not in df.columns
        assert "disease_label" not in df.columns
        assert "clinical_label" not in df.columns
        assert "biomarker_label" not in df.columns
        if "patient_id" in df.columns:
            assert df["patient_id"].fillna("").eq("").all()

    assert not any(
        path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for path in result.repo_dataset_dir.rglob("*")
    )
    audit_summary = (result.repo_dataset_dir / "AUDIT_SUMMARY.md").read_text(encoding="utf-8")
    assert "actual final dissertation dataset images" in audit_summary
    assert "data/images/dissertation_v1/" in audit_summary
    assert any("synthetic fallback" in warning for warning in result.warnings)
    assert any("pending_manual_review" in warning for warning in result.warnings)


def test_lfs_and_gitignore_rules_target_only_final_dataset_images():
    attributes = Path(".gitattributes").read_text(encoding="utf-8")
    ignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "data/images/dissertation_v1/**/*.png filter=lfs diff=lfs merge=lfs -text" in attributes
    assert "data/images/dissertation_v1/**/*.tiff filter=lfs diff=lfs merge=lfs -text" in attributes
    assert "data/images/*" in ignore
    assert "!data/images/dissertation_v1/" in ignore
    assert "!data/images/dissertation_v1/**" in ignore
    assert "data/images/**" not in attributes


def test_builder_outputs_are_deterministic(tmp_path: Path):
    root = tmp_path / "data"
    syntheye_dir = tmp_path / "syntheye"
    _write_syntheye_tree(syntheye_dir)
    colour_dir, oct_dir, cifar_dir = _write_ood_sources(root)
    config = _write_config(tmp_path / "config.yaml")

    kwargs = dict(
        config_path=config,
        root_dir=root,
        prepared_syntheye_dir=syntheye_dir,
        prepared_colour_fundus_dir=colour_dir,
        prepared_oct_dir=oct_dir,
        prepared_cifar_dir=cifar_dir,
        seed=11,
        commit_safe_manifest_package=True,
    )
    first = prepare_dissertation_dataset(
        **kwargs,
        out_image_dir=root / "images" / "first",
        local_manifest_dir=root / "manifests" / "first",
        repo_dataset_dir=tmp_path / "datasets" / "first",
        audit_dir=tmp_path / "reports" / "first",
    )
    second = prepare_dissertation_dataset(
        **kwargs,
        out_image_dir=root / "images" / "second",
        local_manifest_dir=root / "manifests" / "second",
        repo_dataset_dir=tmp_path / "datasets" / "second",
        audit_dir=tmp_path / "reports" / "second",
    )

    first_smoke = pd.read_csv(first.repo_manifests["test_ood_smoke"])
    second_smoke = pd.read_csv(second.repo_manifests["test_ood_smoke"])

    assert first_smoke["source_image_hash"].tolist() == second_smoke["source_image_hash"].tolist()
    assert first_smoke["ood_subtype"].tolist() == second_smoke["ood_subtype"].tolist()


def test_builder_rejects_duplicate_content_across_id_splits(tmp_path: Path):
    root = tmp_path / "data"
    syntheye_dir = tmp_path / "syntheye"
    for idx in range(10):
        _write_png(syntheye_dir / "CLASS0" / f"duplicate_{idx}.png", value=40)

    with pytest.raises(DissertationDatasetError, match="duplicate content across splits"):
        prepare_dissertation_dataset(
            config_path=_write_config(tmp_path / "config.yaml"),
            root_dir=root,
            prepared_syntheye_dir=syntheye_dir,
            out_image_dir=root / "images" / "dissertation_v1",
            local_manifest_dir=root / "manifests" / "generated" / "dissertation_v1",
            repo_dataset_dir=tmp_path / "datasets" / "dissertation_v1",
            audit_dir=tmp_path / "reports" / "local_audits" / "dissertation_v1",
            seed=7,
            allow_synthetic_surrogates=True,
            commit_safe_manifest_package=True,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("patient_id", "patient-001", "patient_id"),
        ("Eye_ID", "eye-001", "banned clinical/private column"),
        ("disease_label", "ABCA4", "banned clinical/private column"),
        ("clinical_label", "diagnosis", "banned clinical/private column"),
        ("biomarker_label", "lesion", "banned clinical/private column"),
        ("image_path", r"C:\\Users\\someone\\private\\image.png", "private/absolute"),
    ],
)
def test_manifest_validator_rejects_private_or_clinical_metadata(column: str, value: str, message: str):
    row = {
        "image_path": "images/dissertation_v1/id/train/id_train_000000.png",
        "label": 0,
        "split": "train",
        "source": "synthetic_faf",
        "ood_type": "id",
        "ood_subtype": "id",
        "patient_id": "",
    }
    row[column] = value
    frame = pd.DataFrame([row])

    with pytest.raises(DissertationDatasetError, match=message):
        validate_dissertation_manifest_frame(frame, manifest_path=Path("manifest.csv"), official_package=True)


def test_manifest_validator_rejects_official_ood_without_subtype():
    frame = pd.DataFrame(
        [
            {
                "image_path": "images/dissertation_v1/ood/modality_shift/colour.png",
                "label": 1,
                "split": "test",
                "source": "public_ood",
                "ood_type": "modality_shift",
                "ood_subtype": "",
                "patient_id": "",
            }
        ]
    )

    with pytest.raises(DissertationDatasetError, match="official OOD rows require non-empty ood_subtype"):
        validate_dissertation_manifest_frame(frame, manifest_path=Path("test_ood.csv"), official_package=True)


def test_builder_can_create_marked_synthetic_surrogates(tmp_path: Path):
    root = tmp_path / "data"
    syntheye_dir = tmp_path / "syntheye"
    _write_syntheye_tree(syntheye_dir, classes=2, per_class=10)

    result = prepare_dissertation_dataset(
        config_path=_write_config(tmp_path / "config.yaml"),
        root_dir=root,
        prepared_syntheye_dir=syntheye_dir,
        out_image_dir=root / "images" / "dissertation_v1",
        local_manifest_dir=root / "manifests" / "generated" / "dissertation_v1",
        repo_dataset_dir=tmp_path / "datasets" / "dissertation_v1",
        audit_dir=tmp_path / "reports" / "local_audits" / "dissertation_v1",
        seed=7,
        allow_synthetic_surrogates=True,
        commit_safe_manifest_package=True,
    )

    modality = pd.read_csv(result.repo_manifests["test_modality"])
    semantic = pd.read_csv(result.repo_manifests["test_semantic"])

    assert set(modality["source_dataset"]) == {"synthetic_surrogate"}
    assert set(semantic["source_dataset"]) == {"synthetic_surrogate"}
    assert modality["is_synthetic_surrogate"].astype(str).str.lower().eq("true").all()
    assert semantic["is_synthetic_surrogate"].astype(str).str.lower().eq("true").all()
    assert set(modality["license_status"]) == {"derived_or_generated_synthetic"}
    assert set(semantic["license_status"]) == {"derived_or_generated_synthetic"}
    assert any("synthetic surrogate" in warning for warning in result.warnings)

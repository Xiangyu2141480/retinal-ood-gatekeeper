from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.manifest_audit import ManifestAuditError, audit_manifests


REQUIRED_COLUMNS = ["image_path", "label", "split", "source", "ood_type"]


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(80, 80, 80)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(rows[0].keys())).to_csv(path, index=False)
    return path


def _id_row(image_path: str = "images/synthetic_faf/id_000.png") -> dict[str, object]:
    return {
        "image_path": image_path,
        "label": 0,
        "split": "train",
        "source": "synthetic_faf",
        "ood_type": "id",
        "patient_id": "",
        "scanner": "synthetic_heidelberg",
        "notes": "toy ID row",
    }


def _ood_row(image_path: str = "images/ood_semantic/natural_000.png") -> dict[str, object]:
    return {
        "image_path": image_path,
        "label": 1,
        "split": "test",
        "source": "public_ood",
        "ood_type": "semantic_outlier",
        "patient_id": "",
        "scanner": "",
        "notes": "toy OOD row",
    }


def test_valid_id_train_manifest(tmp_path: Path):
    _write_png(tmp_path / "data" / "images" / "synthetic_faf" / "id_000.png")
    manifest = _write_manifest(tmp_path / "train.csv", [_id_row()])

    audit = audit_manifests([manifest], root_dir=tmp_path / "data")

    assert audit["totals"]["rows"] == 1
    assert audit["totals"]["missing_files"] == 0
    assert audit["totals"]["duplicate_image_paths"] == 0
    assert not audit["warnings"]


def test_valid_ood_test_manifest_with_subtype_counts(tmp_path: Path):
    _write_png(tmp_path / "data" / "images" / "ood_modality" / "colour_fundus.png")
    row = _ood_row("images/ood_modality/colour_fundus.png")
    row["ood_type"] = "modality_shift"
    row["ood_subtype"] = "colour_fundus"
    manifest = _write_manifest(tmp_path / "test_ood.csv", [row])

    audit = audit_manifests([manifest], root_dir=tmp_path / "data")

    assert audit["manifests"][0]["counts"]["ood_type"] == {"modality_shift": 1}
    assert audit["manifests"][0]["counts"]["ood_subtype"] == {"colour_fundus": 1}


def test_missing_required_column_fails(tmp_path: Path):
    row = _id_row()
    row.pop("ood_type")
    manifest = _write_manifest(tmp_path / "bad.csv", [row])

    with pytest.raises(ManifestAuditError, match="missing required columns"):
        audit_manifests([manifest], root_dir=tmp_path)


def test_train_row_with_label_one_fails(tmp_path: Path):
    row = _ood_row()
    row["split"] = "train"
    manifest = _write_manifest(tmp_path / "bad.csv", [row])

    with pytest.raises(ManifestAuditError, match="train rows must all be label=0"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)


def test_label_zero_with_non_id_ood_type_fails(tmp_path: Path):
    row = _id_row()
    row["ood_type"] = "semantic_outlier"
    manifest = _write_manifest(tmp_path / "bad.csv", [row])

    with pytest.raises(ManifestAuditError, match="label=0 rows must have ood_type=id"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)


def test_label_one_with_id_ood_type_fails(tmp_path: Path):
    row = _ood_row()
    row["ood_type"] = "id"
    manifest = _write_manifest(tmp_path / "bad.csv", [row])

    with pytest.raises(ManifestAuditError, match="label=1 rows must not have ood_type=id"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)


def test_invalid_ood_type_fails(tmp_path: Path):
    row = _ood_row()
    row["ood_type"] = "scanner_shift"
    manifest = _write_manifest(tmp_path / "bad.csv", [row])

    with pytest.raises(ManifestAuditError, match="invalid ood_type"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)


def test_missing_image_file_fails_unless_no_check_files(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "train.csv", [_id_row()])

    with pytest.raises(ManifestAuditError, match="missing image files"):
        audit_manifests([manifest], root_dir=tmp_path / "data")

    audit = audit_manifests([manifest], root_dir=tmp_path / "data", check_files=False)
    assert audit["totals"]["missing_files"] == 0


def test_absolute_private_path_fails(tmp_path: Path):
    private_path = "C:" + "\\Users" + "\\alice\\private.png"
    manifest = _write_manifest(tmp_path / "bad.csv", [_id_row(private_path)])

    with pytest.raises(ManifestAuditError, match="private local paths"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)


def test_sensory_artifact_from_train_source_split_fails(tmp_path: Path):
    row = _ood_row("images/ood_artifact/artifact_000.png")
    row["ood_type"] = "sensory_artifact"
    row["source_split"] = "train"
    manifest = _write_manifest(tmp_path / "test_artifact.csv", [row])

    with pytest.raises(ManifestAuditError, match="source_split=train"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False)

    audit = audit_manifests(
        [manifest],
        root_dir=tmp_path,
        check_files=False,
        allow_train_artifact=True,
    )
    assert audit["totals"]["rows"] == 1


def test_synthetic_only_manifest_named_test_real_id_warns(tmp_path: Path):
    row = _id_row("images/synthetic_faf/test_000.png")
    row["split"] = "test"
    manifest = _write_manifest(tmp_path / "test_real_id.csv", [row])

    audit = audit_manifests([manifest], root_dir=tmp_path, check_files=False)

    assert any("synthetic fallback" in warning for warning in audit["warnings"])


def test_duplicate_across_splits_warns_or_fails(tmp_path: Path):
    train = _id_row("images/synthetic_faf/shared.png")
    val = _id_row("images/synthetic_faf/shared.png")
    val["split"] = "val"
    manifest = _write_manifest(tmp_path / "combined.csv", [train, val])

    audit = audit_manifests([manifest], root_dir=tmp_path, check_files=False)
    assert audit["totals"]["duplicate_image_paths"] == 1
    assert any("duplicate image_path" in warning for warning in audit["warnings"])

    with pytest.raises(ManifestAuditError, match="duplicate image_path"):
        audit_manifests([manifest], root_dir=tmp_path, check_files=False, fail_on_duplicates=True)


def test_write_json_audit_summary(tmp_path: Path):
    _write_png(tmp_path / "data" / "images" / "synthetic_faf" / "id_000.png")
    manifest = _write_manifest(tmp_path / "train.csv", [_id_row()])
    out = tmp_path / "audit.json"

    audit = audit_manifests([manifest], root_dir=tmp_path / "data", write_json=out)

    saved = json.loads(out.read_text())
    assert saved["totals"]["rows"] == audit["totals"]["rows"]

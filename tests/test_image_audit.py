from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.image_audit import ImageAuditError, audit_dataset_images


def _load_script_module():
    script_path = Path("scripts/audit_dataset_images.py")
    spec = importlib.util.spec_from_file_location("audit_dataset_images", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_png(path: Path, *, color: tuple[int, int, int] = (90, 90, 90), size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(
    image_path: str,
    *,
    split: str,
    label: int = 0,
    source: str = "synthetic_faf",
    ood_type: str = "id",
    ood_subtype: str = "",
    patient_id: str = "",
) -> dict[str, object]:
    return {
        "image_path": image_path,
        "label": label,
        "split": split,
        "source": source,
        "ood_type": ood_type,
        "ood_subtype": ood_subtype,
        "patient_id": patient_id,
        "scanner": "",
        "notes": "",
    }


def test_valid_images_pass_and_contact_sheets_are_written(tmp_path: Path):
    root = tmp_path / "data"
    _write_png(root / "images" / "id" / "train.png", color=(80, 80, 80))
    _write_png(root / "images" / "ood" / "colour.png", color=(90, 30, 30))
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _row("images/id/train.png", split="train"),
            _row(
                "images/ood/colour.png",
                split="test",
                label=1,
                source="public_colour",
                ood_type="modality_shift",
                ood_subtype="colour_fundus",
            ),
        ],
    )

    audit = audit_dataset_images(
        [manifest],
        root_dir=root,
        out_dir=tmp_path / "reports" / "local_audits",
        make_contact_sheets=True,
        sample_per_group=2,
    )

    assert audit["totals"]["images"] == 2
    assert audit["totals"]["opened_ok"] == 2
    assert audit["totals"]["corrupt_images"] == 0
    assert audit["totals"]["private_paths"] == 0
    assert audit["totals"]["non_empty_patient_id"] == 0
    assert audit["extension_distribution"] == {".png": 2}
    assert audit["counts"]["ood_type"]["id"] == 1
    assert audit["counts"]["ood_type"]["modality_shift"] == 1
    contact_sheets = audit["contact_sheets"]
    assert contact_sheets
    assert all((tmp_path / "reports" / "local_audits" / path).exists() for path in contact_sheets)


def test_corrupt_image_fails_when_requested(tmp_path: Path):
    root = tmp_path / "data"
    bad = root / "images" / "bad.png"
    bad.parent.mkdir(parents=True)
    bad.write_text("not an image", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "manifest.csv", [_row("images/bad.png", split="test")])

    with pytest.raises(ImageAuditError, match="corrupt"):
        audit_dataset_images([manifest], root_dir=root, fail_on_corrupt=True)


def test_missing_image_fails(tmp_path: Path):
    root = tmp_path / "data"
    manifest = _write_manifest(tmp_path / "manifest.csv", [_row("images/missing.png", split="test")])

    with pytest.raises(ImageAuditError, match="missing"):
        audit_dataset_images([manifest], root_dir=root)


def test_blank_image_and_missing_subtype_are_flagged(tmp_path: Path):
    root = tmp_path / "data"
    _write_png(root / "images" / "blank.png", color=(0, 0, 0))
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _row(
                "images/blank.png",
                split="test",
                label=1,
                source="artifact_generator",
                ood_type="sensory_artifact",
                ood_subtype="",
            )
        ],
    )

    audit = audit_dataset_images([manifest], root_dir=root)

    assert audit["totals"]["blank_or_near_blank_images"] == 1
    assert audit["totals"]["all_black_images"] == 1
    assert any("sensory_artifact" in warning for warning in audit["warnings"])


def test_duplicate_sha_across_train_and_test_can_fail(tmp_path: Path):
    root = tmp_path / "data"
    _write_png(root / "images" / "train.png", color=(70, 70, 70))
    _write_png(root / "images" / "test.png", color=(70, 70, 70))
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _row("images/train.png", split="train"),
            _row("images/test.png", split="test"),
        ],
    )

    with pytest.raises(ImageAuditError, match="duplicate content across splits"):
        audit_dataset_images(
            [manifest],
            root_dir=root,
            fail_on_duplicate_content_across_splits=True,
        )


def test_non_empty_patient_id_fails(tmp_path: Path):
    root = tmp_path / "data"
    _write_png(root / "images" / "id.png")
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_row("images/id.png", split="test", patient_id="private-001")],
    )

    with pytest.raises(ImageAuditError, match="patient_id"):
        audit_dataset_images([manifest], root_dir=root)


def test_script_entrypoint_exposes_audit_function(tmp_path: Path):
    module = _load_script_module()
    root = tmp_path / "data"
    _write_png(root / "images" / "id.png")
    manifest = _write_manifest(tmp_path / "manifest.csv", [_row("images/id.png", split="test")])

    audit = module.audit_dataset_images([manifest], root_dir=root)

    assert audit["totals"]["images"] == 1

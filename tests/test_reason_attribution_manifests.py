from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REASON_TYPES = ("modality_shift", "sensory_artifact", "semantic_outlier")
SUBTYPES = {
    "modality_shift": ("colour_fundus", "oct_screenshot"),
    "sensory_artifact": (
        "text_watermark",
        "rectangle_annotation",
        "arrow_annotation",
        "composite_layout",
        "blur_artifact",
        "border_crop",
        "gaussian_noise",
        "jpeg_compression",
    ),
    "semantic_outlier": ("cifar10_natural",),
}


def _load_manifest_script():
    path = Path("scripts/build_reason_attribution_manifests.py")
    spec = importlib.util.spec_from_file_location("build_reason_attribution_manifests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reason_rows(repeats_per_subtype: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for ood_type, subtypes in SUBTYPES.items():
        for subtype in subtypes:
            for repeat in range(repeats_per_subtype):
                rows.append(
                    {
                        "image_path": f"images/dissertation_v1/ood/{ood_type}/{subtype}/{index:04d}.png",
                        "label": 1,
                        "split": "test",
                        "source": "synthetic_artifact" if ood_type == "sensory_artifact" else "public_ood",
                        "ood_type": ood_type,
                        "ood_subtype": subtype,
                        "source_dataset": f"{ood_type}_source",
                        "source_url": "",
                        "license_status": "research_use",
                        "source_split": "test",
                        "source_image_hash": f"source_hash_{index}",
                        "parent_image_hash": f"parent_hash_{index}",
                        "synthetic_transform": subtype if ood_type == "sensory_artifact" else "",
                        "severity": repeat % 3,
                        "notes": "safe public/synthetic OOD metadata",
                        "patient_id": "",
                        "Eye_ID": "",
                        "clinical_label": "",
                        "disease_label": "",
                        "biomarker_label": "",
                    }
                )
                index += 1
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reason_split_is_deterministic_and_stratified_by_subtype(tmp_path: Path):
    module = _load_manifest_script()
    input_manifest = _write_manifest(tmp_path / "test_ood_full.csv", _reason_rows())

    first = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "first",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_first",
    )
    second = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "second",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_second",
    )

    for split_name in ("train", "val", "test"):
        first_df = pd.read_csv(first.manifest_paths[split_name])
        second_df = pd.read_csv(second.manifest_paths[split_name])
        pd.testing.assert_frame_equal(first_df, second_df)
        assert set(first_df["ood_subtype"]) == {
            subtype for subtypes in SUBTYPES.values() for subtype in subtypes
        }

    assert len(pd.read_csv(first.manifest_paths["train"])) == 33
    assert len(pd.read_csv(first.manifest_paths["val"])) == 11
    assert len(pd.read_csv(first.manifest_paths["test"])) == 11


def test_reason_manifests_are_ood_only_and_drop_forbidden_identifier_columns(tmp_path: Path):
    module = _load_manifest_script()
    input_manifest = _write_manifest(tmp_path / "test_ood_full.csv", _reason_rows())

    result = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "manifests",
        reports_dir=tmp_path / "reports",
    )

    forbidden = {"patient_id", "Eye_ID", "clinical_label", "disease_label", "biomarker_label"}
    for path in result.manifest_paths.values():
        frame = pd.read_csv(path)
        assert set(frame["label"]) == {1}
        assert set(frame["ood_type"]).issubset(set(REASON_TYPES))
        assert frame["ood_subtype"].astype(str).str.strip().ne("").all()
        assert forbidden.isdisjoint(frame.columns)

    audit = result.audit_markdown.read_text(encoding="utf-8")
    assert "Stage 2 explanation only" in audit
    assert "Stage 1 remains ID-only" in audit


def test_reason_manifest_builder_rejects_id_rows_private_paths_and_identifiers(tmp_path: Path):
    module = _load_manifest_script()

    id_rows = _reason_rows()
    id_rows[0]["label"] = 0
    id_rows[0]["ood_type"] = "id"
    with pytest.raises(ValueError, match="OOD-only"):
        module.build_reason_manifests(
            input_manifest=_write_manifest(tmp_path / "id_row.csv", id_rows),
            out_dir=tmp_path / "id_out",
            reports_dir=tmp_path / "id_reports",
        )

    private_rows = _reason_rows()
    private_rows[0]["image_path"] = "C:" + "\\Users\\someone\\private\\image.png"
    with pytest.raises(ValueError, match="private path"):
        module.build_reason_manifests(
            input_manifest=_write_manifest(tmp_path / "private.csv", private_rows),
            out_dir=tmp_path / "private_out",
            reports_dir=tmp_path / "private_reports",
        )

    patient_rows = _reason_rows()
    patient_rows[0]["patient_id"] = "patient-001"
    with pytest.raises(ValueError, match="patient_id"):
        module.build_reason_manifests(
            input_manifest=_write_manifest(tmp_path / "patient.csv", patient_rows),
            out_dir=tmp_path / "patient_out",
            reports_dir=tmp_path / "patient_reports",
        )


def test_reason_manifest_builder_writes_audit_summary_csv_and_markdown(tmp_path: Path):
    module = _load_manifest_script()
    input_manifest = _write_manifest(tmp_path / "test_ood_full.csv", _reason_rows())

    result = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "manifests",
        reports_dir=tmp_path / "reports",
    )

    audit_csv = pd.read_csv(result.audit_csv)
    assert {"section", "label", "count"}.issubset(audit_csv.columns)
    assert {"split", "ood_type", "ood_subtype", "safety"}.issubset(set(audit_csv["section"]))
    assert result.audit_markdown.exists()

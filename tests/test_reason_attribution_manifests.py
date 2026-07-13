from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
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


def _load_grouped_split_module():
    path = Path("src/retinal_ood/reason_attribution/grouped_split.py")
    spec = importlib.util.spec_from_file_location("reason_attribution_grouped_split", path)
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


def _real_reason_manifest_path() -> Path:
    return Path("datasets/dissertation_v1/manifests/test_ood_full.csv")


def _expected_grouped_subtype_counts() -> dict[str, dict[str, int]]:
    return {
        "train": {
            "arrow_annotation": 90,
            "blur_artifact": 90,
            "border_crop": 90,
            "cifar10_natural": 300,
            "colour_fundus": 120,
            "composite_layout": 90,
            "gaussian_noise": 90,
            "jpeg_compression": 90,
            "oct_screenshot": 120,
            "rectangle_annotation": 90,
            "text_watermark": 90,
        },
        "val": {
            "arrow_annotation": 30,
            "blur_artifact": 30,
            "border_crop": 30,
            "cifar10_natural": 100,
            "colour_fundus": 40,
            "composite_layout": 30,
            "gaussian_noise": 30,
            "jpeg_compression": 30,
            "oct_screenshot": 40,
            "rectangle_annotation": 30,
            "text_watermark": 30,
        },
        "test": {
            "arrow_annotation": 30,
            "blur_artifact": 30,
            "border_crop": 30,
            "cifar10_natural": 100,
            "colour_fundus": 40,
            "composite_layout": 30,
            "gaussian_noise": 30,
            "jpeg_compression": 30,
            "oct_screenshot": 40,
            "rectangle_annotation": 30,
            "text_watermark": 30,
        },
    }


@pytest.mark.parametrize(
    ("parent_image_hash", "image_path", "expected"),
    [
        ("  parent-hash  ", "images/fallback.png", "parent-hash"),
        ("", "  images/fallback.png  ", "images/fallback.png"),
        ("   ", "images/fallback.png", "images/fallback.png"),
        (None, "images/fallback.png", "images/fallback.png"),
        (np.nan, "images/fallback.png", "images/fallback.png"),
        ("nan", "images/fallback.png", "images/fallback.png"),
        (" NaN ", "images/fallback.png", "images/fallback.png"),
    ],
)
def test_normalize_group_id_uses_parent_hash_or_image_path_fallback(
    parent_image_hash: object,
    image_path: str,
    expected: str,
):
    grouped_split = _load_grouped_split_module()
    assert grouped_split.normalize_group_id(parent_image_hash, image_path) == expected


def test_parent_grouped_split_matches_expected_counts_and_invariants():
    grouped_split = _load_grouped_split_module()
    frame = pd.read_csv(_real_reason_manifest_path())

    split_frames = grouped_split.parent_grouped_stratified_split(
        frame,
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert set(split_frames) == {"train", "val", "test"}
    expected_sizes = {"train": 1260, "val": 420, "test": 420}
    expected_subtype_counts = _expected_grouped_subtype_counts()
    normalized_groups_by_split: dict[str, set[str]] = {}
    image_paths_by_split: dict[str, set[str]] = {}

    for split_name, expected_size in expected_sizes.items():
        split_frame = split_frames[split_name]
        assert len(split_frame) == expected_size
        assert set(split_frame["split"]) == {split_name}
        assert {"group_id", "_group_id", "__group_id"}.isdisjoint(split_frame.columns)
        assert split_frame["ood_subtype"].value_counts().sort_index().to_dict() == expected_subtype_counts[split_name]
        normalized_groups = {
            grouped_split.normalize_group_id(parent_hash, image_path)
            for parent_hash, image_path in zip(split_frame["parent_image_hash"], split_frame["image_path"])
        }
        normalized_groups_by_split[split_name] = normalized_groups
        image_paths_by_split[split_name] = set(split_frame["image_path"])

    for left_name, right_name in itertools.combinations(("train", "val", "test"), 2):
        assert image_paths_by_split[left_name].isdisjoint(image_paths_by_split[right_name])
        assert normalized_groups_by_split[left_name].isdisjoint(normalized_groups_by_split[right_name])

    combined_frame = pd.concat(split_frames.values(), ignore_index=True)
    combined_frame["_normalized_group_id"] = [
        grouped_split.normalize_group_id(parent_hash, image_path)
        for parent_hash, image_path in zip(combined_frame["parent_image_hash"], combined_frame["image_path"])
    ]
    sensory = combined_frame[combined_frame["ood_type"].eq("sensory_artifact")]
    assert sensory.groupby("_normalized_group_id")["split"].nunique().eq(1).all()
    assert sensory.groupby("_normalized_group_id")["ood_subtype"].nunique().eq(8).all()


def test_grouped_reason_manifests_are_invariant_to_input_row_order(tmp_path: Path):
    module = _load_manifest_script()
    source = pd.read_csv(_real_reason_manifest_path())
    shuffled = source.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    original_manifest = tmp_path / "original.csv"
    shuffled_manifest = tmp_path / "shuffled.csv"
    source.to_csv(original_manifest, index=False)
    shuffled.to_csv(shuffled_manifest, index=False)

    first = module.build_reason_manifests(
        input_manifest=original_manifest,
        out_dir=tmp_path / "first",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_first",
        split_mode="grouped",
        output_prefix="reason_grouped",
    )
    second = module.build_reason_manifests(
        input_manifest=shuffled_manifest,
        out_dir=tmp_path / "second",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_second",
        split_mode="grouped",
        output_prefix="reason_grouped",
    )

    for split_name in ("train", "val", "test"):
        assert first.manifest_paths[split_name].name == f"reason_grouped_{split_name}.csv"
        first_df = pd.read_csv(first.manifest_paths[split_name])
        second_df = pd.read_csv(second.manifest_paths[split_name])
        pd.testing.assert_frame_equal(first_df, second_df)


def test_grouped_reason_manifests_are_reproducible_and_do_not_write_group_ids(tmp_path: Path):
    module = _load_manifest_script()
    input_manifest = _real_reason_manifest_path()

    first = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "first",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_first",
        split_mode="grouped",
        output_prefix="reason_grouped",
    )
    second = module.build_reason_manifests(
        input_manifest=input_manifest,
        out_dir=tmp_path / "second",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        reports_dir=tmp_path / "reports_second",
        split_mode="grouped",
        output_prefix="reason_grouped",
    )

    for split_name in ("train", "val", "test"):
        assert first.manifest_paths[split_name].read_bytes() == second.manifest_paths[split_name].read_bytes()
        frame = pd.read_csv(first.manifest_paths[split_name])
        assert {"group_id", "_group_id", "__group_id"}.isdisjoint(frame.columns)


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

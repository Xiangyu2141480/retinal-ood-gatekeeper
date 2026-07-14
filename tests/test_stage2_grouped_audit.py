from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


EXPECTED_SPLIT_SUMMARY = pd.DataFrame(
    [
        {
            "split": "train",
            "total_rows": 1260,
            "unique_image_paths": 1260,
            "unique_groups": 630,
            "modality_shift": 240,
            "sensory_artifact": 720,
            "semantic_outlier": 300,
        },
        {
            "split": "validation",
            "total_rows": 420,
            "unique_image_paths": 420,
            "unique_groups": 210,
            "modality_shift": 80,
            "sensory_artifact": 240,
            "semantic_outlier": 100,
        },
        {
            "split": "test",
            "total_rows": 420,
            "unique_image_paths": 420,
            "unique_groups": 210,
            "modality_shift": 80,
            "sensory_artifact": 240,
            "semantic_outlier": 100,
        },
    ]
)

EXPECTED_OVERLAP_SUMMARY = pd.DataFrame(
    [
        {"overlap_scope": "train-val", "image_path_overlap_count": 0, "group_id_overlap_count": 0},
        {"overlap_scope": "train-test", "image_path_overlap_count": 0, "group_id_overlap_count": 0},
        {"overlap_scope": "val-test", "image_path_overlap_count": 0, "group_id_overlap_count": 0},
        {"overlap_scope": "all-three", "image_path_overlap_count": 0, "group_id_overlap_count": 0},
    ]
)


def _load_manifest_script():
    path = Path("scripts/build_reason_attribution_manifests.py")
    spec = importlib.util.spec_from_file_location("build_reason_attribution_manifests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_audit_script():
    path = Path("scripts/audit_stage2_grouped.py")
    spec = importlib.util.spec_from_file_location("audit_stage2_grouped", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_reason_manifest_path() -> Path:
    return Path("datasets/dissertation_v1/manifests/test_ood_full.csv")


def _expected_subtype_distribution() -> pd.DataFrame:
    rows = [
        ("modality_shift", "colour_fundus", 120, 40, 40, 200),
        ("modality_shift", "oct_screenshot", 120, 40, 40, 200),
        ("sensory_artifact", "text_watermark", 90, 30, 30, 150),
        ("sensory_artifact", "rectangle_annotation", 90, 30, 30, 150),
        ("sensory_artifact", "arrow_annotation", 90, 30, 30, 150),
        ("sensory_artifact", "composite_layout", 90, 30, 30, 150),
        ("sensory_artifact", "blur_artifact", 90, 30, 30, 150),
        ("sensory_artifact", "border_crop", 90, 30, 30, 150),
        ("sensory_artifact", "gaussian_noise", 90, 30, 30, 150),
        ("sensory_artifact", "jpeg_compression", 90, 30, 30, 150),
        ("semantic_outlier", "cifar10_natural", 300, 100, 100, 500),
    ]
    return pd.DataFrame(
        rows,
        columns=["family", "subtype", "train", "validation", "test", "total"],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _shared_group_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(
            {
                "image_path": f"images/dissertation_v1/ood/modality_shift/colour_fundus/{index:04d}.png",
                "label": 1,
                "split": "test",
                "source": "public_ood",
                "ood_type": "modality_shift",
                "ood_subtype": "colour_fundus",
                "source_dataset": "modality_shift_source",
                "source_url": "",
                "license_status": "research_use",
                "source_split": "test",
                "source_image_hash": f"source_hash_{index}",
                "parent_image_hash": "shared-parent",
                "synthetic_transform": "",
                "severity": 0,
                "notes": "safe public OOD metadata",
            }
        )
    return rows


def test_audit_stage2_grouped_cli_help_lists_required_arguments():
    result = subprocess.run(
        [sys.executable, "scripts/audit_stage2_grouped.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--input", "--train", "--val", "--test", "--out-dir", "--seed"):
        assert flag in result.stdout


def test_audit_grouped_manifests_writes_required_reports_and_hashes(tmp_path: Path):
    builder = _load_manifest_script()
    audit = _load_audit_script()
    build_result = builder.build_reason_manifests(
        input_manifest=_real_reason_manifest_path(),
        out_dir=tmp_path / "manifests",
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        split_mode="grouped",
        output_prefix="reason_grouped",
        reports_dir=tmp_path / "builder_reports",
    )

    audit_result = audit.audit_grouped_manifests(
        input_manifest=_real_reason_manifest_path(),
        train_manifest=build_result.manifest_paths["train"],
        val_manifest=build_result.manifest_paths["val"],
        test_manifest=build_result.manifest_paths["test"],
        out_dir=tmp_path / "audit_reports",
        seed=42,
    )

    split_summary = pd.read_csv(audit_result.split_summary_path)
    overlap_summary = pd.read_csv(audit_result.group_overlap_summary_path)
    subtype_distribution = pd.read_csv(audit_result.subtype_distribution_path)

    pd.testing.assert_frame_equal(split_summary, EXPECTED_SPLIT_SUMMARY)
    pd.testing.assert_frame_equal(overlap_summary, EXPECTED_OVERLAP_SUMMARY)
    pd.testing.assert_frame_equal(subtype_distribution, _expected_subtype_distribution())

    expected_input_sha = _sha256(_real_reason_manifest_path())
    expected_manifest_sha = {
        split_name: _sha256(path) for split_name, path in build_result.manifest_paths.items()
    }
    assert audit_result.input_sha256 == expected_input_sha
    assert audit_result.manifest_sha256 == expected_manifest_sha

    audit_markdown = audit_result.split_audit_path.read_text(encoding="utf-8")
    assert "seed: 42" in audit_markdown
    assert "algorithm: parent_grouped_stratified_split" in audit_markdown
    assert "ratios: train=0.6, validation=0.2, test=0.2" in audit_markdown
    assert f"input manifest sha256: {expected_input_sha}" in audit_markdown
    assert f"train manifest sha256: {expected_manifest_sha['train']}" in audit_markdown
    assert f"val manifest sha256: {expected_manifest_sha['val']}" in audit_markdown
    assert f"test manifest sha256: {expected_manifest_sha['test']}" in audit_markdown
    assert "shared image paths across splits: 0" in audit_markdown
    assert "shared group IDs across splits: 0" in audit_markdown


def test_audit_grouped_manifests_rejects_cross_split_group_overlap(tmp_path: Path):
    audit = _load_audit_script()
    source_rows = _shared_group_rows()
    source_manifest = _write_manifest(tmp_path / "source.csv", source_rows)
    train_manifest = _write_manifest(tmp_path / "train.csv", [{**source_rows[0], "split": "train"}])
    val_manifest = _write_manifest(tmp_path / "val.csv", [{**source_rows[1], "split": "val"}])
    test_manifest = _write_manifest(tmp_path / "test.csv", [{**source_rows[2], "split": "test"}])

    with pytest.raises(ValueError, match="all-three.*group_id overlap=1"):
        audit.audit_grouped_manifests(
            input_manifest=source_manifest,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            out_dir=tmp_path / "audit_reports",
            seed=42,
            expected_split_sizes={"train": 1, "val": 1, "test": 1},
            expected_subtype_counts={
                "train": {"colour_fundus": 1},
                "val": {"colour_fundus": 1},
                "test": {"colour_fundus": 1},
            },
        )

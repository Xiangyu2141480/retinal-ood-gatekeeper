#!/usr/bin/env python
"""Audit parent-grouped Stage 2 manifests and write deterministic reports."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.reason_attribution.classifier import FAMILY_CLASSES, SUBTYPE_CLASSES
from retinal_ood.reason_attribution.grouped_split import normalize_group_id

SPLIT_ORDER = ("train", "val", "test")
SPLIT_LABELS = {"train": "train", "val": "validation", "test": "test"}
DEFAULT_SPLIT_SIZES = {"train": 1260, "val": 420, "test": 420}
DEFAULT_SUBTYPE_COUNTS = {
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


class GroupedAuditResult(NamedTuple):
    split_summary_path: Path
    group_overlap_summary_path: Path
    subtype_distribution_path: Path
    split_audit_path: Path
    input_sha256: str
    manifest_sha256: dict[str, str]


def audit_grouped_manifests(
    *,
    input_manifest: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    out_dir: str | Path,
    seed: int,
    expected_split_sizes: dict[str, int] | None = None,
    expected_subtype_counts: dict[str, dict[str, int]] | None = None,
) -> GroupedAuditResult:
    """Validate grouped manifests and write deterministic audit artifacts."""
    input_path = Path(input_manifest)
    manifest_paths = {
        "train": Path(train_manifest),
        "val": Path(val_manifest),
        "test": Path(test_manifest),
    }
    for path in [input_path, *manifest_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Audit input does not exist: {path}")

    expected_sizes = DEFAULT_SPLIT_SIZES if expected_split_sizes is None else dict(expected_split_sizes)
    expected_subtypes = DEFAULT_SUBTYPE_COUNTS if expected_subtype_counts is None else {
        split_name: dict(counts) for split_name, counts in expected_subtype_counts.items()
    }
    _validate_expected_keys(expected_sizes, expected_subtypes)

    source = pd.read_csv(input_path)
    _validate_source_manifest(source)
    split_frames = {
        split_name: _load_and_validate_split_manifest(
            manifest_path,
            split_name=split_name,
            source=source,
        )
        for split_name, manifest_path in manifest_paths.items()
    }

    overlap_summary = _build_overlap_summary(split_frames)
    _fail_on_overlap(overlap_summary)
    _validate_source_coverage(source, split_frames)
    _validate_expected_counts(split_frames, expected_sizes, expected_subtypes)

    split_summary = _build_split_summary(split_frames)
    subtype_distribution = _build_subtype_distribution(source, split_frames)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary_path = output_dir / "split_summary.csv"
    group_overlap_summary_path = output_dir / "group_overlap_summary.csv"
    subtype_distribution_path = output_dir / "subtype_distribution.csv"
    split_audit_path = output_dir / "split_audit.md"
    split_summary.to_csv(split_summary_path, index=False)
    overlap_summary.to_csv(group_overlap_summary_path, index=False)
    subtype_distribution.to_csv(subtype_distribution_path, index=False)

    input_sha256 = _sha256(input_path)
    manifest_sha256 = {
        split_name: _sha256(manifest_path)
        for split_name, manifest_path in manifest_paths.items()
    }
    _write_split_audit_markdown(
        split_audit_path,
        split_summary=split_summary,
        overlap_summary=overlap_summary,
        subtype_distribution=subtype_distribution,
        seed=seed,
        input_sha256=input_sha256,
        manifest_sha256=manifest_sha256,
    )
    return GroupedAuditResult(
        split_summary_path=split_summary_path,
        group_overlap_summary_path=group_overlap_summary_path,
        subtype_distribution_path=subtype_distribution_path,
        split_audit_path=split_audit_path,
        input_sha256=input_sha256,
        manifest_sha256=manifest_sha256,
    )


def _validate_expected_keys(
    expected_sizes: dict[str, int],
    expected_subtypes: dict[str, dict[str, int]],
) -> None:
    if set(expected_sizes) != set(SPLIT_ORDER):
        raise ValueError(f"expected_split_sizes must contain exactly {list(SPLIT_ORDER)}")
    if set(expected_subtypes) != set(SPLIT_ORDER):
        raise ValueError(f"expected_subtype_counts must contain exactly {list(SPLIT_ORDER)}")


def _validate_source_manifest(source: pd.DataFrame) -> None:
    required_columns = {"image_path", "ood_type", "ood_subtype", "parent_image_hash"}
    missing = required_columns - set(source.columns)
    if missing:
        raise ValueError(f"Source manifest missing required columns: {sorted(missing)}")
    if source.empty:
        raise ValueError("Source manifest contains no rows")
    if source["image_path"].astype(str).str.strip().eq("").any():
        raise ValueError("Source manifest contains an empty image_path")
    if source["image_path"].duplicated().any():
        duplicate = source.loc[source["image_path"].duplicated(), "image_path"].iloc[0]
        raise ValueError(f"Source manifest contains duplicate image_path values: {duplicate}")


def _load_and_validate_split_manifest(
    manifest_path: Path,
    *,
    split_name: str,
    source: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required_columns = {"image_path", "split", "ood_type", "ood_subtype", "parent_image_hash"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name} manifest missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{split_name} manifest contains no rows")
    if set(frame["split"]) != {split_name}:
        raise ValueError(f"{split_name} manifest must contain only split={split_name!r} rows")
    if frame["image_path"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name} manifest contains an empty image_path")
    if frame["image_path"].duplicated().any():
        duplicate = frame.loc[frame["image_path"].duplicated(), "image_path"].iloc[0]
        raise ValueError(f"{split_name} manifest contains duplicate image_path values: {duplicate}")
    missing_source = sorted(set(frame["image_path"]) - set(source["image_path"]))
    if missing_source:
        raise ValueError(f"{split_name} manifest contains rows not present in the source manifest: {missing_source[0]}")

    comparison_columns = [column for column in frame.columns if column != "split"]
    source_for_comparison = source.copy()
    for column in comparison_columns:
        if column not in source_for_comparison.columns:
            source_for_comparison[column] = ""
    expected = (
        source_for_comparison.set_index("image_path", drop=False)
        .loc[frame["image_path"], comparison_columns]
        .reset_index(drop=True)
    )
    actual = frame[comparison_columns].reset_index(drop=True)
    expected = expected.fillna("")
    actual = actual.fillna("")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise ValueError(f"{split_name} manifest contains rows or metadata that do not match the source manifest") from error

    validated = frame.copy()
    validated["_normalized_group_id"] = [
        normalize_group_id(parent_hash, image_path)
        for parent_hash, image_path in zip(validated["parent_image_hash"], validated["image_path"])
    ]
    if validated["_normalized_group_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name} manifest contains an empty normalized group ID")
    return validated


def _build_overlap_summary(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    image_paths = {
        split_name: set(frame["image_path"].astype(str))
        for split_name, frame in split_frames.items()
    }
    group_ids = {
        split_name: set(frame["_normalized_group_id"].astype(str))
        for split_name, frame in split_frames.items()
    }
    rows = []
    pair_rows = [
        ("train-val", "train", "val"),
        ("train-test", "train", "test"),
        ("val-test", "val", "test"),
    ]
    for label, left_name, right_name in pair_rows:
        rows.append(
            {
                "overlap_scope": label,
                "image_path_overlap_count": len(image_paths[left_name] & image_paths[right_name]),
                "group_id_overlap_count": len(group_ids[left_name] & group_ids[right_name]),
            }
        )
    rows.append(
        {
            "overlap_scope": "all-three",
            "image_path_overlap_count": len(image_paths["train"] & image_paths["val"] & image_paths["test"]),
            "group_id_overlap_count": len(group_ids["train"] & group_ids["val"] & group_ids["test"]),
        }
    )
    return pd.DataFrame(
        rows,
        columns=["overlap_scope", "image_path_overlap_count", "group_id_overlap_count"],
    )


def _fail_on_overlap(overlap_summary: pd.DataFrame) -> None:
    problems: list[str] = []
    for row in overlap_summary.itertuples(index=False):
        if int(row.image_path_overlap_count) > 0:
            problems.append(f"{row.overlap_scope} image_path overlap={int(row.image_path_overlap_count)}")
        if int(row.group_id_overlap_count) > 0:
            problems.append(f"{row.overlap_scope} group_id overlap={int(row.group_id_overlap_count)}")
    if problems:
        raise ValueError("Grouped audit overlap detected: " + "; ".join(problems))


def _validate_source_coverage(source: pd.DataFrame, split_frames: dict[str, pd.DataFrame]) -> None:
    combined_paths = set().union(*(set(frame["image_path"].astype(str)) for frame in split_frames.values()))
    source_paths = set(source["image_path"].astype(str))
    missing_paths = sorted(source_paths - combined_paths)
    if missing_paths:
        raise ValueError(f"Grouped manifests are missing source rows: {missing_paths[0]}")
    extra_paths = sorted(combined_paths - source_paths)
    if extra_paths:
        raise ValueError(f"Grouped manifests contain rows outside the source manifest: {extra_paths[0]}")


def _validate_expected_counts(
    split_frames: dict[str, pd.DataFrame],
    expected_sizes: dict[str, int],
    expected_subtypes: dict[str, dict[str, int]],
) -> None:
    for split_name in SPLIT_ORDER:
        frame = split_frames[split_name]
        actual_size = int(len(frame))
        expected_size = int(expected_sizes[split_name])
        if actual_size != expected_size:
            raise ValueError(f"{split_name} manifest row count mismatch: expected {expected_size}, found {actual_size}")
        actual_subtypes = frame["ood_subtype"].value_counts().sort_index().to_dict()
        if actual_subtypes != expected_subtypes[split_name]:
            raise ValueError(
                f"{split_name} manifest subtype counts mismatch: expected {expected_subtypes[split_name]}, found {actual_subtypes}"
            )


def _build_split_summary(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name in SPLIT_ORDER:
        frame = split_frames[split_name]
        row = {
            "split": SPLIT_LABELS[split_name],
            "total_rows": int(len(frame)),
            "unique_image_paths": int(frame["image_path"].nunique()),
            "unique_groups": int(frame["_normalized_group_id"].nunique()),
        }
        for family in FAMILY_CLASSES:
            row[family] = int(frame["ood_type"].eq(family).sum())
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "total_rows",
            "unique_image_paths",
            "unique_groups",
            *FAMILY_CLASSES,
        ],
    )


def _build_subtype_distribution(source: pd.DataFrame, split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    subtype_to_family = (
        source[["ood_subtype", "ood_type"]]
        .drop_duplicates()
        .sort_values(["ood_type", "ood_subtype"], kind="stable")
        .set_index("ood_subtype")["ood_type"]
        .to_dict()
    )
    ordered_subtypes = [subtype for subtype in SUBTYPE_CLASSES if subtype in subtype_to_family]
    extra_subtypes = sorted(set(subtype_to_family) - set(ordered_subtypes))
    ordered_subtypes.extend(extra_subtypes)

    rows = []
    for subtype in ordered_subtypes:
        family = str(subtype_to_family[subtype])
        train_count = int(split_frames["train"]["ood_subtype"].eq(subtype).sum())
        val_count = int(split_frames["val"]["ood_subtype"].eq(subtype).sum())
        test_count = int(split_frames["test"]["ood_subtype"].eq(subtype).sum())
        rows.append(
            {
                "family": family,
                "subtype": subtype,
                "train": train_count,
                "validation": val_count,
                "test": test_count,
                "total": train_count + val_count + test_count,
            }
        )
    distribution = pd.DataFrame(
        rows,
        columns=["family", "subtype", "train", "validation", "test", "total"],
    )
    family_order = {family: index for index, family in enumerate(FAMILY_CLASSES)}
    subtype_order = {subtype: index for index, subtype in enumerate(SUBTYPE_CLASSES)}
    return (
        distribution.sort_values(
            by=["family", "subtype"],
            key=lambda column: column.map(
                family_order if column.name == "family" else subtype_order
            ).fillna(10_000),
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _write_split_audit_markdown(
    path: Path,
    *,
    split_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    subtype_distribution: pd.DataFrame,
    seed: int,
    input_sha256: str,
    manifest_sha256: dict[str, str],
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Stage 2 Grouped Split Audit",
                "",
                f"- seed: {seed}",
                "- algorithm: parent_grouped_stratified_split",
                "- ratios: train=0.6, validation=0.2, test=0.2",
                f"- input manifest sha256: {input_sha256}",
                f"- train manifest sha256: {manifest_sha256['train']}",
                f"- val manifest sha256: {manifest_sha256['val']}",
                f"- test manifest sha256: {manifest_sha256['test']}",
                "",
                "## Split Summary",
                "",
                dataframe_to_markdown(split_summary),
                "",
                "## Group Overlap Summary",
                "",
                dataframe_to_markdown(overlap_summary),
                "",
                "## Subtype Distribution",
                "",
                dataframe_to_markdown(subtype_distribution),
                "",
                "shared image paths across splits: 0",
                "shared group IDs across splits: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit deterministic parent-grouped Stage 2 manifests.")
    parser.add_argument("--input", required=True, help="Source manifest, usually test_ood_full.csv")
    parser.add_argument("--train", required=True, help="Grouped training manifest")
    parser.add_argument("--val", required=True, help="Grouped validation manifest")
    parser.add_argument("--test", required=True, help="Grouped test manifest")
    parser.add_argument("--out-dir", required=True, help="Directory for deterministic audit outputs")
    parser.add_argument("--seed", type=int, default=42, help="Recorded audit seed")
    args = parser.parse_args()
    result = audit_grouped_manifests(
        input_manifest=args.input,
        train_manifest=args.train,
        val_manifest=args.val,
        test_manifest=args.test,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    print(f"split_summary: {result.split_summary_path}")
    print(f"group_overlap_summary: {result.group_overlap_summary_path}")
    print(f"subtype_distribution: {result.subtype_distribution_path}")
    print(f"split_audit: {result.split_audit_path}")


if __name__ == "__main__":
    main()

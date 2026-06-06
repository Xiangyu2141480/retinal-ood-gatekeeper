"""Image-level dataset audit utilities for local retinal OOD experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from retinal_ood.data.manifest_audit import (
    PRIVATE_PATH_RE,
    REQUIRED_COLUMNS,
    VALID_OOD_TYPES,
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUBTYPE_REQUIRED_OOD_TYPES = {"modality_shift", "sensory_artifact"}
TINY_IMAGE_MIN_SIZE = 32
NEAR_BLANK_STD_THRESHOLD = 1.0
SUBTYPE_PATTERNS = {
    "colour_fundus": ["colour_fundus", "color_fundus"],
    "oct_screenshot": ["oct_screenshot"],
    "cifar10_natural": ["cifar10", "ood_semantic/natural", "natural/"],
    "text_watermark": ["text_watermark"],
    "rectangle_annotation": ["rectangle_annotation"],
    "arrow_annotation": ["arrow_annotation"],
    "composite_layout": ["composite_layout"],
    "blur_artifact": ["blur_artifact"],
    "border_crop": ["border_crop"],
    "gaussian_noise": ["gaussian_noise"],
    "jpeg_compression": ["jpeg_compression"],
}


class ImageAuditError(ValueError):
    """Raised when the image audit finds a hard dataset-quality failure."""


def audit_dataset_images(
    manifest_paths: list[str | Path],
    *,
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    write_json: str | Path | None = None,
    make_contact_sheets: bool = False,
    sample_per_group: int = 16,
    fail_on_corrupt: bool = False,
    fail_on_duplicate_content_across_splits: bool = False,
) -> dict[str, Any]:
    """Audit image files referenced by one or more manifests.

    The audit reads image metadata and aggregate intensity statistics only. Raw images,
    generated contact sheets, and local JSON reports should remain outside Git history.
    """
    if not manifest_paths:
        raise ImageAuditError("At least one manifest is required")
    if sample_per_group <= 0:
        raise ValueError("sample_per_group must be positive")

    root = Path(root_dir).resolve()
    manifests = _read_manifest_rows(manifest_paths)
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    stats_rows: list[dict[str, float]] = []
    duplicate_tracker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corrupt_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    private_path_rows: list[dict[str, Any]] = []

    metadata_totals = _validate_manifest_metadata(manifests, errors, warnings)

    for row_idx, row in manifests.iterrows():
        image_path = str(row["image_path"]).strip()
        manifest_name = str(row["_manifest"])
        if _is_private_or_absolute_path(image_path):
            private_path_rows.append(_issue_row(row, Path(image_path), "private_path"))
            errors.append(
                f"{manifest_name}:{row_idx + 2}: private/absolute image_path is not allowed: {image_path}"
            )
            continue
        resolved = (root / image_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append(f"{manifest_name}:{row_idx + 2}: image_path resolves outside --root-dir: {image_path}")
            continue
        if not resolved.exists():
            missing_rows.append(_issue_row(row, resolved, "missing"))
            continue

        record = _base_record(row, resolved)
        try:
            image_stats = _inspect_image(resolved)
        except (OSError, UnidentifiedImageError) as exc:
            corrupt_rows.append({**record, "error": str(exc)})
            continue
        record.update(image_stats["record"])
        records.append(record)
        stats_rows.append(image_stats["stats"])
        duplicate_tracker[record["sha256"]].append(record)

    if missing_rows:
        errors.append(f"{len(missing_rows)} missing image files found")
    if corrupt_rows and fail_on_corrupt:
        errors.append(f"{len(corrupt_rows)} corrupt image files found")

    duplicate_content = _duplicate_content(duplicate_tracker)
    duplicate_across_splits = _duplicate_content_across_splits(duplicate_content)
    if duplicate_across_splits and fail_on_duplicate_content_across_splits:
        errors.append(f"{len(duplicate_across_splits)} duplicate content across splits found")

    audit = _build_audit(
        manifests,
        records,
        stats_rows,
        root=root,
        missing_rows=missing_rows,
        corrupt_rows=corrupt_rows,
        duplicate_content=duplicate_content,
        duplicate_across_splits=duplicate_across_splits,
        warnings=warnings,
        private_path_rows=private_path_rows,
        metadata_totals=metadata_totals,
    )
    if make_contact_sheets:
        if out_dir is None:
            raise ValueError("out_dir must be provided when make_contact_sheets=True")
        audit["contact_sheets"] = _write_contact_sheets(
            records,
            root=root,
            out_dir=Path(out_dir) / "contact_sheets",
            sample_per_group=sample_per_group,
        )
    else:
        audit["contact_sheets"] = []

    if write_json is not None:
        out_path = Path(write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if errors:
        raise ImageAuditError("; ".join(errors))
    return audit


def format_image_audit_summary(audit: dict[str, Any]) -> str:
    """Render a concise CLI summary for PR validation logs."""
    totals = audit.get("totals", {})
    lines = [
        f"Root dir: {audit.get('root_dir', '')}",
        f"Images: {totals.get('images', 0)}",
        f"Opened OK: {totals.get('opened_ok', 0)}",
        f"Missing images: {totals.get('missing_images', 0)}",
        f"Corrupt images: {totals.get('corrupt_images', 0)}",
        f"Private paths: {totals.get('private_paths', 0)}",
        f"Non-empty patient_id: {totals.get('non_empty_patient_id', 0)}",
        f"Blank/near-blank images: {totals.get('blank_or_near_blank_images', 0)}",
        f"All-black images: {totals.get('all_black_images', 0)}",
        f"All-white images: {totals.get('all_white_images', 0)}",
        f"Tiny images: {totals.get('suspiciously_tiny_images', 0)}",
        f"Duplicate SHA256 groups: {totals.get('duplicate_content_groups', 0)}",
        f"Duplicate content across splits: {totals.get('duplicate_content_across_splits', 0)}",
        f"Contact sheets: {len(audit.get('contact_sheets', []))}",
    ]
    if audit.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in audit["warnings"])
    return "\n".join(lines)


def _read_manifest_rows(manifest_paths: list[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        try:
            frame = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise ImageAuditError(f"Manifest file does not exist: {path}") from exc
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ImageAuditError(f"Manifest {path} missing required columns: {missing}")
        frame = frame.copy()
        frame["_manifest"] = path.as_posix()
        frames.append(frame)
    if not frames:
        raise ImageAuditError("No manifests were provided")
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise ImageAuditError("Manifest rows are empty")
    for column in ["image_path", "split", "source", "ood_type", "ood_subtype", "patient_id"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna("").astype(str).str.strip()
    if "ood_subtype" not in merged.columns:
        merged["ood_subtype"] = ""
    merged["_explicit_ood_subtype_missing"] = merged["ood_subtype"].eq("")
    merged["ood_subtype"] = [
        subtype or infer_ood_subtype_from_row(row)
        for subtype, (_, row) in zip(merged["ood_subtype"].tolist(), merged.iterrows())
    ]
    return merged


def _validate_manifest_metadata(df: pd.DataFrame, errors: list[str], warnings: list[str]) -> dict[str, int]:
    totals = {"non_empty_patient_id": 0}
    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("manifest labels must be 0 or 1")
    if "patient_id" in df.columns:
        non_empty = df["patient_id"].fillna("").astype(str).str.strip().ne("")
        totals["non_empty_patient_id"] = int(non_empty.sum())
        if non_empty.any():
            errors.append(f"{int(non_empty.sum())} non-empty patient_id values found")
    invalid_ood = ~df["ood_type"].isin(VALID_OOD_TYPES)
    if invalid_ood.any():
        errors.append(f"invalid ood_type values found: {sorted(df.loc[invalid_ood, 'ood_type'].unique())}")
    train_bad = (df["split"] == "train") & ((labels != 0) | (df["ood_type"] != "id"))
    if train_bad.any():
        errors.append("train split must contain label=0 and ood_type=id only")
    _warn_missing_subtypes(df, warnings)
    return totals


def _warn_missing_subtypes(df: pd.DataFrame, warnings: list[str]) -> None:
    explicit_missing = df.get(
        "_explicit_ood_subtype_missing",
        pd.Series(False, index=df.index),
    ).fillna(False)
    subtype = df["ood_subtype"].fillna("").astype(str).str.strip()
    for ood_type in sorted(SUBTYPE_REQUIRED_OOD_TYPES):
        explicit_count = int(((df["ood_type"] == ood_type) & explicit_missing).sum())
        unresolved_count = int(((df["ood_type"] == ood_type) & subtype.eq("")).sum())
        if explicit_count:
            warnings.append(
                f"{explicit_count} {ood_type} rows are missing explicit ood_subtype; "
                "the audit inferred values from image_path/notes where possible"
            )
        if unresolved_count:
            warnings.append(f"{unresolved_count} {ood_type} rows still have unresolved ood_subtype")
    semantic = df["ood_type"] == "semantic_outlier"
    if semantic.any():
        source = df.get("source", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        count = int((semantic & subtype.eq("") & source.eq("")).sum())
        if count:
            warnings.append(f"{count} semantic_outlier rows are missing both source and ood_subtype")


def infer_ood_subtype_from_row(row: pd.Series | dict[str, Any]) -> str:
    """Infer a local OOD subtype from image_path/notes when manifests omit it."""
    text = " ".join(
        str(row.get(field, ""))
        for field in ["image_path", "notes", "source"]
    ).lower().replace("\\", "/")
    for subtype, patterns in SUBTYPE_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return subtype
    return ""


def _base_record(row: pd.Series, resolved: Path) -> dict[str, Any]:
    return {
        "manifest": str(row["_manifest"]),
        "image_path": str(row["image_path"]).replace("\\", "/"),
        "resolved_path": resolved.as_posix(),
        "label": int(row["label"]),
        "split": str(row["split"]),
        "source": str(row.get("source", "")),
        "ood_type": str(row.get("ood_type", "")),
        "ood_subtype": str(row.get("ood_subtype", "")),
        "extension": resolved.suffix.lower(),
    }


def _inspect_image(path: Path) -> dict[str, Any]:
    sha = _sha256_file(path)
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
    intensity_min = float(arr.min())
    intensity_max = float(arr.max())
    intensity_mean = float(arr.mean())
    intensity_std = float(arr.std())
    is_near_blank = intensity_std <= NEAR_BLANK_STD_THRESHOLD
    record = {
        "sha256": sha,
        "mode": mode,
        "width": int(width),
        "height": int(height),
        "size": f"{width}x{height}",
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "intensity_mean": intensity_mean,
        "intensity_std": intensity_std,
        "blank_or_near_blank": bool(is_near_blank),
        "all_black": bool(intensity_max == 0.0),
        "all_white": bool(intensity_min == 255.0),
        "suspiciously_tiny": bool(width < TINY_IMAGE_MIN_SIZE or height < TINY_IMAGE_MIN_SIZE),
    }
    return {"record": record, "stats": record}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _issue_row(row: pd.Series, resolved: Path, issue: str) -> dict[str, Any]:
    return {
        "manifest": str(row.get("_manifest", "")),
        "image_path": str(row.get("image_path", "")),
        "resolved_path": resolved.as_posix(),
        "split": str(row.get("split", "")),
        "issue": issue,
    }


def _duplicate_content(duplicate_tracker: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sha, records in sorted(duplicate_tracker.items()):
        if len(records) <= 1:
            continue
        rows.append(
            {
                "sha256": sha,
                "count": len(records),
                "splits": sorted({record["split"] for record in records}),
                "image_paths": [record["image_path"] for record in records],
            }
        )
    return rows


def _duplicate_content_across_splits(duplicate_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in duplicate_content if len(row["splits"]) > 1]


def _build_audit(
    manifests: pd.DataFrame,
    records: list[dict[str, Any]],
    stats_rows: list[dict[str, float]],
    *,
    root: Path,
    missing_rows: list[dict[str, Any]],
    corrupt_rows: list[dict[str, Any]],
    duplicate_content: list[dict[str, Any]],
    duplicate_across_splits: list[dict[str, Any]],
    warnings: list[str],
    private_path_rows: list[dict[str, Any]],
    metadata_totals: dict[str, int],
) -> dict[str, Any]:
    records_df = pd.DataFrame(records)
    stats = _intensity_summary(stats_rows)
    totals = {
        "rows": int(len(manifests)),
        "images": int(len(records)),
        "opened_ok": int(len(records)),
        "missing_images": int(len(missing_rows)),
        "corrupt_images": int(len(corrupt_rows)),
        "private_paths": int(len(private_path_rows)),
        "non_empty_patient_id": int(metadata_totals.get("non_empty_patient_id", 0)),
        "blank_or_near_blank_images": _count_bool(records_df, "blank_or_near_blank"),
        "all_black_images": _count_bool(records_df, "all_black"),
        "all_white_images": _count_bool(records_df, "all_white"),
        "suspiciously_tiny_images": _count_bool(records_df, "suspiciously_tiny"),
        "duplicate_content_groups": len(duplicate_content),
        "duplicate_content_across_splits": len(duplicate_across_splits),
    }
    return {
        "root_dir": root.as_posix(),
        "totals": totals,
        "counts": _counts(manifests),
        "extension_distribution": _value_counts(records_df, "extension"),
        "image_mode_distribution": _value_counts(records_df, "mode"),
        "width_distribution": _value_counts(records_df, "width"),
        "height_distribution": _value_counts(records_df, "height"),
        "size_distribution": _value_counts(records_df, "size"),
        "intensity_summary": stats,
        "blank_or_near_blank_images": _flagged_records(records, "blank_or_near_blank"),
        "all_black_images": _flagged_records(records, "all_black"),
        "all_white_images": _flagged_records(records, "all_white"),
        "suspiciously_tiny_images": _flagged_records(records, "suspiciously_tiny"),
        "duplicate_content": duplicate_content,
        "duplicate_content_across_splits": duplicate_across_splits,
        "missing_images": missing_rows,
        "corrupt_images": corrupt_rows,
        "private_paths": private_path_rows,
        "warnings": warnings,
    }


def _counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    fields = ["_manifest", "label", "split", "source", "ood_type"]
    if "ood_subtype" in df.columns:
        fields.append("ood_subtype")
    output: dict[str, dict[str, int]] = {}
    for field in fields:
        key = "manifest" if field == "_manifest" else field
        output[key] = dict(sorted(Counter(str(value) for value in df[field].fillna("").tolist()).items()))
    return output


def _intensity_summary(stats_rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not stats_rows:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    mins = [row["intensity_min"] for row in stats_rows]
    maxes = [row["intensity_max"] for row in stats_rows]
    means = np.asarray([row["intensity_mean"] for row in stats_rows], dtype=float)
    stds = np.asarray([row["intensity_std"] for row in stats_rows], dtype=float)
    return {
        "count": int(len(stats_rows)),
        "min": float(min(mins)),
        "max": float(max(maxes)),
        "mean_of_image_means": float(means.mean()),
        "std_of_image_means": float(means.std()),
        "mean_of_image_stds": float(stds.mean()),
    }


def _count_bool(records_df: pd.DataFrame, column: str) -> int:
    if records_df.empty or column not in records_df.columns:
        return 0
    return int(records_df[column].fillna(False).astype(bool).sum())


def _value_counts(records_df: pd.DataFrame, column: str) -> dict[str, int]:
    if records_df.empty or column not in records_df.columns:
        return {}
    return dict(sorted(Counter(str(value) for value in records_df[column].fillna("").tolist()).items()))


def _flagged_records(records: list[dict[str, Any]], flag: str) -> list[dict[str, Any]]:
    fields = ["image_path", "split", "source", "ood_type", "ood_subtype", "width", "height", "intensity_mean", "intensity_std"]
    return [{field: record.get(field, "") for field in fields} for record in records if record.get(flag)]


def _write_contact_sheets(
    records: list[dict[str, Any]],
    *,
    root: Path,
    out_dir: Path,
    sample_per_group: int,
) -> list[str]:
    if not records:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for group_field in ["source", "ood_type", "ood_subtype"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            value = str(record.get(group_field, "")).strip() or "missing"
            groups[value].append(record)
        for group_value, group_records in sorted(groups.items()):
            selected = sorted(group_records, key=lambda item: item["image_path"])[:sample_per_group]
            filename = f"{group_field}_{_safe_stem(group_value)}.jpg"
            path = out_dir / filename
            _save_contact_sheet(selected, path)
            written.append(path.relative_to(out_dir.parent).as_posix())
    return sorted(written)


def _save_contact_sheet(records: list[dict[str, Any]], path: Path, *, thumb_size: int = 96) -> None:
    if not records:
        return
    cols = min(4, len(records))
    rows = math.ceil(len(records) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(255, 255, 255))
    for idx, record in enumerate(records):
        with Image.open(record["resolved_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size))
            x = (idx % cols) * thumb_size + (thumb_size - thumb.width) // 2
            y = (idx // cols) * thumb_size + (thumb_size - thumb.height) // 2
            sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "missing"


def _is_private_or_absolute_path(image_path: str) -> bool:
    if PRIVATE_PATH_RE.search(image_path):
        return True
    if Path(image_path).is_absolute():
        return True
    return PureWindowsPath(image_path).is_absolute()

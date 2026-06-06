"""Manifest validation and audit utilities for local FAF OOD experiments."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["image_path", "label", "split", "source", "ood_type"]
RECOMMENDED_COLUMNS = [
    "patient_id",
    "scanner",
    "notes",
    "ood_subtype",
    "source_split",
    "source_image_hash",
]
VALID_LABELS = {0, 1}
VALID_SPLITS = {"train", "val", "test"}
VALID_OOD_TYPES = {"id", "modality_shift", "sensory_artifact", "semantic_outlier"}
PRIVATE_PATH_RE = re.compile(r"(^|[;,\s])(?:[A-Za-z]:[\\/])?Users[\\/]", re.IGNORECASE)


class ManifestAuditError(ValueError):
    """Raised when a manifest violates a hard validation rule."""


def audit_manifests(
    manifest_paths: list[str | Path],
    *,
    root_dir: str | Path,
    check_files: bool = True,
    fail_on_duplicates: bool = False,
    allow_train_artifact: bool = False,
    write_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate manifest CSVs and return a serializable audit summary.

    The audit is intentionally conservative: training rows must be ID-only, OOD rows must
    use the literature-review taxonomy, and paths must stay relative to ``root_dir``.
    """
    if not manifest_paths:
        raise ManifestAuditError("At least one manifest path is required")

    root = Path(root_dir).resolve()
    warnings: list[str] = []
    manifest_summaries: list[dict[str, Any]] = []
    duplicate_tracker: dict[str, set[str]] = defaultdict(set)
    duplicate_locations: dict[str, list[str]] = defaultdict(list)
    totals = {
        "rows": 0,
        "missing_files": 0,
        "duplicate_image_paths": 0,
        "non_empty_patient_id": 0,
    }

    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        df = _read_manifest(path)
        _validate_required_columns(path, df)
        df = _normalized_df(df)
        summary = _empty_manifest_summary(path)
        summary["rows"] = int(len(df))
        summary["recommended_missing_columns"] = [
            column for column in RECOMMENDED_COLUMNS if column not in df.columns
        ]

        if df.empty:
            raise ManifestAuditError(f"Manifest {path} contains no rows")

        _validate_labels(path, df)
        _validate_splits(path, df)
        _validate_ood_types(path, df)
        _validate_label_ood_consistency(path, df)
        _validate_training_rows(path, df)
        _validate_official_ood_manifest(path, df)
        _validate_paths(path, df, root, check_files, summary)
        _validate_train_artifact_sources(path, df, allow_train_artifact)

        non_empty_patient_id = _count_non_empty_patient_ids(df)
        summary["non_empty_patient_id"] = non_empty_patient_id
        if non_empty_patient_id:
            warnings.append(f"{path}: {non_empty_patient_id} non-empty patient_id values found")

        if _looks_like_synthetic_fallback_real_id(path, df):
            warnings.append(
                f"{path}: filename suggests real ID validation, but all sources look synthetic; "
                "treat this as a synthetic fallback, not real clinical FAF validation"
            )

        summary["counts"] = _counts(df)
        manifest_summaries.append(summary)
        totals["rows"] += summary["rows"]
        totals["missing_files"] += summary["missing_files"]
        totals["non_empty_patient_id"] += non_empty_patient_id
        _record_duplicates(path, df, duplicate_tracker, duplicate_locations)

    duplicate_paths = {
        image_path: locations
        for image_path, locations in duplicate_locations.items()
        if len(duplicate_tracker[image_path]) > 1
    }
    totals["duplicate_image_paths"] = len(duplicate_paths)
    if duplicate_paths:
        warning = (
            f"{len(duplicate_paths)} duplicate image_path values appear across train/val/test splits"
        )
        warnings.append(warning)
        if fail_on_duplicates:
            raise ManifestAuditError(f"duplicate image_path values found: {duplicate_paths}")

    audit = {
        "root_dir": str(root),
        "manifests": manifest_summaries,
        "totals": totals,
        "warnings": warnings,
    }
    if write_json is not None:
        out_path = Path(write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def format_audit_summary(audit: dict[str, Any]) -> str:
    """Render a concise human-readable audit summary."""
    lines = [f"Root dir: {audit['root_dir']}"]
    for manifest in audit["manifests"]:
        lines.append("")
        lines.append(f"Manifest: {manifest['path']}")
        lines.append(f"  rows: {manifest['rows']}")
        lines.append(f"  missing_files: {manifest['missing_files']}")
        lines.append(f"  non_empty_patient_id: {manifest['non_empty_patient_id']}")
        for field, counts in manifest["counts"].items():
            lines.append(f"  {field}: {counts}")
    lines.append("")
    lines.append(f"Totals: {audit['totals']}")
    if audit["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in audit["warnings"])
    return "\n".join(lines)


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ManifestAuditError(f"Manifest {path} is empty") from exc
    except FileNotFoundError as exc:
        raise ManifestAuditError(f"Manifest file does not exist: {path}") from exc


def _validate_required_columns(path: Path, df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ManifestAuditError(f"Manifest {path} missing required columns: {missing}")


def _normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["image_path", "split", "source", "ood_type", *RECOMMENDED_COLUMNS]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def _empty_manifest_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": 0,
        "counts": {},
        "missing_files": 0,
        "duplicate_image_paths": 0,
        "non_empty_patient_id": 0,
        "recommended_missing_columns": [],
    }


def _validate_labels(path: Path, df: pd.DataFrame) -> None:
    labels = pd.to_numeric(df["label"], errors="coerce")
    invalid = labels.isna() | ~labels.isin(VALID_LABELS)
    if invalid.any():
        examples = df.loc[invalid, "label"].head(5).tolist()
        raise ManifestAuditError(f"Manifest {path} has invalid label values: {examples}")
    df["label"] = labels.astype(int)


def _validate_splits(path: Path, df: pd.DataFrame) -> None:
    invalid = ~df["split"].isin(VALID_SPLITS)
    if invalid.any():
        examples = df.loc[invalid, "split"].head(5).tolist()
        raise ManifestAuditError(f"Manifest {path} has invalid split values: {examples}")


def _validate_ood_types(path: Path, df: pd.DataFrame) -> None:
    invalid = ~df["ood_type"].isin(VALID_OOD_TYPES)
    if invalid.any():
        examples = df.loc[invalid, "ood_type"].head(5).tolist()
        raise ManifestAuditError(
            f"Manifest {path} has invalid ood_type values: {examples}; "
            f"expected one of {sorted(VALID_OOD_TYPES)}"
        )


def _validate_label_ood_consistency(path: Path, df: pd.DataFrame) -> None:
    id_bad = (df["label"] == 0) & (df["ood_type"] != "id")
    if id_bad.any():
        raise ManifestAuditError(f"Manifest {path}: label=0 rows must have ood_type=id")
    ood_bad = (df["label"] == 1) & (df["ood_type"] == "id")
    if ood_bad.any():
        raise ManifestAuditError(f"Manifest {path}: label=1 rows must not have ood_type=id")


def _validate_training_rows(path: Path, df: pd.DataFrame) -> None:
    train = df["split"] == "train"
    bad = train & ((df["label"] != 0) | (df["ood_type"] != "id"))
    if bad.any():
        raise ManifestAuditError(f"Manifest {path}: train rows must all be label=0 and ood_type=id")


def _validate_official_ood_manifest(path: Path, df: pd.DataFrame) -> None:
    name = path.name.lower()
    looks_ood = any(token in name for token in ["ood", "artifact", "semantic", "modality"])
    if looks_ood and (df["split"] == "train").any():
        raise ManifestAuditError(f"Manifest {path}: official OOD manifests must not contain split=train")


def _validate_paths(path: Path, df: pd.DataFrame, root: Path, check_files: bool, summary: dict[str, Any]) -> None:
    missing_files = 0
    for raw in df["image_path"].tolist():
        image_path = str(raw).strip()
        if not image_path:
            raise ManifestAuditError(f"Manifest {path} contains empty image_path values")
        if _is_private_or_absolute_path(image_path):
            raise ManifestAuditError(f"Manifest {path} contains absolute/private local paths: {image_path}")
        resolved = (root / image_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestAuditError(f"Manifest {path} image_path resolves outside --root-dir: {image_path}") from exc
        if check_files and not resolved.exists():
            missing_files += 1
    summary["missing_files"] = missing_files
    if check_files and missing_files:
        raise ManifestAuditError(f"Manifest {path} has {missing_files} missing image files")


def _is_private_or_absolute_path(image_path: str) -> bool:
    if PRIVATE_PATH_RE.search(image_path):
        return True
    if Path(image_path).is_absolute():
        return True
    return PureWindowsPath(image_path).is_absolute()


def _validate_train_artifact_sources(path: Path, df: pd.DataFrame, allow_train_artifact: bool) -> None:
    if allow_train_artifact or "source_split" not in df.columns:
        return
    bad = (df["ood_type"] == "sensory_artifact") & (df["source_split"].str.lower() == "train")
    if bad.any():
        raise ManifestAuditError(
            f"Manifest {path}: sensory_artifact rows must not use source_split=train "
            "unless --allow-train-artifact is passed"
        )


def _count_non_empty_patient_ids(df: pd.DataFrame) -> int:
    if "patient_id" not in df.columns:
        return 0
    return int(df["patient_id"].fillna("").astype(str).str.strip().ne("").sum())


def _looks_like_synthetic_fallback_real_id(path: Path, df: pd.DataFrame) -> bool:
    if "real" not in path.name.lower():
        return False
    sources = [str(source).lower() for source in df["source"].dropna().unique().tolist()]
    return bool(sources) and all("synthetic" in source for source in sources)


def _counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    fields = ["label", "split", "source", "ood_type"]
    if "ood_subtype" in df.columns:
        fields.append("ood_subtype")
    output: dict[str, dict[str, int]] = {}
    for field in fields:
        counts = Counter(str(value) for value in df[field].fillna("").tolist())
        output[field] = dict(sorted(counts.items()))
    return output


def _record_duplicates(
    path: Path,
    df: pd.DataFrame,
    duplicate_tracker: dict[str, set[str]],
    duplicate_locations: dict[str, list[str]],
) -> None:
    for row_idx, row in df.iterrows():
        image_path = str(row["image_path"]).replace("\\", "/").strip().lower()
        split = str(row["split"])
        duplicate_tracker[image_path].add(split)
        duplicate_locations[image_path].append(f"{path}:{row_idx + 2}:{split}")

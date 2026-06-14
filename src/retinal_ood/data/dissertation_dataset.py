"""Build the dissertation v1 manifest package for FAF OOD gatekeeper experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter

from retinal_ood.data.artifacts import ARTIFACT_TYPES, apply_artifact
from retinal_ood.data.manifest_audit import PRIVATE_PATH_RE, VALID_OOD_TYPES

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ARTIFACT_SUBTYPES: tuple[str, ...] = tuple(ARTIFACT_TYPES)
MODALITY_SUBTYPES = ("colour_fundus", "oct_screenshot")
SEMANTIC_SUBTYPE = "cifar10_natural"
EXPECTED_DATASET_CONTRACT_ID = "dissertation_faf_ood_v1"
SYNTHETIC_FALLBACK_NOTE = "synthetic fallback only; not real clinical FAF validation"
BANNED_PRIVATE_COLUMNS = {"Eye_ID", "disease_label", "clinical_label", "biomarker_label"}
MANIFEST_COLUMNS = [
    "image_path",
    "label",
    "split",
    "source",
    "ood_type",
    "ood_subtype",
    "source_dataset",
    "source_split",
    "source_image_hash",
    "parent_image_hash",
    "synthetic_transform",
    "semantic_class",
    "is_synthetic_surrogate",
    "patient_id",
    "scanner",
    "license_status",
    "notes",
]


class DissertationDatasetError(ValueError):
    """Raised when the dissertation dataset package violates a hard contract."""


@dataclass(frozen=True)
class DissertationDatasetBuildResult:
    """Paths and safe aggregate metadata emitted by the dissertation dataset builder."""

    local_manifests: dict[str, Path]
    repo_manifests: dict[str, Path]
    repo_dataset_dir: Path
    audit_dir: Path
    audit_summary: dict[str, Any]
    warnings: list[str]


@dataclass(frozen=True)
class DissertationImageArchiveResult:
    """Safe aggregate metadata for a curated dissertation image archive."""

    archive_path: Path
    manifest_path: Path
    image_count: int
    total_bytes: int
    sha256: str


@dataclass(frozen=True)
class DissertationChecksumResult:
    """Summary for checksum writing or verification."""

    checksum_path: Path
    checked_files: int
    total_bytes: int


def prepare_dissertation_dataset(
    *,
    config_path: str | Path,
    root_dir: str | Path,
    prepared_syntheye_dir: str | Path,
    out_image_dir: str | Path,
    repo_dataset_dir: str | Path,
    local_manifest_dir: str | Path | None = None,
    audit_dir: str | Path | None = None,
    prepared_colour_fundus_dir: str | Path | None = None,
    prepared_oct_dir: str | Path | None = None,
    prepared_cifar_dir: str | Path | None = None,
    syntheye_dir: str | Path | None = None,
    colour_fundus_dir: str | Path | None = None,
    oct_dir: str | Path | None = None,
    cifar_root: str | Path | None = None,
    seed: int | None = None,
    allow_download_cifar: bool = False,
    allow_synthetic_surrogates: bool = False,
    allow_val_artifact: bool = False,
    commit_safe_manifest_package: bool = False,
    commit_individual_lfs_images: bool = False,
) -> DissertationDatasetBuildResult:
    """Create local curated images plus a commit-safe dissertation manifest package.

    The training split is strictly ID-only synthetic FAF. OOD rows are evaluation-only
    and always carry an explicit literature-review subtype.
    """

    config = _read_config(config_path)
    _validate_config(config)
    build_seed = int(seed if seed is not None else config.get("seed", 42))
    rng = np.random.default_rng(build_seed)
    warnings: list[str] = []

    if allow_download_cifar:
        warnings.append(
            "allow_download_cifar was provided, but this builder does not download data; "
            "prepare CIFAR/Open Images locally and pass --prepared-cifar-dir"
        )

    root = Path(root_dir)
    out_images = Path(out_image_dir)
    local_manifests_dir = Path(local_manifest_dir) if local_manifest_dir is not None else root / "manifests" / "generated" / "dissertation_v1"
    repo_dir = Path(repo_dataset_dir)
    repo_manifest_dir = repo_dir / "manifests"
    local_audit_dir = Path(audit_dir) if audit_dir is not None else Path("reports") / "local_audits" / "dissertation_v1"
    if commit_individual_lfs_images:
        _reset_direct_image_output_dir(out_images, root)

    for directory in [out_images, local_manifests_dir, repo_manifest_dir, local_audit_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    id_source = Path(syntheye_dir or prepared_syntheye_dir)
    train_rows, val_rows, test_id_rows = _prepare_id_rows(
        source_dir=id_source,
        out_image_dir=out_images,
        root_dir=root,
        seed=build_seed,
        warnings=warnings,
    )
    _fail_duplicate_content_across_splits([*train_rows, *val_rows, *test_id_rows])

    artifact_source_splits = {"test"} if not allow_val_artifact else {"val", "test"}
    artifact_sources = [
        *test_id_rows,
        *(val_rows if allow_val_artifact else []),
    ]
    artifact_rows = _prepare_artifact_rows(
        id_rows=artifact_sources,
        out_image_dir=out_images,
        root_dir=root,
        rng=rng,
        allowed_source_splits=artifact_source_splits,
    )
    modality_rows = _prepare_modality_rows(
        colour_dir=Path(colour_fundus_dir or prepared_colour_fundus_dir)
        if (colour_fundus_dir or prepared_colour_fundus_dir)
        else None,
        oct_dir=Path(oct_dir or prepared_oct_dir) if (oct_dir or prepared_oct_dir) else None,
        out_image_dir=out_images,
        root_dir=root,
        rng=rng,
        config=config,
        allow_synthetic_surrogates=allow_synthetic_surrogates,
        warnings=warnings,
    )
    semantic_rows = _prepare_semantic_rows(
        source_dir=Path(cifar_root or prepared_cifar_dir) if (cifar_root or prepared_cifar_dir) else None,
        out_image_dir=out_images,
        root_dir=root,
        rng=rng,
        config=config,
        allow_synthetic_surrogates=allow_synthetic_surrogates,
        warnings=warnings,
    )

    if not modality_rows:
        raise DissertationDatasetError(
            "No modality_shift rows were created; pass prepared modality folders or "
            "--allow-synthetic-surrogates for clearly marked surrogate rows"
        )
    if not semantic_rows:
        raise DissertationDatasetError(
            "No semantic_outlier rows were created; pass a prepared natural-image folder or "
            "--allow-synthetic-surrogates for clearly marked surrogate rows"
        )

    test_ood_rows = [*artifact_rows, *modality_rows, *semantic_rows]
    balanced_by_type = _balanced_by_group(test_ood_rows, group_field="ood_type", seed=build_seed)
    balanced_by_subtype = _balanced_by_group(test_ood_rows, group_field="ood_subtype", seed=build_seed)
    smoke_rows = _smoke_by_subtype(test_ood_rows, per_subtype=2, seed=build_seed)

    manifest_rows = {
        "train_id": train_rows,
        "val_id": val_rows,
        "test_id_synthetic_fallback": test_id_rows,
        "test_artifact": artifact_rows,
        "test_ood_artifact": artifact_rows,
        "test_modality": modality_rows,
        "test_ood_modality": modality_rows,
        "test_semantic": semantic_rows,
        "test_ood_semantic": semantic_rows,
        "test_ood": test_ood_rows,
        "test_ood_full": test_ood_rows,
        "test_ood_balanced_by_type": balanced_by_type,
        "test_ood_balanced_by_subtype": balanced_by_subtype,
        "test_ood_smoke": smoke_rows,
    }
    _validate_manifest_collection(manifest_rows)

    local_paths = _write_manifests(manifest_rows, local_manifests_dir)
    repo_paths = _write_manifests(manifest_rows, repo_manifest_dir)
    audit_summary = _build_audit_summary(manifest_rows, warnings=warnings, seed=build_seed)
    (local_audit_dir / "manifest_audit.json").write_text(
        json.dumps(audit_summary, indent=2),
        encoding="utf-8",
    )

    if commit_safe_manifest_package or commit_individual_lfs_images:
        _write_repo_package_docs(
            repo_dir,
            manifest_rows=manifest_rows,
            audit_summary=audit_summary,
            warnings=warnings,
            individual_lfs_images=commit_individual_lfs_images,
        )

    return DissertationDatasetBuildResult(
        local_manifests=local_paths,
        repo_manifests=repo_paths,
        repo_dataset_dir=repo_dir,
        audit_dir=local_audit_dir,
        audit_summary=audit_summary,
        warnings=warnings,
    )


def create_dissertation_image_archive(
    *,
    image_dir: str | Path,
    root_dir: str | Path,
    archive_path: str | Path,
) -> DissertationImageArchiveResult:
    """Create a deterministic ZIP archive of curated local images for Git LFS.

    Archive members are stored relative to ``root_dir`` so unpacking into a
    school-server data root recreates paths used by the manifests.
    """

    source_dir = Path(image_dir)
    root = Path(root_dir).resolve()
    archive = Path(archive_path)
    if not source_dir.exists():
        raise DissertationDatasetError(f"Image directory does not exist: {source_dir}")
    files = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not files:
        raise DissertationDatasetError(f"Image directory contains no supported images: {source_dir}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in files:
            resolved = path.resolve()
            try:
                arcname = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise DissertationDatasetError(f"Archive image is outside root_dir: {path}") from exc
            if _is_private_or_absolute_path(arcname):
                raise DissertationDatasetError(f"Archive member path is unsafe: {arcname}")
            info = zipfile.ZipInfo(arcname)
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            data = path.read_bytes()
            handle.writestr(info, data)
            rows.append(
                {
                    "image_path": arcname,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
            total_bytes += len(data)

    manifest_path = archive.with_suffix(".manifest.csv")
    pd.DataFrame(rows, columns=["image_path", "sha256", "bytes"]).to_csv(manifest_path, index=False)
    return DissertationImageArchiveResult(
        archive_path=archive,
        manifest_path=manifest_path,
        image_count=len(files),
        total_bytes=total_bytes,
        sha256=_sha256_file(archive),
    )


def unpack_dissertation_image_archive(
    *,
    archive_path: str | Path,
    root_dir: str | Path,
) -> DissertationImageArchiveResult:
    """Unpack a Git LFS image archive into the private data root safely."""

    archive = Path(archive_path)
    root = Path(root_dir).resolve()
    if not archive.exists():
        raise DissertationDatasetError(f"Archive does not exist: {archive}")
    image_count = 0
    total_bytes = 0
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            member_name = member.filename.replace("\\", "/")
            if _is_private_or_absolute_path(member_name) or ".." in Path(member_name).parts:
                raise DissertationDatasetError(f"Archive member path is unsafe: {member.filename}")
            target = (root / member_name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise DissertationDatasetError(f"Archive member resolves outside root: {member.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            data = handle.read(member)
            target.write_bytes(data)
            total_bytes += len(data)
            if target.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_count += 1
    return DissertationImageArchiveResult(
        archive_path=archive,
        manifest_path=archive.with_suffix(".manifest.csv"),
        image_count=image_count,
        total_bytes=total_bytes,
        sha256=_sha256_file(archive),
    )


def verify_dissertation_image_files(
    *,
    image_dir: str | Path,
    root_dir: str | Path,
) -> DissertationImageArchiveResult:
    """Verify direct individual image files exist for the primary LFS workflow."""

    source_dir = Path(image_dir)
    root = Path(root_dir).resolve()
    if not source_dir.exists():
        raise DissertationDatasetError(f"Image directory does not exist: {source_dir}")
    files = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not files:
        raise DissertationDatasetError(f"Image directory contains no supported images: {source_dir}")
    total_bytes = 0
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise DissertationDatasetError(f"Image file is outside root_dir: {path}") from exc
        if _is_private_or_absolute_path(relative):
            raise DissertationDatasetError(f"Image path is unsafe: {relative}")
        total_bytes += path.stat().st_size
    return DissertationImageArchiveResult(
        archive_path=source_dir,
        manifest_path=source_dir,
        image_count=len(files),
        total_bytes=total_bytes,
        sha256="",
    )


def write_dissertation_checksums(
    *,
    dataset_dir: str | Path,
    root_dir: str | Path,
    image_dir: str | Path | None = None,
) -> DissertationChecksumResult:
    """Write SHA256 checksums for package metadata, LFS archive, and local images."""

    package_dir = Path(dataset_dir)
    root = Path(root_dir).resolve()
    checksum_path = package_dir / "checksums.sha256"
    entries: list[tuple[str, str, int]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(package_dir).as_posix()
        if _is_private_or_absolute_path(relative):
            raise DissertationDatasetError(f"Checksum path is unsafe: {relative}")
        entries.append((_sha256_file(path), relative, path.stat().st_size))

    if image_dir is not None:
        image_root = Path(image_dir)
        for path in sorted(image_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise DissertationDatasetError(f"Checksum image is outside root_dir: {path}") from exc
            if _is_private_or_absolute_path(relative):
                raise DissertationDatasetError(f"Checksum image path is unsafe: {relative}")
            entries.append((_sha256_file(path), relative, path.stat().st_size))

    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(
        "".join(f"{sha}  {relative}\n" for sha, relative, _ in entries),
        encoding="utf-8",
    )
    return DissertationChecksumResult(
        checksum_path=checksum_path,
        checked_files=len(entries),
        total_bytes=sum(size for _, _, size in entries),
    )


def verify_dissertation_checksums(
    *,
    dataset_dir: str | Path,
    root_dir: str | Path,
    checksum_path: str | Path | None = None,
) -> DissertationChecksumResult:
    """Verify package and unpacked image checksums without reading private paths."""

    package_dir = Path(dataset_dir)
    root = Path(root_dir)
    checksums = Path(checksum_path) if checksum_path is not None else package_dir / "checksums.sha256"
    if not checksums.exists():
        raise DissertationDatasetError(f"Checksum file does not exist: {checksums}")
    checked = 0
    total_bytes = 0
    errors: list[str] = []
    for line_idx, raw_line in enumerate(checksums.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError:
            errors.append(f"{checksums}:{line_idx}: malformed checksum line")
            continue
        relative = relative.strip()
        if _is_private_or_absolute_path(relative) or ".." in Path(relative).parts:
            errors.append(f"{checksums}:{line_idx}: unsafe checksum path {relative}")
            continue
        target = (root / relative) if relative.startswith("images/") else (package_dir / relative)
        if not target.exists():
            errors.append(f"{checksums}:{line_idx}: missing file {relative}")
            continue
        actual = _sha256_file(target)
        if actual != expected:
            errors.append(f"{checksums}:{line_idx}: checksum mismatch for {relative}")
            continue
        checked += 1
        total_bytes += target.stat().st_size
    if errors:
        raise DissertationDatasetError("; ".join(errors))
    return DissertationChecksumResult(
        checksum_path=checksums,
        checked_files=checked,
        total_bytes=total_bytes,
    )


def validate_dissertation_manifest_frame(
    df: pd.DataFrame,
    *,
    manifest_path: str | Path,
    official_package: bool = True,
) -> None:
    """Validate a manifest frame before it enters the commit-safe package."""

    path = Path(manifest_path)
    missing = [column for column in ["image_path", "label", "split", "source", "ood_type"] if column not in df]
    if missing:
        raise DissertationDatasetError(f"Manifest {path} missing required columns: {missing}")
    banned = sorted(BANNED_PRIVATE_COLUMNS.intersection(df.columns))
    if banned:
        raise DissertationDatasetError(
            f"Manifest {path} contains banned clinical/private column(s): {banned}"
        )
    if df.empty:
        raise DissertationDatasetError(f"Manifest {path} contains no rows")

    frame = df.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise DissertationDatasetError(f"Manifest {path} has invalid label values")
    frame["label"] = labels.astype(int)

    invalid_splits = ~frame["split"].isin({"train", "val", "test"})
    if invalid_splits.any():
        raise DissertationDatasetError(f"Manifest {path} has invalid split values")
    invalid_ood = ~frame["ood_type"].isin(VALID_OOD_TYPES)
    if invalid_ood.any():
        raise DissertationDatasetError(f"Manifest {path} has invalid ood_type values")

    for image_path in frame["image_path"].astype(str):
        if _is_private_or_absolute_path(image_path):
            raise DissertationDatasetError(
                f"Manifest {path} contains private/absolute image_path: {image_path}"
            )

    if "patient_id" in frame.columns:
        non_empty_patient = frame["patient_id"].fillna("").astype(str).str.strip().ne("")
        if non_empty_patient.any():
            raise DissertationDatasetError(f"Manifest {path} contains non-empty patient_id")

    train_bad = (frame["split"] == "train") & (
        (frame["label"] != 0) | (frame["ood_type"] != "id")
    )
    if train_bad.any():
        raise DissertationDatasetError("train split must contain label=0 and ood_type=id only")

    id_bad = (frame["label"] == 0) & (frame["ood_type"] != "id")
    ood_bad = (frame["label"] == 1) & (frame["ood_type"] == "id")
    if id_bad.any() or ood_bad.any():
        raise DissertationDatasetError(f"Manifest {path} has label/ood_type mismatch")

    if "source_split" in frame.columns:
        artifact_from_train = (
            (frame["ood_type"] == "sensory_artifact")
            & (frame["source_split"].str.lower() == "train")
        )
        if artifact_from_train.any():
            raise DissertationDatasetError(
                f"Manifest {path}: sensory_artifact rows must not use source_split=train"
            )

    if official_package and "ood_subtype" in frame.columns:
        official_ood = frame["ood_type"] != "id"
        missing_subtype = frame["ood_subtype"].fillna("").astype(str).str.strip().eq("")
        if (official_ood & missing_subtype).any():
            raise DissertationDatasetError(
                f"Manifest {path}: official OOD rows require non-empty ood_subtype"
            )
    elif official_package and (frame["ood_type"] != "id").any():
        raise DissertationDatasetError(f"Manifest {path}: official OOD rows require ood_subtype")


def _read_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DissertationDatasetError(f"Dataset config does not exist: {path}") from exc
    if not isinstance(loaded, dict):
        raise DissertationDatasetError(f"Dataset config must be a YAML mapping: {path}")
    return loaded


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("dataset_contract_id") != EXPECTED_DATASET_CONTRACT_ID:
        raise DissertationDatasetError(
            "dataset_contract_id must be dissertation_faf_ood_v1 for this package"
        )
    if config.get("task_type") != "unsupervised_ood_gatekeeper":
        raise DissertationDatasetError("task_type must be unsupervised_ood_gatekeeper")
    if config.get("target_modality") != "retinal_faf":
        raise DissertationDatasetError("target_modality must be retinal_faf")
    if config.get("not_a_disease_classifier") is not True:
        raise DissertationDatasetError("not_a_disease_classifier must be true")
    training_policy = config.get("training_policy", {})
    if not isinstance(training_policy, dict) or training_policy.get("id_only") is not True:
        raise DissertationDatasetError("training_policy.id_only must be true")
    if config.get("synthetic_id_fallback_warning") is not True:
        raise DissertationDatasetError("synthetic_id_fallback_warning must be true")


def _prepare_id_rows(
    *,
    source_dir: Path,
    out_image_dir: Path,
    root_dir: Path,
    seed: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_images, stratified = _discover_syntheye_images(source_dir)
    if not stratified:
        warnings.append(
            "SynthEye source classes were unavailable; ID split is deterministic but not stratified"
        )
    warnings.append(SYNTHETIC_FALLBACK_NOTE)

    rng = np.random.default_rng(seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    counters = {"train": 0, "val": 0, "test": 0}
    for class_name, paths in sorted(class_images.items()):
        shuffled = list(rng.permutation(paths))
        split_paths = _split_paths(shuffled)
        for split_name, source_paths in split_paths.items():
            for source_path in source_paths:
                output_split_dir = "test_synthetic_fallback" if split_name == "test" else split_name
                output_path = (
                    out_image_dir
                    / "id"
                    / output_split_dir
                    / f"id_{output_split_dir}_{counters[split_name]:06d}.png"
                )
                _save_as_png(source_path, output_path)
                source_hash = _sha256_file(source_path)
                notes = f"syntheye_class={class_name}; synthetic_id_faf"
                if split_name == "test":
                    notes = f"{notes}; {SYNTHETIC_FALLBACK_NOTE}"
                rows_by_split[split_name].append(
                    _row(
                        image_path=_manifest_image_path(output_path, root_dir),
                        label=0,
                        split=split_name,
                        source="synthetic_faf",
                        ood_type="id",
                        ood_subtype="id",
                        source_dataset="UCL SynthEye synthetic FAF",
                        source_split=split_name,
                        source_image_hash=source_hash,
                        scanner="synthetic_heidelberg",
                        license_status="pending_manual_review",
                        notes=notes,
                    )
                )
                counters[split_name] += 1
    return rows_by_split["train"], rows_by_split["val"], rows_by_split["test"]


def _discover_syntheye_images(source_dir: Path) -> tuple[dict[str, list[Path]], bool]:
    if not source_dir.exists():
        raise DissertationDatasetError(f"SynthEye source directory does not exist: {source_dir}")
    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if class_dirs:
        class_images = {
            class_dir.name: _discover_images(class_dir)
            for class_dir in class_dirs
        }
        empty = [class_name for class_name, images in class_images.items() if not images]
        if empty:
            raise DissertationDatasetError(f"SynthEye class folders contain no images: {empty}")
        return class_images, True
    images = _discover_images(source_dir)
    if not images:
        raise DissertationDatasetError(f"SynthEye source directory contains no images: {source_dir}")
    return {"unavailable": images}, False


def _prepare_artifact_rows(
    *,
    id_rows: list[dict[str, Any]],
    out_image_dir: Path,
    root_dir: Path,
    rng: np.random.Generator,
    allowed_source_splits: set[str],
) -> list[dict[str, Any]]:
    if not id_rows:
        raise DissertationDatasetError("No held-out ID rows are available for artifact generation")
    rows: list[dict[str, Any]] = []
    for source_idx, parent_row in enumerate(id_rows):
        source_split = str(parent_row["split"])
        if source_split not in allowed_source_splits:
            raise DissertationDatasetError(
                f"Artifact source split {source_split} is not allowed; expected {allowed_source_splits}"
            )
        parent_path = root_dir / str(parent_row["image_path"])
        with Image.open(parent_path) as image:
            parent_image = image.convert("RGB")
        for artifact_idx, artifact_type in enumerate(ARTIFACT_SUBTYPES):
            corrupted = apply_artifact(parent_image, artifact_type, rng=rng)
            output_path = (
                out_image_dir
                / "ood"
                / "sensory_artifact"
                / artifact_type
                / f"artifact_{source_idx:06d}_{artifact_idx:02d}_{artifact_type}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            corrupted.save(output_path, format="PNG")
            rows.append(
                _row(
                    image_path=_manifest_image_path(output_path, root_dir),
                    label=1,
                    split="test",
                    source="synthetic_artifact",
                    ood_type="sensory_artifact",
                    ood_subtype=artifact_type,
                    source_dataset="derived_from_synthetic_faf",
                    source_split=source_split,
                    source_image_hash=_sha256_file(output_path),
                    parent_image_hash=str(parent_row["source_image_hash"]),
                    synthetic_transform=artifact_type,
                    scanner="synthetic_heidelberg",
                    license_status="derived_or_generated_synthetic",
                    notes="generated from held-out ID only; never from train",
                )
            )
    return rows


def _prepare_modality_rows(
    *,
    colour_dir: Path | None,
    oct_dir: Path | None,
    out_image_dir: Path,
    root_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
    allow_synthetic_surrogates: bool,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_specs = [
        (colour_dir, "colour_fundus", "prepared_colour_fundus"),
        (oct_dir, "oct_screenshot", "prepared_oct_screenshot"),
    ]
    for directory, subtype, source_dataset in source_specs:
        images = _discover_images(directory) if directory is not None and directory.exists() else []
        if directory is not None and directory.exists() and not images:
            raise DissertationDatasetError(f"Prepared {subtype} directory contains no images: {directory}")
        if images:
            warnings.append(f"{subtype} license_status=pending_manual_review")
            rows.extend(
                _copy_ood_images(
                    images=images,
                    out_image_dir=out_image_dir,
                    root_dir=root_dir,
                    ood_type="modality_shift",
                    ood_subtype=subtype,
                    source="public_modality_shift",
                    source_dataset=source_dataset,
                    license_status="pending_manual_review",
                    semantic_class="",
                )
            )
            continue
        if allow_synthetic_surrogates:
            count = int(_config_targets(config).get("surrogate_modality_per_subtype", 200))
            warnings.append(f"{subtype}: using synthetic surrogate rows, not real modality_shift data")
            rows.extend(
                _generate_surrogate_rows(
                    count=count,
                    out_image_dir=out_image_dir,
                    root_dir=root_dir,
                    rng=rng,
                    ood_type="modality_shift",
                    ood_subtype=subtype,
                    source="synthetic_surrogate",
                    semantic_class="",
                )
            )
    return rows


def _prepare_semantic_rows(
    *,
    source_dir: Path | None,
    out_image_dir: Path,
    root_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
    allow_synthetic_surrogates: bool,
    warnings: list[str],
) -> list[dict[str, Any]]:
    images = _discover_images(source_dir) if source_dir is not None and source_dir.exists() else []
    if source_dir is not None and source_dir.exists() and not images:
        raise DissertationDatasetError(f"Prepared semantic directory contains no images: {source_dir}")
    if images:
        warnings.append("cifar10_natural license_status=pending_manual_review")
        return _copy_ood_images(
            images=images,
            out_image_dir=out_image_dir,
            root_dir=root_dir,
            ood_type="semantic_outlier",
            ood_subtype=SEMANTIC_SUBTYPE,
            source="public_semantic_outlier",
            source_dataset="prepared_cifar10_or_natural",
            license_status="pending_manual_review",
            semantic_class="from_parent",
        )
    if allow_synthetic_surrogates:
        count = int(_config_targets(config).get("surrogate_semantic", 500))
        warnings.append("semantic_outlier: using synthetic surrogate rows, not real natural images")
        return _generate_surrogate_rows(
            count=count,
            out_image_dir=out_image_dir,
            root_dir=root_dir,
            rng=rng,
            ood_type="semantic_outlier",
            ood_subtype=SEMANTIC_SUBTYPE,
            source="synthetic_surrogate",
            semantic_class="synthetic_surrogate",
        )
    return []


def _copy_ood_images(
    *,
    images: list[Path],
    out_image_dir: Path,
    root_dir: Path,
    ood_type: str,
    ood_subtype: str,
    source: str,
    source_dataset: str,
    license_status: str,
    semantic_class: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, source_path in enumerate(images):
        output_path = (
            out_image_dir
            / "ood"
            / ood_type
            / ood_subtype
            / f"{ood_subtype}_{idx:06d}.png"
        )
        _save_as_png(source_path, output_path)
        resolved_semantic_class = source_path.parent.name if semantic_class == "from_parent" else semantic_class
        rows.append(
            _row(
                image_path=_manifest_image_path(output_path, root_dir),
                label=1,
                split="test",
                source=source,
                ood_type=ood_type,
                ood_subtype=ood_subtype,
                source_dataset=source_dataset,
                source_split="test",
                source_image_hash=_sha256_file(source_path),
                semantic_class=resolved_semantic_class,
                license_status=license_status,
                notes="public/prepared OOD source; verify license before dissertation submission",
            )
        )
    return rows


def _generate_surrogate_rows(
    *,
    count: int,
    out_image_dir: Path,
    root_dir: Path,
    rng: np.random.Generator,
    ood_type: str,
    ood_subtype: str,
    source: str,
    semantic_class: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        output_path = (
            out_image_dir
            / "ood"
            / ood_type
            / ood_subtype
            / f"synthetic_surrogate_{ood_subtype}_{idx:06d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = _surrogate_image(ood_type=ood_type, ood_subtype=ood_subtype, rng=rng)
        image.save(output_path, format="PNG")
        rows.append(
            _row(
                image_path=_manifest_image_path(output_path, root_dir),
                label=1,
                split="test",
                source=source,
                ood_type=ood_type,
                ood_subtype=ood_subtype,
                source_dataset="synthetic_surrogate",
                source_split="test",
                source_image_hash=_sha256_file(output_path),
                semantic_class=semantic_class,
                is_synthetic_surrogate="true",
                license_status="derived_or_generated_synthetic",
                notes="synthetic surrogate for pipeline testing; do not report as real public OOD",
            )
        )
    return rows


def _surrogate_image(*, ood_type: str, ood_subtype: str, rng: np.random.Generator) -> Image.Image:
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    if ood_type == "modality_shift" and ood_subtype == "oct_screenshot":
        image = image.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")
    if ood_type == "modality_shift" and ood_subtype == "colour_fundus":
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 56, 56), outline=(220, 80, 50), width=4)
        draw.line((32, 12, 32, 52), fill=(240, 230, 120), width=2)
    return image


def _balanced_by_group(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DissertationDatasetError("Cannot build balanced OOD manifest from empty rows")
    counts = frame[group_field].value_counts()
    n = int(counts.min())
    sampled = [
        group.sort_values(["ood_subtype", "image_path"]).sample(n=n, random_state=seed)
        for _, group in sorted(frame.groupby(group_field), key=lambda item: str(item[0]))
    ]
    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values([group_field, "ood_subtype", "image_path"], kind="stable")
        .to_dict(orient="records")
    )


def _smoke_by_subtype(
    rows: list[dict[str, Any]],
    *,
    per_subtype: int,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    sampled = [
        group.sort_values("image_path").sample(n=min(per_subtype, len(group)), random_state=seed)
        for _, group in sorted(frame.groupby("ood_subtype"), key=lambda item: str(item[0]))
    ]
    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["ood_subtype", "image_path"], kind="stable")
        .to_dict(orient="records")
    )


def _validate_manifest_collection(manifest_rows: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in manifest_rows.items():
        frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        validate_dissertation_manifest_frame(
            frame,
            manifest_path=Path(f"{name}.csv"),
            official_package=True,
        )


def _write_manifests(rows_by_name: dict[str, list[dict[str, Any]]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, rows in rows_by_name.items():
        path = out_dir / f"{name}.csv"
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(path, index=False)
        outputs[name] = path
    return outputs


def _build_audit_summary(
    manifest_rows: dict[str, list[dict[str, Any]]],
    *,
    warnings: list[str],
    seed: int,
) -> dict[str, Any]:
    counts = {}
    for name, rows in manifest_rows.items():
        frame = pd.DataFrame(rows)
        counts[name] = {
            "rows": int(len(frame)),
            "label": _counts(frame, "label"),
            "split": _counts(frame, "split"),
            "ood_type": _counts(frame, "ood_type"),
            "ood_subtype": _counts(frame, "ood_subtype"),
        }
    return {
        "dataset_contract_id": EXPECTED_DATASET_CONTRACT_ID,
        "seed": seed,
        "not_a_disease_classifier": True,
        "image_files_committed": False,
        "counts": counts,
        "warnings": sorted(set(warnings)),
    }


def _write_repo_package_docs(
    repo_dir: Path,
    *,
    manifest_rows: dict[str, list[dict[str, Any]]],
    audit_summary: dict[str, Any],
    warnings: list[str],
    individual_lfs_images: bool = False,
) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    counts = audit_summary["counts"]
    statement = "This dataset package contains manifests and metadata only; image files are not committed."
    if individual_lfs_images:
        statement = (
            "This repository includes the actual final dissertation dataset images "
            "under `data/images/dissertation_v1/` through Git LFS."
        )
        lfs_statement = (
            "Individual PNG/JPG/TIFF files are tracked by Git LFS and must not be "
            "stored as normal Git blobs."
        )
    else:
        lfs_statement = (
            "Individual PNG/JPG/TIFF files are not committed as normal Git blobs; "
            "the curated image package is stored as a Git LFS archive."
        )
    README = f"""# Dissertation Dataset v1

This package defines the dissertation dataset for the retinal FAF OOD gatekeeper.
It is not a disease-classification dataset.

{statement}
{lfs_statement}

Use `scripts/prepare_dissertation_dataset.py` to rebuild local private images and
manifests from your prepared data folders. The committed CSV files contain only
relative paths, labels, taxonomy fields, hashes, and safe provenance notes.

The canonical image files are under `data/images/dissertation_v1/`.

## Manifests

- `manifests/train_id.csv`: ID-only synthetic FAF training rows.
- `manifests/val_id.csv`: ID-only synthetic FAF validation rows.
- `manifests/test_id_synthetic_fallback.csv`: synthetic fallback ID test rows,
  not real clinical FAF validation.
- `manifests/test_artifact.csv`: generated sensory artifact OOD rows.
- `manifests/test_modality.csv`: wrong-modality OOD rows.
- `manifests/test_semantic.csv`: semantic outlier OOD rows.
- `manifests/test_ood*.csv`: dissertation-named artifact/modality/semantic,
  full, balanced, and smoke OOD evaluation subsets.

## School Server

```bash
git lfs pull
python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv
python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/val_id.csv --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt --fail-on-duplicate-content-across-splits
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
```
"""
    DATASET_CARD = f"""# Dataset Card: Dissertation FAF OOD v1

## Task

Unsupervised OOD gatekeeper for retinal FAF image quality control. Training uses
valid FAF only (`label=0`, `ood_type=id`). The system must reject invalid/OOD
inputs upstream of any diagnostic model.

## Important Warning

`test_id_synthetic_fallback.csv` is synthetic fallback data only. It is not real
clinical FAF validation and should be described as a proof-of-concept limitation.

## Counts

```json
{json.dumps(counts, indent=2)}
```

## Privacy

{statement}
{lfs_statement}
Patient identifiers and clinical disease labels are not included.
"""
    SOURCES = """# Sources

## ID FAF

- UCL SynthEye synthetic FAF dataset, used as proof-of-concept ID FAF.
- SynthEye class folder names are used only for deterministic split/audit notes,
  not as disease labels.

## OOD

- Sensory artifacts are generated from held-out ID rows only.
- Colour fundus, OCT screenshot, and natural-image sources must be prepared
  locally and license-reviewed before dissertation publication.
- Rows marked `is_synthetic_surrogate=true` are pipeline placeholders only and
  must not be reported as real public OOD evidence.

## School Server Reproducibility

```bash
git lfs pull
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/val_id.csv datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv datasets/dissertation_v1/manifests/test_ood.csv
python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/val_id.csv --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt --fail-on-duplicate-content-across-splits
```
"""
    warning_lines = "\n".join(f"- {warning}" for warning in sorted(set(warnings))) or "- None"
    AUDIT = f"""# Audit Summary

{statement}
{lfs_statement}

## Safe Aggregate Counts

```json
{json.dumps(counts, indent=2)}
```

## Warnings

{warning_lines}

## Archive

- Git LFS image root: `data/images/dissertation_v1/`
- Checksum file: `checksums.sha256`
- Primary workflow: `git clone`, `git lfs pull`, then validate/audit manifests.
"""
    files = {
        "README.md": README,
        "DATASET_CARD.md": DATASET_CARD,
        "SOURCES.md": SOURCES,
        "AUDIT_SUMMARY.md": AUDIT,
    }
    for filename, text in files.items():
        (repo_dir / filename).write_text(text, encoding="utf-8")

    manifest_total = sum(len(rows) for rows in manifest_rows.values())
    if manifest_total <= 0:
        raise DissertationDatasetError("No manifest rows were written to the package")


def _discover_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _reset_direct_image_output_dir(out_image_dir: Path, root_dir: Path) -> None:
    resolved = out_image_dir.resolve()
    root = root_dir.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise DissertationDatasetError(
            f"Refusing to clean image output outside root_dir: {out_image_dir}"
        ) from exc
    if relative.as_posix() != "images/dissertation_v1":
        raise DissertationDatasetError(
            "Direct LFS image mode only cleans root_dir/images/dissertation_v1; "
            f"got {relative.as_posix()}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _split_paths(paths: list[Path]) -> dict[str, list[Path]]:
    n_items = len(paths)
    n_train = int(round(n_items * 0.70))
    n_val = int(round(n_items * 0.15))
    if n_train + n_val > n_items:
        n_val = max(0, n_items - n_train)
    return {
        "train": paths[:n_train],
        "val": paths[n_train : n_train + n_val],
        "test": paths[n_train + n_val :],
    }


def _row(**values: Any) -> dict[str, Any]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(values)
    row.setdefault("patient_id", "")
    row.setdefault("is_synthetic_surrogate", "false")
    return row


def _save_as_png(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.save(output_path, format="PNG")


def _manifest_image_path(path: Path, root_dir: Path) -> str:
    resolved = path.resolve()
    root = root_dir.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        raise DissertationDatasetError(f"Output image path is outside root_dir: {path}") from None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fail_duplicate_content_across_splits(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(str(row["source_image_hash"]), set()).add(str(row["split"]))
    duplicate = {sha: splits for sha, splits in seen.items() if len(splits) > 1}
    if duplicate:
        raise DissertationDatasetError(
            f"duplicate content across splits found for {len(duplicate)} SHA256 group(s)"
        )


def _config_targets(config: dict[str, Any]) -> dict[str, Any]:
    targets = config.get("targets", {})
    return targets if isinstance(targets, dict) else {}


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df:
        return {}
    return {
        str(key): int(value)
        for key, value in Counter(str(value) for value in df[column].fillna("").tolist()).items()
    }


def _is_private_or_absolute_path(image_path: str) -> bool:
    if PRIVATE_PATH_RE.search(image_path):
        return True
    if Path(image_path).is_absolute():
        return True
    return PureWindowsPath(image_path).is_absolute()

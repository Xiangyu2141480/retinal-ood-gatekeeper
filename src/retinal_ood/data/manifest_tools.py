"""Utilities for building public OOD manifests without committing image data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

OUTPUT_COLUMNS = ["image_path", "label", "split", "source", "ood_type", "patient_id", "scanner", "notes"]
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OODType = Literal["modality_shift", "sensory_artifact", "semantic_outlier"]
VALID_OOD_TYPES = {"modality_shift", "sensory_artifact", "semantic_outlier"}
OOD_TAXONOMY_NOTE = (
    "Allowed OOD taxonomy: modality_shift for wrong imaging modality, "
    "sensory_artifact for text/watermark/annotation/composite/corruption artifacts, "
    "semantic_outlier for non-retinal or unrelated images."
)


def build_ood_manifest(
    *,
    root_dir: str | Path,
    mappings: list[tuple[str, str]],
    out_manifest: str | Path,
    split: str = "test",
    source: str = "public_ood",
) -> pd.DataFrame:
    """Build an OOD manifest from folder-to-literature-category mappings."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    if not mappings:
        raise ValueError("At least one folder mapping is required")

    root_path = Path(root_dir)
    rows: list[dict[str, str | int]] = []
    for relative_folder, ood_type in mappings:
        if ood_type not in VALID_OOD_TYPES:
            raise ValueError(f"Invalid ood_type {ood_type!r}. {OOD_TAXONOMY_NOTE}")
        folder = root_path / relative_folder
        images = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported image files found for mapping folder: {folder}")
        for image_path in images:
            rows.append(
                {
                    "image_path": image_path.resolve().relative_to(root_path.resolve()).as_posix(),
                    "label": 1,
                    "split": split,
                    "source": source,
                    "ood_type": ood_type,
                    "patient_id": "",
                    "scanner": "",
                    "notes": f"folder={Path(relative_folder).as_posix()}",
                }
            )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_path = Path(out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def merge_manifest_files(manifest_paths: list[str | Path], out_manifest: str | Path) -> pd.DataFrame:
    """Merge manifest CSVs after validating that all required output columns exist."""
    if not manifest_paths:
        raise ValueError("At least one manifest path is required")

    frames: list[pd.DataFrame] = []
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        df = pd.read_csv(path)
        missing = set(OUTPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Manifest {path} missing required columns: {sorted(missing)}")
        frames.append(df[OUTPUT_COLUMNS])

    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise ValueError("Merged manifest would contain no rows")
    out_path = Path(out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


def parse_mapping(raw_mapping: str) -> tuple[str, str]:
    """Parse ``relative/folder=ood_type`` CLI mapping strings."""
    if "=" not in raw_mapping:
        raise ValueError(f"Mapping must use relative/folder=ood_type format: {raw_mapping}")
    folder, ood_type = raw_mapping.split("=", 1)
    folder = folder.strip().replace("\\", "/")
    ood_type = ood_type.strip()
    if not folder or not ood_type:
        raise ValueError(f"Mapping must include both folder and ood_type: {raw_mapping}")
    return folder, ood_type

"""Manifest-based image dataset.

The project uses CSV manifests so private data can stay outside GitHub. A manifest records
where each image lives and whether it is ID or OOD for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pandas.errors import EmptyDataError
from PIL import Image

REQUIRED_COLUMNS = {"image_path", "split", "label", "category"}
OPTIONAL_COLUMNS = {"patient_id", "scanner", "notes"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    label: int
    split: str
    category: str
    metadata: dict[str, Any]


class ManifestImageDataset:
    """Dataset backed by a CSV manifest.

    Parameters
    ----------
    manifest_path:
        CSV containing at least image_path,split,label,category. Optional metadata columns
        include patient_id, scanner, and notes.
    root_dir:
        Optional base directory for relative image paths.
    transform:
        Optional callable applied to PIL images.
    require_files:
        If true, validate every image exists during initialization.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        require_files: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {self.manifest_path}")

        try:
            self.df = pd.read_csv(self.manifest_path)
        except EmptyDataError as exc:
            raise ValueError(f"Manifest {self.manifest_path} is empty or malformed") from exc

        self._validate_schema()
        self._validate_rows()

        if require_files:
            missing_files = [str(p) for p in self._resolved_paths() if not p.exists()]
            if missing_files:
                preview = missing_files[:5]
                raise FileNotFoundError(
                    f"Manifest {self.manifest_path} references {len(missing_files)} missing "
                    f"image files, examples: {preview}"
                )

    def _validate_schema(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} missing required columns: {sorted(missing)}")
        if self.df.empty:
            raise ValueError(f"Manifest {self.manifest_path} contains no rows")
        for column in OPTIONAL_COLUMNS:
            if column not in self.df.columns:
                self.df[column] = ""

    def _validate_rows(self) -> None:
        self._validate_image_paths()
        self._validate_splits()
        self._validate_labels()
        self._validate_categories()

    def _validate_image_paths(self) -> None:
        paths = self.df["image_path"].astype("string")
        invalid_mask = paths.isna() | (paths.str.strip() == "")
        if invalid_mask.any():
            rows = self._row_numbers(invalid_mask)
            raise ValueError(f"Manifest {self.manifest_path} has empty image_path values at rows: {rows}")

    def _validate_splits(self) -> None:
        splits = self.df["split"].astype("string").str.strip().str.lower()
        invalid_mask = splits.isna() | ~splits.isin(VALID_SPLITS)
        if invalid_mask.any():
            invalid_values = sorted(set(self.df.loc[invalid_mask, "split"].astype(str).tolist()))
            raise ValueError(
                f"Manifest {self.manifest_path} has invalid split values {invalid_values}; "
                f"expected one of {sorted(VALID_SPLITS)}"
            )
        self.df["split"] = splits

    def _validate_labels(self) -> None:
        labels = pd.to_numeric(self.df["label"], errors="coerce")
        invalid_mask = labels.isna() | ~labels.isin(VALID_LABELS)
        if invalid_mask.any():
            invalid_values = sorted(set(self.df.loc[invalid_mask, "label"].astype(str).tolist()))
            raise ValueError(
                f"Manifest {self.manifest_path} has invalid label values {invalid_values}; "
                "expected binary labels 0=ID and 1=OOD"
            )
        self.df["label"] = labels.astype(int)

    def _validate_categories(self) -> None:
        categories = self.df["category"].astype("string")
        invalid_mask = categories.isna() | (categories.str.strip() == "")
        if invalid_mask.any():
            rows = self._row_numbers(invalid_mask)
            raise ValueError(f"Manifest {self.manifest_path} has empty category values at rows: {rows}")
        self.df["category"] = categories.str.strip()

    @staticmethod
    def _row_numbers(mask: pd.Series) -> list[int]:
        """Return 1-based CSV data-row numbers for user-facing error messages."""
        return [int(idx) + 2 for idx in mask[mask].index.tolist()]

    def _resolve_path(self, image_path: str | Path) -> Path:
        p = Path(image_path)
        if p.is_absolute() or self.root_dir is None:
            return p
        return self.root_dir / p

    def _resolved_paths(self) -> list[Path]:
        return [self._resolve_path(p) for p in self.df["image_path"].tolist()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, int, dict[str, Any]]:
        row = self.df.iloc[idx]
        path = self._resolve_path(row["image_path"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        metadata = row.to_dict()
        metadata["resolved_image_path"] = str(path)
        return image, label, metadata

    def id_subset(self) -> "ManifestImageDataset":
        """Return a shallow dataset containing only ID/normal samples."""
        clone = object.__new__(ManifestImageDataset)
        clone.manifest_path = self.manifest_path
        clone.root_dir = self.root_dir
        clone.transform = self.transform
        clone.df = self.df[self.df["label"] == 0].reset_index(drop=True)
        return clone

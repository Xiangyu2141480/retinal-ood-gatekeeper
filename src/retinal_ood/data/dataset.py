"""Manifest-based image dataset.

The project uses CSV manifests so private data can stay outside GitHub. A manifest records
where each image lives and whether it is ID or OOD for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

REQUIRED_COLUMNS = {"image_path", "label", "split", "source", "ood_type"}


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    label: int
    split: str
    source: str
    ood_type: str
    metadata: dict[str, Any]


class ManifestImageDataset:
    """Dataset backed by a CSV manifest.

    Parameters
    ----------
    manifest_path:
        CSV containing at least image_path,label,split,source,ood_type.
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
        self.df = pd.read_csv(self.manifest_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} missing required columns: {sorted(missing)}")
        if require_files:
            missing_files = [str(p) for p in self._resolved_paths() if not p.exists()]
            if missing_files:
                preview = missing_files[:5]
                raise FileNotFoundError(f"Missing {len(missing_files)} image files, examples: {preview}")

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

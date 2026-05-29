"""Prepare SynthEye FAF images as in-distribution manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

OUTPUT_COLUMNS = ["image_path", "label", "split", "source", "ood_type", "patient_id", "scanner", "notes"]
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SyntheyePreparationResult:
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    total_images: int
    class_counts: dict[str, int]


def prepare_syntheye_dataset(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    manifest_dir: str | Path,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SyntheyePreparationResult:
    """Copy SynthEye class folders into anonymized ID images and split manifests.

    SynthEye folder names are recorded only in notes for stratification/audit. They are not
    disease labels for this project.
    """
    input_path = Path(input_dir)
    output_path = Path(out_dir)
    manifests_path = Path(manifest_dir)
    _validate_fractions(train_fraction, val_fraction)
    class_to_images = _discover_class_images(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    manifests_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest_root = _infer_manifest_root(output_path)
    split_rows: dict[SplitName, list[dict[str, str | int]]] = {"train": [], "val": [], "test": []}
    image_counter = 0

    for class_name, image_paths in class_to_images.items():
        shuffled = list(rng.permutation(image_paths))
        split_assignments = _split_class_images(shuffled, train_fraction=train_fraction, val_fraction=val_fraction)
        for split_name in ("train", "val", "test"):
            for source_path in split_assignments[split_name]:
                output_file = output_path / f"synthetic_faf_{image_counter:06d}.png"
                _copy_as_png(source_path, output_file)
                split_rows[split_name].append(
                    {
                        "image_path": _manifest_image_path(output_file, manifest_root),
                        "label": 0,
                        "split": split_name,
                        "source": "synthetic_faf",
                        "ood_type": "id",
                        "patient_id": "",
                        "scanner": "synthetic_heidelberg",
                        "notes": f"syntheye_class={class_name}; synthetic_id_faf",
                    }
                )
                image_counter += 1

    manifest_paths = {
        "train": manifests_path / "train_synthetic_faf.csv",
        "val": manifests_path / "val_synthetic_faf.csv",
        "test": manifests_path / "test_real_id.csv",
    }
    for split_name, rows in split_rows.items():
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(manifest_paths[split_name], index=False)

    return SyntheyePreparationResult(
        train_manifest=manifest_paths["train"],
        val_manifest=manifest_paths["val"],
        test_manifest=manifest_paths["test"],
        total_images=sum(len(paths) for paths in class_to_images.values()),
        class_counts={class_name: len(paths) for class_name, paths in class_to_images.items()},
    )


def _discover_class_images(input_dir: Path) -> dict[str, list[Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"SynthEye input directory does not exist: {input_dir}")
    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"SynthEye input directory has no class folders: {input_dir}")

    class_to_images: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"SynthEye class folder has no supported images: {class_dir}")
        class_to_images[class_dir.name] = images
    return class_to_images


def _split_class_images(
    image_paths: list[Path],
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[SplitName, list[Path]]:
    n_images = len(image_paths)
    n_train = int(round(n_images * train_fraction))
    n_val = int(round(n_images * val_fraction))
    if n_train + n_val > n_images:
        n_val = max(0, n_images - n_train)
    return {
        "train": image_paths[:n_train],
        "val": image_paths[n_train : n_train + n_val],
        "test": image_paths[n_train + n_val :],
    }


def _copy_as_png(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as image:
        image.save(output_path, format="PNG")


def _infer_manifest_root(out_dir: Path) -> Path | None:
    resolved = out_dir.resolve()
    if resolved.parent.name == "images":
        return resolved.parent.parent
    return None


def _manifest_image_path(path: Path, root_dir: Path | None) -> str:
    resolved = path.resolve()
    if root_dir is not None:
        try:
            return resolved.relative_to(root_dir).as_posix()
        except ValueError:
            pass
    return str(path)


def _validate_fractions(train_fraction: float, val_fraction: float) -> None:
    if train_fraction <= 0 or val_fraction < 0:
        raise ValueError("train_fraction must be positive and val_fraction must be non-negative")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")

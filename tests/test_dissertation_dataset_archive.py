from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from retinal_ood.data.dissertation_dataset import (
    create_dissertation_image_archive,
    unpack_dissertation_image_archive,
    verify_dissertation_checksums,
    verify_dissertation_image_files,
    write_dissertation_checksums,
)


def _write_png(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(value, value, value)).save(path)


def test_image_archive_round_trips_relative_manifest_paths(tmp_path: Path):
    root = tmp_path / "data"
    image_dir = root / "images" / "dissertation_v1"
    _write_png(image_dir / "id" / "train" / "id_train_000000.png", value=80)
    _write_png(
        image_dir / "ood" / "semantic_outlier" / "cifar10_natural" / "semantic_000000.png",
        value=120,
    )

    archive = create_dissertation_image_archive(
        image_dir=image_dir,
        root_dir=root,
        archive_path=tmp_path / "datasets" / "dissertation_v1" / "archives" / "images.zip",
    )

    assert archive.image_count == 2
    assert archive.total_bytes > 0
    with zipfile.ZipFile(archive.archive_path) as handle:
        assert sorted(handle.namelist()) == [
            "images/dissertation_v1/id/train/id_train_000000.png",
            "images/dissertation_v1/ood/semantic_outlier/cifar10_natural/semantic_000000.png",
        ]

    unpack_root = tmp_path / "school_server_data"
    unpacked = unpack_dissertation_image_archive(
        archive_path=archive.archive_path,
        root_dir=unpack_root,
    )

    assert unpacked.image_count == 2
    assert (unpack_root / "images" / "dissertation_v1" / "id" / "train" / "id_train_000000.png").exists()

    checksums = write_dissertation_checksums(
        dataset_dir=archive.archive_path.parent.parent,
        root_dir=root,
        image_dir=image_dir,
    )
    assert checksums.checked_files >= 3
    verified = verify_dissertation_checksums(
        dataset_dir=archive.archive_path.parent.parent,
        root_dir=root,
    )
    assert verified.checked_files == checksums.checked_files


def test_image_archive_manifest_has_no_private_paths(tmp_path: Path):
    root = tmp_path / "data"
    image_dir = root / "images" / "dissertation_v1"
    _write_png(image_dir / "id" / "test" / "id_test_000000.png", value=90)

    archive = create_dissertation_image_archive(
        image_dir=image_dir,
        root_dir=root,
        archive_path=tmp_path / "archive.zip",
    )

    text = archive.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "images/dissertation_v1/id/test/id_test_000000.png" in text


def test_direct_image_verification_does_not_require_archive(tmp_path: Path):
    root = tmp_path / "data"
    image_dir = root / "images" / "dissertation_v1"
    dataset_dir = tmp_path / "datasets" / "dissertation_v1"
    _write_png(image_dir / "id" / "train" / "id_train_000000.png", value=90)
    (dataset_dir / "README.md").parent.mkdir(parents=True)
    (dataset_dir / "README.md").write_text("direct images through Git LFS\n", encoding="utf-8")

    written = write_dissertation_checksums(
        dataset_dir=dataset_dir,
        root_dir=root,
        image_dir=image_dir,
    )
    direct = verify_dissertation_image_files(image_dir=image_dir, root_dir=root)
    verified = verify_dissertation_checksums(dataset_dir=dataset_dir, root_dir=root)

    assert direct.image_count == 1
    assert written.checked_files == 2
    assert verified.checked_files == 2


def test_unpack_script_help_loads():
    import importlib.util

    script_path = Path("scripts/unpack_dissertation_dataset.py")
    spec = importlib.util.spec_from_file_location("unpack_dissertation_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")

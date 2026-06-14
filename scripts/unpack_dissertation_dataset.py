#!/usr/bin/env python
"""Unpack and verify the dissertation v1 Git LFS image archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.dissertation_dataset import (
    unpack_dissertation_image_archive,
    verify_dissertation_checksums,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack datasets/dissertation_v1/lfs image archive into a private data root"
    )
    parser.add_argument("--dataset-dir", default="datasets/dissertation_v1")
    parser.add_argument("--root-dir", default="data", help="Private data root to receive images/")
    parser.add_argument(
        "--archive",
        default=None,
        help="Archive path; defaults to <dataset-dir>/lfs/dissertation_v1_images.zip",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify package files and unpacked images against checksums.sha256",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    archive = Path(args.archive) if args.archive else dataset_dir / "lfs" / "dissertation_v1_images.zip"
    unpacked = unpack_dissertation_image_archive(archive_path=archive, root_dir=args.root_dir)
    print(f"Unpacked {unpacked.image_count} images to {args.root_dir}")
    print(f"Archive SHA256: {unpacked.sha256}")
    if args.verify_checksums:
        verified = verify_dissertation_checksums(dataset_dir=dataset_dir, root_dir=args.root_dir)
        print(f"Verified {verified.checked_files} checksum entries")


if __name__ == "__main__":
    main()

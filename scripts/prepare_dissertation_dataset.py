#!/usr/bin/env python
"""Prepare the dissertation v1 dataset package from local, uncommitted images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.dissertation_dataset import (
    create_dissertation_image_archive,
    prepare_dissertation_dataset,
    write_dissertation_checksums,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local curated images and commit-safe dissertation v1 manifests"
    )
    parser.add_argument("--config", required=True, help="Dataset contract YAML")
    parser.add_argument("--root-dir", required=True, help="Private data root, usually data/")
    parser.add_argument("--prepared-syntheye-dir", required=True, help="Prepared or raw SynthEye FAF dir")
    parser.add_argument("--prepared-colour-fundus-dir", help="Prepared colour fundus OOD image dir")
    parser.add_argument("--prepared-oct-dir", help="Prepared OCT screenshot/B-scan OOD image dir")
    parser.add_argument("--prepared-cifar-dir", help="Prepared CIFAR/natural image OOD dir")
    parser.add_argument("--syntheye-dir", help="Alias for raw SynthEye source dir")
    parser.add_argument("--colour-fundus-dir", help="Alias for raw/prepared colour fundus dir")
    parser.add_argument("--oct-dir", help="Alias for raw/prepared OCT dir")
    parser.add_argument("--cifar-root", help="Alias for raw/prepared CIFAR/natural dir")
    parser.add_argument("--out-image-dir", required=True, help="Ignored local curated image output dir")
    parser.add_argument("--local-manifest-dir", required=True, help="Ignored local manifest output dir")
    parser.add_argument("--repo-dataset-dir", required=True, help="Commit-safe dataset package dir")
    parser.add_argument("--audit-dir", required=True, help="Ignored local audit output dir")
    parser.add_argument(
        "--archive-path",
        help="Optional Git LFS image archive path, usually datasets/dissertation_v1/lfs/dissertation_v1_images.zip",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-download-cifar",
        action="store_true",
        help="Accepted for reproducibility notes; this script does not download data",
    )
    parser.add_argument(
        "--allow-synthetic-surrogates",
        action="store_true",
        help="Generate clearly marked OOD surrogate rows when public OOD folders are unavailable",
    )
    parser.add_argument(
        "--allow-val-artifact",
        action="store_true",
        help="Allow artifact generation from validation ID rows for controlled smoke tests",
    )
    parser.add_argument(
        "--commit-safe-manifest-package",
        action="store_true",
        help="Write metadata-only docs and manifests under --repo-dataset-dir",
    )
    args = parser.parse_args()

    result = prepare_dissertation_dataset(
        config_path=args.config,
        root_dir=args.root_dir,
        prepared_syntheye_dir=args.prepared_syntheye_dir,
        prepared_colour_fundus_dir=args.prepared_colour_fundus_dir,
        prepared_oct_dir=args.prepared_oct_dir,
        prepared_cifar_dir=args.prepared_cifar_dir,
        syntheye_dir=args.syntheye_dir,
        colour_fundus_dir=args.colour_fundus_dir,
        oct_dir=args.oct_dir,
        cifar_root=args.cifar_root,
        out_image_dir=args.out_image_dir,
        local_manifest_dir=args.local_manifest_dir,
        repo_dataset_dir=args.repo_dataset_dir,
        audit_dir=args.audit_dir,
        seed=args.seed,
        allow_download_cifar=args.allow_download_cifar,
        allow_synthetic_surrogates=args.allow_synthetic_surrogates,
        allow_val_artifact=args.allow_val_artifact,
        commit_safe_manifest_package=args.commit_safe_manifest_package,
    )
    print("Dissertation dataset v1 package prepared")
    print(f"Repo package: {result.repo_dataset_dir}")
    print(f"Local audit: {result.audit_dir / 'manifest_audit.json'}")
    for name, path in sorted(result.repo_manifests.items()):
        rows = result.audit_summary["counts"][name]["rows"]
        print(f"{name}: {rows} rows -> {path}")
    if result.warnings:
        print("Warnings:")
        for warning in sorted(set(result.warnings)):
            print(f"  - {warning}")
    if args.archive_path:
        archive = create_dissertation_image_archive(
            image_dir=args.out_image_dir,
            root_dir=args.root_dir,
            archive_path=args.archive_path,
        )
        print(f"Image archive: {archive.archive_path}")
        print(f"Image archive manifest: {archive.manifest_path}")
        print(f"Image archive SHA256: {archive.sha256}")
        checksums = write_dissertation_checksums(
            dataset_dir=args.repo_dataset_dir,
            root_dir=args.root_dir,
            image_dir=args.out_image_dir,
        )
        print(f"Checksums: {checksums.checksum_path} ({checksums.checked_files} files)")


if __name__ == "__main__":
    main()

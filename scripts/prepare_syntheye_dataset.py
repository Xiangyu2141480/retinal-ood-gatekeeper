#!/usr/bin/env python
"""Prepare SynthEye FAF images into private data/images and ID manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.syntheye import prepare_syntheye_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="SynthEye folder with class subdirectories")
    parser.add_argument("--out-dir", required=True, help="Ignored image output directory")
    parser.add_argument("--manifest-dir", required=True, help="Directory for generated manifest CSV files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = prepare_syntheye_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
    )
    print(f"Prepared {result.total_images} SynthEye ID images")
    print(f"Class counts: {result.class_counts}")
    print(f"Train manifest: {result.train_manifest}")
    print(f"Val manifest: {result.val_manifest}")
    print(f"Test ID manifest: {result.test_manifest}")


if __name__ == "__main__":
    main()

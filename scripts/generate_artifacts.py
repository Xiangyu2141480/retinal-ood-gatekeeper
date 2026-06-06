#!/usr/bin/env python
"""Generate synthetic sensory artifacts for OOD stress testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.artifacts import ARTIFACT_TYPES, generate_artifact_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, help="Manifest containing held-out valid FAF rows")
    parser.add_argument("--out-dir", required=True, help="Directory where corrupted image copies are written")
    parser.add_argument("--out-manifest", required=True, help="CSV manifest to write for generated OOD artifacts")
    parser.add_argument("--root-dir", help="Optional root directory for relative manifest paths")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Input manifest splits allowed as artifact sources; default excludes train to avoid leakage",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Optional number of ID source images to use")
    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=list(ARTIFACT_TYPES),
        choices=list(ARTIFACT_TYPES),
        help="Artifact types to generate for each ID source image",
    )
    args = parser.parse_args()

    manifest_df = generate_artifact_dataset(
        args.input_manifest,
        args.out_dir,
        args.out_manifest,
        root_dir=args.root_dir,
        artifact_types=args.artifacts,
        split=args.split,
        source_splits=args.source_splits,
        seed=args.seed,
        limit=args.limit,
    )
    print(f"Saved {len(manifest_df)} artifact images")
    print(f"Saved artifact manifest to {args.out_manifest}")


if __name__ == "__main__":
    main()

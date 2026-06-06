#!/usr/bin/env python
"""Build an OOD manifest from folder mappings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.manifest_tools import build_ood_manifest, parse_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="Data root for relative image paths")
    parser.add_argument("--out-manifest", required=True, help="Manifest CSV to write")
    parser.add_argument(
        "--mapping",
        action="append",
        required=True,
        help="Folder mapping in relative/folder=ood_type format",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--source", default="public_ood")
    args = parser.parse_args()

    df = build_ood_manifest(
        root_dir=args.root_dir,
        mappings=[parse_mapping(raw) for raw in args.mapping],
        out_manifest=args.out_manifest,
        split=args.split,
        source=args.source,
    )
    print(f"Saved {len(df)} OOD rows to {args.out_manifest}")
    print(df.groupby("ood_type").size().to_dict())


if __name__ == "__main__":
    main()

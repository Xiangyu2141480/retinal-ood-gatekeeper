#!/usr/bin/env python
"""Merge manifest CSV files into one manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.manifest_tools import merge_manifest_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Merged manifest CSV to write")
    parser.add_argument("manifests", nargs="+", help="Input manifest CSV files")
    args = parser.parse_args()

    df = merge_manifest_files(args.manifests, args.out)
    print(f"Saved {len(df)} merged manifest rows to {args.out}")
    print(df.groupby("ood_type").size().to_dict())


if __name__ == "__main__":
    main()

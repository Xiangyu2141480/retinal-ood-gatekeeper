#!/usr/bin/env python
"""Validate local dataset manifests and print an audit summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.manifest_audit import ManifestAuditError, audit_manifests, format_audit_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one or more retinal OOD manifest CSVs")
    parser.add_argument("manifests", nargs="+", help="Manifest CSV files to validate")
    parser.add_argument("--root-dir", required=True, help="Root directory for relative image_path values")
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Validate schema/taxonomy/path style without checking file existence",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Fail if the same image_path appears across train/val/test splits",
    )
    parser.add_argument(
        "--allow-train-artifact",
        action="store_true",
        help="Allow sensory_artifact rows with source_split=train for controlled smoke tests",
    )
    parser.add_argument("--write-json", help="Optional path for a JSON audit summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        audit = audit_manifests(
            args.manifests,
            root_dir=args.root_dir,
            check_files=not args.no_check_files,
            fail_on_duplicates=args.fail_on_duplicates,
            allow_train_artifact=args.allow_train_artifact,
            write_json=args.write_json,
        )
    except ManifestAuditError as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print(format_audit_summary(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

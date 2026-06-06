#!/usr/bin/env python
"""Audit local dataset image files referenced by manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.image_audit import (
    ImageAuditError,
    audit_dataset_images,
    format_image_audit_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image-level dataset quality for local manifests")
    parser.add_argument("--root-dir", required=True, help="Dataset root used by manifest image_path")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Manifest CSV to audit; repeat for train/val/test manifests",
    )
    parser.add_argument("--out-dir", default="reports/local_audits", help="Local audit output directory")
    parser.add_argument("--write-json", help="Optional JSON audit output path")
    parser.add_argument("--make-contact-sheets", action="store_true", help="Write local-only contact sheets")
    parser.add_argument("--sample-per-group", type=int, default=16, help="Images per contact-sheet group")
    parser.add_argument("--fail-on-corrupt", action="store_true", help="Fail if any image cannot be opened")
    parser.add_argument(
        "--fail-on-duplicate-content-across-splits",
        action="store_true",
        help="Fail when the same SHA256 image content appears across train/val/test splits",
    )
    args = parser.parse_args()

    try:
        audit = audit_dataset_images(
            args.manifest,
            root_dir=args.root_dir,
            out_dir=args.out_dir,
            write_json=args.write_json,
            make_contact_sheets=args.make_contact_sheets,
            sample_per_group=args.sample_per_group,
            fail_on_corrupt=args.fail_on_corrupt,
            fail_on_duplicate_content_across_splits=args.fail_on_duplicate_content_across_splits,
        )
    except ImageAuditError as exc:
        print(f"Image audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(format_image_audit_summary(audit))


if __name__ == "__main__":
    main()

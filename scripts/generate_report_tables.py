#!/usr/bin/env python
"""Generate dissertation result tables from evaluation metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.report_tables import write_report_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run outputs")
    parser.add_argument(
        "--out",
        default="reports/generated/experiment_summary.md",
        help="Markdown path for the overall experiment summary",
    )
    parser.add_argument("--csv-out", help="Optional CSV path for the overall experiment summary")
    parser.add_argument("--per-ood-out", help="Optional Markdown path for per-OOD category metrics")
    parser.add_argument("--per-ood-csv-out", help="Optional CSV path for per-OOD category metrics")
    args = parser.parse_args()

    paths = write_report_tables(
        args.runs_dir,
        overall_markdown=args.out,
        overall_csv=args.csv_out,
        per_ood_markdown=args.per_ood_out,
        per_ood_csv=args.per_ood_csv_out,
    )
    print(f"Saved overall CSV to {paths.overall_csv}")
    print(f"Saved overall Markdown to {paths.overall_markdown}")
    print(f"Saved per-OOD CSV to {paths.per_ood_csv}")
    print(f"Saved per-OOD Markdown to {paths.per_ood_markdown}")


if __name__ == "__main__":
    main()

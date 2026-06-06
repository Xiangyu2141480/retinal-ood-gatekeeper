#!/usr/bin/env python
"""Generate reports/generated/index.md and companion dissertation tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.report_index import generate_report_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dissertation report index from generated experiment outputs."
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/generated",
        help="Directory containing metrics_summary.csv and generated run outputs",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Cases per TP/FP/FN/borderline bucket")
    args = parser.parse_args()

    outputs = generate_report_index(args.reports_dir, top_k=args.top_k)
    print(f"Saved report index to {outputs.files['index']}")
    for name, path in sorted(outputs.files.items()):
        if name != "index":
            print(f"Saved {name} to {path}")


if __name__ == "__main__":
    main()

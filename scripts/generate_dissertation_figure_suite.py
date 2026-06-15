#!/usr/bin/env python
"""Generate the curated dissertation figure suite from grid report outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.visualization.dissertation_suite import generate_dissertation_figure_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="Grid report directory containing metrics_summary.csv and runs/*/evaluation/scores.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/dissertation_figures",
        help="Output directory for curated figures, captions, and figure_index.md",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI")
    args = parser.parse_args()

    outputs = generate_dissertation_figure_suite(args.reports_dir, args.out_dir, dpi=args.dpi)
    for key, path in sorted(outputs.files.items()):
        print(f"Saved {key} to {path}")


if __name__ == "__main__":
    main()

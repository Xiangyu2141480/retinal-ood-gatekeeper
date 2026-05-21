#!/usr/bin/env python
"""Generate dissertation figure PNGs from aggregate evaluation metrics."""

from __future__ import annotations

import argparse

from retinal_ood.visualization.dissertation_figures import generate_dissertation_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run outputs")
    parser.add_argument(
        "--out-dir",
        default="reports/generated/figures",
        help="Directory where figure PNGs and manifest JSON will be written",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI")
    args = parser.parse_args()

    outputs = generate_dissertation_figures(args.runs_dir, args.out_dir, dpi=args.dpi)
    for name, path in sorted(outputs.figures.items()):
        print(f"Saved {name} figure to {path}")
    print(f"Saved figure manifest to {outputs.manifest}")


if __name__ == "__main__":
    main()

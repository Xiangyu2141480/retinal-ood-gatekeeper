#!/usr/bin/env python
"""Generate the dissertation multi-scheme comparison package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.visualization.multi_scheme_suite import (  # noqa: E402
    RunRoot,
    generate_multi_scheme_comparison_package,
)


def _parse_run_root(value: str) -> RunRoot:
    if "=" not in value:
        raise ValueError("--run-root values must use eval_set=path")
    eval_set, raw_path = value.split("=", 1)
    eval_set = eval_set.strip()
    raw_path = raw_path.strip()
    if not eval_set or not raw_path:
        raise ValueError("--run-root values must use eval_set=path")
    return RunRoot(eval_set=eval_set, path=Path(raw_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-scheme dissertation tables, figures, captions, and selection guide."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Evaluation set and generated run root, formatted as eval_set=path. Repeat as needed.",
    )
    parser.add_argument(
        "--results-dir",
        default="reports/dissertation_results/multi_scheme_comparison",
        help="Output directory for CSV/Markdown result tables.",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/dissertation_figures",
        help="Output directory for dissertation figures and figure guide files.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG figure resolution.")
    args = parser.parse_args()

    run_roots = [_parse_run_root(value) for value in args.run_root]
    outputs = generate_multi_scheme_comparison_package(
        run_roots,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
    )
    print(f"Wrote {len(outputs.tables)} table files to {Path(args.results_dir)}")
    print(f"Wrote {len(outputs.figures)} figure/guide files to {Path(args.figures_dir)}")


if __name__ == "__main__":
    main()

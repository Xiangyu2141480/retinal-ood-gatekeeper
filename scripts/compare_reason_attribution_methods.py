#!/usr/bin/env python
"""Compare optional Stage 2 rejected-input reason attribution methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.reason_attribution.comparison import (
    ComparisonConfig,
    run_reason_attribution_method_comparison,
)


def _parse_method_names(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(names) if names else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare optional post-rejection Stage 2 reason attribution methods."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    parser.add_argument("--include-subtype", action="store_true")
    parser.add_argument("--include-hierarchical", action="store_true")
    parser.add_argument("--include-rbf-svm", action="store_true")
    parser.add_argument(
        "--method",
        action="append",
        help="Optional method name or comma-separated list. Defaults to all required methods.",
    )
    parser.add_argument(
        "--global-image-size",
        type=int,
        default=24,
        help="Side length for deterministic pooled-pixel global features z(x).",
    )
    parser.add_argument(
        "--statistics-image-size",
        type=int,
        default=224,
        help="Side length for low-level image-statistics features h(x).",
    )
    args = parser.parse_args()

    result = run_reason_attribution_method_comparison(
        ComparisonConfig(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            test_manifest=args.test_manifest,
            root_dir=args.root_dir,
            out_dir=args.out_dir,
            figures_dir=args.figures_dir,
            seed=args.seed,
            unknown_threshold=args.unknown_threshold,
            include_subtype=args.include_subtype,
            include_hierarchical=args.include_hierarchical,
            include_rbf_svm=args.include_rbf_svm,
            method_names=_parse_method_names(args.method),
            global_image_size=args.global_image_size,
            statistics_image_size=args.statistics_image_size,
        )
    )

    print(f"out_dir: {result.out_dir}")
    print(f"figures_dir: {result.figures_dir}")
    print(f"best_family_method: {result.best_family_method}")
    print(f"best_subtype_method: {result.best_subtype_method}")
    print(f"selected_method: {result.selected_method}")
    for name, path in sorted(result.tables.items()):
        print(f"table:{name}: {path}")
    for name, path in sorted(result.figures.items()):
        print(f"figure:{name}: {path}")


if __name__ == "__main__":
    main()

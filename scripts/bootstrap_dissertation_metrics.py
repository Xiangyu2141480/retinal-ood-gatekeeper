#!/usr/bin/env python
"""Bootstrap confidence intervals for dissertation OOD gatekeeper metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.bootstrap import bootstrap_metric_ci  # noqa: E402
from retinal_ood.evaluation.report_tables import dataframe_to_markdown  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SCHEME_SCORES = {
    "image_statistics": (
        "Image statistics",
        "reports/generated/dissertation_runs/multi_scheme_primary/runs/image_statistics/evaluation/scores.csv",
    ),
    "autoencoder": (
        "Autoencoder",
        "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/autoencoder_baseline/evaluation/scores.csv",
    ),
    "global_feature_knn": (
        "Global feature kNN",
        "reports/generated/dissertation_runs/multi_scheme_primary/runs/global_feature_knn/evaluation/scores.csv",
    ),
    "mahalanobis_feature": (
        "Mahalanobis feature",
        "reports/generated/dissertation_runs/multi_scheme_primary/runs/mahalanobis_feature/evaluation/scores.csv",
    ),
    "patchcore_l3": (
        "PatchCore layer3",
        "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/evaluation/scores.csv",
    ),
    "patchcore_l2_l3": (
        "PatchCore layer2+layer3",
        "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer2_layer3/evaluation/scores.csv",
    ),
}


def _parse_scheme_score(value: str) -> tuple[str, str, Path]:
    """Parse name=label=path so ad hoc score tables can be bootstrapped."""

    parts = value.split("=", maxsplit=2)
    if len(parts) != 3:
        raise ValueError("--scheme-score values must use name=label=path")
    return parts[0], parts[1], Path(parts[2])


def _threshold_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    ood_scores = scores[labels == 1]
    if len(ood_scores) == 0:
        raise ValueError("Cannot derive threshold without OOD examples.")
    return float(np.quantile(ood_scores, 0.05))


def build_bootstrap_table(
    scheme_scores: dict[str, tuple[str, Path]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scheme, (label, score_path) in scheme_scores.items():
        scores_df = pd.read_csv(score_path)
        labels = scores_df["label"].astype(int).to_numpy()
        scores = scores_df["score"].astype(float).to_numpy()
        threshold = _threshold_at_95_tpr(labels, scores)
        ci = bootstrap_metric_ci(labels, scores, threshold=threshold, n_bootstrap=bootstrap_samples, seed=seed)
        ci.insert(0, "scheme", scheme)
        ci.insert(1, "scheme_label", label)
        ci.insert(2, "threshold_policy", "research_95_tpr")
        rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def _save_ci_plot(table: pd.DataFrame, metric: str, path: Path, *, title: str, xlabel: str) -> None:
    plot_df = table[table["metric"] == metric].copy()
    plot_df = plot_df.sort_values("estimate")
    y = np.arange(len(plot_df))
    err_low = plot_df["estimate"].to_numpy() - plot_df["ci_low"].to_numpy()
    err_high = plot_df["ci_high"].to_numpy() - plot_df["estimate"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(plot_df["estimate"], y, xerr=[err_low, err_high], fmt="o", color="#2454a6", ecolor="#8fb2df")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["scheme_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _markdown_summary(table: pd.DataFrame, *, bootstrap_samples: int) -> str:
    selected = table[table["metric"].isin(["auroc", "auprc", "fpr_at_95_tpr"])].copy()
    selected["estimate"] = selected["estimate"].map(lambda value: f"{value:.3f}")
    selected["95% CI"] = selected.apply(lambda row: f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]", axis=1)
    selected = selected[["scheme_label", "metric", "estimate", "95% CI"]]
    return "\n".join(
        [
            "# Bootstrap confidence intervals",
            "",
            f"Bootstrap resamples: `{bootstrap_samples}`.",
            "",
            dataframe_to_markdown(selected),
            "",
            "All intervals use ID/OOD labels from the held-out OOD evaluation split. The threshold column is used only",
            "for threshold-dependent safety metrics; AUROC and AUPRC remain threshold-free ranking metrics.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dissertation_results/robustness_analysis"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/dissertation_figures/robustness"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scheme-score",
        action="append",
        default=[],
        help="Optional score table as name=label=path. If omitted, dissertation defaults are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scheme_score:
        scheme_scores = {
            name: (label, path)
            for name, label, path in (_parse_scheme_score(value) for value in args.scheme_score)
        }
    else:
        scheme_scores = {name: (label, Path(path)) for name, (label, path) in DEFAULT_SCHEME_SCORES.items()}

    table = build_bootstrap_table(scheme_scores, bootstrap_samples=args.bootstrap_samples, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)
    (args.output_dir / "bootstrap_ci.md").write_text(
        _markdown_summary(table, bootstrap_samples=args.bootstrap_samples),
        encoding="utf-8",
    )

    _save_ci_plot(
        table,
        "auroc",
        args.figures_dir / "figure_bootstrap_ci_main_metrics.png",
        title="Bootstrap AUROC confidence intervals",
        xlabel="AUROC",
    )
    _save_ci_plot(
        table,
        "fpr_at_95_tpr",
        args.figures_dir / "figure_bootstrap_ci_fpr95.png",
        title="Bootstrap FPR at 95% TPR confidence intervals",
        xlabel="FPR at 95% TPR",
    )
    print(f"Wrote bootstrap table to {args.output_dir / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build final dissertation delivery indexes from committed evidence outputs.

This script intentionally does not train models or regenerate expensive experiment outputs.
It reads the merged dataset, comparison, robustness, and figure artifacts, then writes a
compact handoff bundle under reports/dissertation_final/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


DEFAULT_OUT_DIR = Path("reports/dissertation_final")

REQUIRED_EVIDENCE = [
    Path("docs/datasets/dissertation_dataset_v1.md"),
    Path("docs/datasets/dissertation_dataset_card_v1.md"),
    Path("docs/experiments/dissertation_results_interpretation.md"),
    Path("docs/experiments/dissertation_robustness_key_findings.md"),
    Path("reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv"),
    Path("reports/dissertation_results/multi_scheme_comparison/scheme_overview.csv"),
    Path("reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv"),
    Path("reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.csv"),
    Path("reports/dissertation_results/robustness_analysis/bootstrap_ci.csv"),
    Path("reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv"),
    Path("reports/dissertation_results/robustness_analysis/subtype_influence.csv"),
    Path("reports/dissertation_results/robustness_analysis/result_provenance.md"),
    Path("reports/dissertation_figures/figure_index.md"),
    Path("reports/dissertation_figures/robustness/figure_index.md"),
    Path("reports/dissertation_figures/robustness/top_figures_for_thesis.md"),
    Path("datasets/dissertation_v1/manifests/train_id.csv"),
    Path("datasets/dissertation_v1/manifests/val_id.csv"),
    Path("datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv"),
    Path("datasets/dissertation_v1/manifests/test_ood_full.csv"),
    Path("datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv"),
]

MAIN_FIGURES = [
    (
        "reports/dissertation_figures/figure_system_pipeline_overview.png",
        "System pipeline overview",
        "Methodology",
        "main text",
        "must_include",
        "Defines the binary ACCEPT/REJECT gatekeeper and prevents disease-classifier framing.",
    ),
    (
        "reports/dissertation_figures/figure_dataset_taxonomy.png",
        "Dataset taxonomy",
        "Dataset",
        "main text",
        "must_include",
        "Shows ID FAF versus modality-shift, sensory-artifact, and semantic-outlier OOD groups.",
    ),
    (
        "reports/dissertation_figures/figure_roc_overall_model_comparison.png",
        "Overall ROC comparison",
        "Results",
        "main text",
        "must_include",
        "Main ranking evidence for the unsupervised gatekeeper schemes.",
    ),
    (
        "reports/dissertation_figures/figure_pr_overall_model_comparison.png",
        "Overall PR comparison",
        "Results",
        "main text",
        "must_include",
        "Complements ROC under OOD-heavy evaluation sets.",
    ),
    (
        "reports/dissertation_figures/figure_layer_ablation_patchcore.png",
        "PatchCore layer ablation",
        "Experiments",
        "main text",
        "optional",
        "Summarizes PatchCore layer choice and why L3 is the qualitative/localization companion.",
    ),
    (
        "reports/dissertation_figures/figure_per_ood_type_comparison.png",
        "Per-OOD-type comparison",
        "Results",
        "main text",
        "must_include",
        "Shows how model performance changes across modality, artifact, and semantic OOD.",
    ),
    (
        "reports/dissertation_figures/figure_score_distribution_with_threshold.png",
        "Score distribution with threshold",
        "Threshold Safety",
        "main text",
        "must_include",
        "Makes the ID-calibrated decision threshold visible.",
    ),
    (
        "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png",
        "Sensory artifact heatmaps",
        "Discussion",
        "main text",
        "optional",
        "Provides visual localization evidence for generated artifact cases.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_bootstrap_ci_main_metrics.png",
        "Bootstrap confidence intervals",
        "Robustness Analysis",
        "main text",
        "must_include",
        "Adds uncertainty around the main ranking claims.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png",
        "Threshold policy trade-off",
        "Robustness Analysis",
        "main text",
        "must_include",
        "Separates research thresholds from deployment-style ID-calibrated thresholds.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png",
        "Feature-space PCA by OOD type",
        "Discussion",
        "main text",
        "must_include",
        "Explains why Mahalanobis works well for global feature shifts.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_method_disagreement_examples.png",
        "Method disagreement examples",
        "Failure Analysis",
        "main text",
        "must_include",
        "Shows why PatchCore remains useful as localizable supporting evidence.",
    ),
]

APPENDIX_FIGURES = [
    (
        "reports/dissertation_figures/figure_per_ood_subtype_comparison.png",
        "Per-OOD-subtype comparison",
        "Appendix",
        "backup",
        "Detailed subtype diagnostics.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_artifact_severity_reject_rate.png",
        "Artifact severity reject rate",
        "Appendix / Robustness",
        "optional",
        "Shows non-universal monotonicity and hard local artifacts.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_false_negative_ood_examples.png",
        "False-negative OOD examples",
        "Appendix / Failure Analysis",
        "optional",
        "Supports text-watermark limitation discussion.",
    ),
    (
        "reports/dissertation_figures/robustness/figure_runtime_vs_performance.png",
        "Runtime versus performance",
        "Appendix / Deployment Practicality",
        "optional",
        "Local smoke scoring estimate, not a deployment benchmark.",
    ),
]

MAIN_TABLES = [
    (
        "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.md",
        "Overall model metrics",
        "Results",
        "main text",
        "must_include",
        "Main AUROC/AUPRC/FPR@95%TPR comparison.",
    ),
    (
        "reports/dissertation_results/multi_scheme_comparison/model_selection_summary.md",
        "Model selection summary",
        "Results",
        "main text",
        "must_include",
        "Explains selected quantitative gatekeeper.",
    ),
    (
        "reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.md",
        "PatchCore layer ablation",
        "Experiments",
        "main text",
        "optional",
        "Layer-level PatchCore comparison.",
    ),
    (
        "reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.md",
        "Per-OOD-type metrics",
        "Results",
        "main text",
        "must_include",
        "Breaks performance down by OOD type.",
    ),
    (
        "reports/dissertation_results/robustness_analysis/bootstrap_ci.md",
        "Bootstrap confidence intervals",
        "Robustness Analysis",
        "main text",
        "must_include",
        "Uncertainty around main metrics.",
    ),
    (
        "reports/dissertation_results/robustness_analysis/threshold_policy_sweep.md",
        "Threshold policy sweep",
        "Threshold Safety",
        "main text",
        "must_include",
        "Research versus ID-calibrated deployment thresholds.",
    ),
    (
        "reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md",
        "Robustness takeaway table",
        "Discussion",
        "main text",
        "must_include",
        "Concise claim-to-implication summary.",
    ),
    (
        "reports/dissertation_results/robustness_analysis/runtime_resource_summary.md",
        "Runtime/resource summary",
        "Appendix",
        "appendix",
        "optional",
        "Local smoke scoring estimates and model artifact sizes.",
    ),
]

CLAIMS = [
    (
        "The system is an upstream OOD gatekeeper, not a disease classifier.",
        "README.md; docs/dissertation/README.md",
        "Documentation and decision wording",
        "figure_system_pipeline_overview.png",
        "Use ACCEPT valid FAF / REJECT invalid or OOD wording.",
    ),
    (
        "Training uses ID-only FAF images.",
        "datasets/dissertation_v1/manifests/train_id.csv",
        "scripts/validate_manifests.py",
        "final_result_summary.csv",
        "Training rows use label=0 and ood_type=id.",
    ),
    (
        "OOD data is evaluation/stress-test only.",
        "reports/dissertation_results/robustness_analysis/result_provenance.md",
        "scripts/generate_dissertation_robustness_analysis.py",
        "threshold_policy_sweep.md",
        "OOD labels are grouping/evaluation labels only.",
    ),
    (
        "Dataset v1 covers ID, modality shift, sensory artifact, and semantic outlier inputs.",
        "datasets/dissertation_v1/manifests/test_ood_full.csv",
        "scripts/validate_manifests.py",
        "figure_dataset_taxonomy.png",
        "Four input categories; not four supervised classes.",
    ),
    (
        "Mahalanobis is the best quantitative model on dataset v1.",
        "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
        "scripts/generate_multi_scheme_comparison_package.py",
        "figure_roc_overall_model_comparison.png",
        "AUROC 0.9724 and FPR@95%TPR 0.2000 on balanced subtype evaluation.",
    ),
    (
        "PatchCore L3 is the best localizable discussion model.",
        "reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv",
        "scripts/generate_selected_patchcore_heatmaps.py",
        "figure_method_disagreement_examples.png",
        "Quantitatively weaker than Mahalanobis but provides heatmap/localization evidence.",
    ),
    (
        "FPR@95%TPR is critical for safety interpretation.",
        "reports/dissertation_results/robustness_analysis/bootstrap_ci.csv",
        "scripts/bootstrap_dissertation_metrics.py",
        "figure_bootstrap_ci_fpr95.png",
        "Intervals overlap, so threshold-safety claims stay cautious.",
    ),
    (
        "Results are proof-of-concept due to synthetic ID fallback.",
        "docs/experiments/dissertation_robustness_key_findings.md",
        "Final documentation review",
        "claims_and_limitations_matrix.md",
        "Not real clinical FAF validation and not deployment-ready.",
    ),
]


def _repo_path(repo_root: Path, relative: str | Path) -> Path:
    return repo_root / Path(relative)


def _require(paths: list[Path], repo_root: Path, strict: bool, skip_missing: bool) -> list[Path]:
    missing = [path for path in paths if not _repo_path(repo_root, path).exists()]
    if missing and strict and not skip_missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    return missing


def _read_csv(repo_root: Path, relative: str | Path) -> pd.DataFrame:
    path = _repo_path(repo_root, relative)
    return pd.read_csv(path)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _manifest_summary(repo_root: Path) -> tuple[list[dict[str, str]], int, str]:
    manifests = [
        ("train_id", Path("datasets/dissertation_v1/manifests/train_id.csv")),
        ("val_id", Path("datasets/dissertation_v1/manifests/val_id.csv")),
        (
            "test_id_synthetic_fallback",
            Path("datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv"),
        ),
        ("test_ood_full", Path("datasets/dissertation_v1/manifests/test_ood_full.csv")),
        (
            "test_ood_balanced_by_subtype",
            Path("datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv"),
        ),
    ]
    rows: list[dict[str, str]] = []
    categories: set[str] = set()
    total = 0
    for name, relative in manifests:
        frame = _read_csv(repo_root, relative)
        total += len(frame)
        if "ood_type" in frame.columns:
            categories.update(str(value) for value in frame["ood_type"].dropna().unique())
        rows.append(
            {
                "split_or_manifest": name,
                "rows": str(len(frame)),
                "label_counts": _counts(frame, "label"),
                "ood_type_counts": _counts(frame, "ood_type"),
            }
        )
    ordered_categories = ", ".join(sorted(categories))
    return rows, total, ordered_categories


def _dataset_image_count(repo_root: Path) -> int:
    image_root = repo_root / "data/images/dissertation_v1"
    if not image_root.exists():
        return 0
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sum(1 for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _counts(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    counts = frame[column].value_counts().sort_index()
    return "; ".join(f"{key}={value}" for key, value in counts.items())


def _best_rows(repo_root: Path) -> dict[str, str]:
    metrics = _read_csv(repo_root, "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv")
    best = metrics.sort_values("auroc", ascending=False).iloc[0]
    patchcore = metrics[metrics["scheme"].astype(str).str.contains("patchcore_l3", na=False)]
    patchcore_row = patchcore.iloc[0] if not patchcore.empty else metrics.iloc[0]

    threshold = _read_csv(repo_root, "reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv")
    threshold_row = threshold[
        (threshold["scheme"] == "mahalanobis_feature")
        & (threshold["policy"] == "val_id_quantile_95")
    ]
    threshold_text = "Mahalanobis val_id_quantile_95"
    if not threshold_row.empty:
        row = threshold_row.iloc[0]
        threshold_text = (
            "Mahalanobis val_id_quantile_95: ID false rejection "
            f"{_format_float(row['id_false_rejection_rate'])}, OOD recall "
            f"{_format_float(row['ood_recall'])}"
        )

    subtype = _read_csv(repo_root, "reports/dissertation_results/robustness_analysis/subtype_influence.csv")
    maha_only = subtype[
        (subtype["scheme"] == "mahalanobis_feature") & (subtype["analysis"] == "only_subtype")
    ]
    hardest = maha_only.sort_values("auroc", ascending=True).iloc[0] if not maha_only.empty else None
    hardest_text = "text_watermark"
    if hardest is not None:
        hardest_text = f"{hardest['ood_subtype']} (AUROC {_format_float(hardest['auroc'])})"

    scheme_overview = _read_csv(repo_root, "reports/dissertation_results/multi_scheme_comparison/scheme_overview.csv")
    completed = int((scheme_overview["completion_status"] == "completed").sum())

    return {
        "best_quantitative_model": str(best["scheme_label"]),
        "best_quantitative_metrics": (
            f"AUROC {_format_float(best['auroc'])}, AUPRC {_format_float(best['auprc'])}, "
            f"FPR@95%TPR {_format_float(best['fpr_at_95_tpr'])}"
        ),
        "best_localization_model": str(patchcore_row["scheme_label"]),
        "safety_threshold_result": threshold_text,
        "hardest_subtype": hardest_text,
        "completed_model_configurations": str(completed),
        "scheme_count": str(len(metrics)),
    }


def _write_markdown_table(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_delivery_bundle(
    repo_root: str | Path = ".",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    refresh: bool = False,
    strict: bool = False,
    skip_missing: bool = False,
) -> dict[str, Path]:
    """Write the final delivery bundle and return output names to paths."""

    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    required = list(REQUIRED_EVIDENCE)
    required.extend(Path(row[0]) for row in MAIN_FIGURES[:8])
    missing = _require(required, repo_root, strict=strict, skip_missing=skip_missing)

    manifest_rows, manifest_total, categories = _manifest_summary(repo_root)
    image_count = _dataset_image_count(repo_root)
    best = _best_rows(repo_root)
    summary_rows = [
        {
            "item": "packaged_dataset_images",
            "value": str(image_count),
            "evidence": "data/images/dissertation_v1/",
        },
        {
            "item": "dataset_manifest_rows",
            "value": str(manifest_total),
            "evidence": "datasets/dissertation_v1/manifests/*.csv",
        },
        {
            "item": "dataset_categories",
            "value": categories,
            "evidence": "datasets/dissertation_v1/manifests/test_ood_full.csv",
        },
        {
            "item": "number_of_schemes",
            "value": best["scheme_count"],
            "evidence": "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
        },
        {
            "item": "completed_model_configurations",
            "value": best["completed_model_configurations"],
            "evidence": "reports/dissertation_results/multi_scheme_comparison/scheme_overview.csv",
        },
        {
            "item": "best_quantitative_model",
            "value": f"{best['best_quantitative_model']} ({best['best_quantitative_metrics']})",
            "evidence": "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
        },
        {
            "item": "best_localization_model",
            "value": best["best_localization_model"],
            "evidence": "reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv",
        },
        {
            "item": "safety_threshold_result",
            "value": best["safety_threshold_result"],
            "evidence": "reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv",
        },
        {
            "item": "hardest_subtype",
            "value": best["hardest_subtype"],
            "evidence": "reports/dissertation_results/robustness_analysis/subtype_influence.csv",
        },
        {
            "item": "key_robustness_result",
            "value": "Mahalanobis remains strongest under bootstrap AUROC/AUPRC; FPR intervals overlap.",
            "evidence": "reports/dissertation_results/robustness_analysis/bootstrap_ci.csv",
        },
        {
            "item": "key_failure_analysis_result",
            "value": "Text watermark dominates Mahalanobis false negatives; PatchCore L3 catches many misses.",
            "evidence": "reports/dissertation_results/robustness_analysis/method_disagreement_cases.md",
        },
    ]

    summary_md = out_dir / "final_result_summary.md"
    summary_csv = out_dir / "final_result_summary.csv"
    _write_csv(summary_csv, ["item", "value", "evidence"], summary_rows)
    summary_md.write_text(
        "# Final Dissertation Result Summary\n\n"
        "This is a thesis-ready summary of existing merged outputs. It does not rerun training.\n\n"
        "The project remains an unsupervised binary FAF OOD gatekeeper, not a disease classifier. "
        "Training is ID-only and OOD data is evaluation/stress-test only. "
        "`test_id_synthetic_fallback.csv` is synthetic ID fallback, not real clinical FAF validation.\n\n"
        "## Dataset Manifests\n\n"
        + _markdown_table_text(
            ["split_or_manifest", "rows", "label_counts", "ood_type_counts"], manifest_rows
        )
        + "\n## Summary Rows\n\n"
        + _markdown_table_text(["item", "value", "evidence"], summary_rows),
        encoding="utf-8",
    )

    figure_rows = [
        {
            "path": path,
            "title": title,
            "chapter": chapter,
            "placement": placement,
            "status": status,
            "why": why,
        }
        for path, title, chapter, placement, status, why in MAIN_FIGURES
    ]
    figure_index = out_dir / "final_figure_index.md"
    _write_markdown_table(
        figure_index,
        ["path", "title", "chapter", "placement", "status", "why"],
        figure_rows,
    )

    table_rows = [
        {
            "path": path,
            "title": title,
            "chapter": chapter,
            "placement": placement,
            "status": status,
            "why": why,
        }
        for path, title, chapter, placement, status, why in MAIN_TABLES
    ]
    table_index = out_dir / "final_table_index.md"
    _write_markdown_table(
        table_index,
        ["path", "title", "chapter", "placement", "status", "why"],
        table_rows,
    )

    checklist = out_dir / "final_reproducibility_checklist.md"
    checklist.write_text(
        "# Final Reproducibility Checklist\n\n"
        "One-command final verification after environment setup and `git lfs pull`:\n\n"
        "```bash\n"
        "python scripts/build_dissertation_delivery_bundle.py --strict --out-dir "
        "reports/dissertation_final && python scripts/final_repository_audit.py && "
        "python scripts/validate_manifests.py --root-dir data "
        "datasets/dissertation_v1/manifests/train_id.csv "
        "datasets/dissertation_v1/manifests/test_ood_full.csv\n"
        "```\n\n"
        "- [ ] `git lfs install && git lfs pull`\n"
        "- [ ] `pip install -e \".[dev]\"`\n"
        "- [ ] `python scripts/validate_manifests.py --root-dir data "
        "datasets/dissertation_v1/manifests/train_id.csv "
        "datasets/dissertation_v1/manifests/test_ood_full.csv`\n"
        "- [ ] `python scripts/audit_dataset_images.py --root-dir data --manifest "
        "datasets/dissertation_v1/manifests/train_id.csv --manifest "
        "datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt`\n"
        "- [ ] `python scripts/run_experiment_grid.py --grid-config configs/multi_scheme_grid.yaml "
        "--root-dir data --out-dir reports/generated/dissertation_delivery_smoke --only "
        "mahalanobis_feature --train-manifest datasets/dissertation_v1/manifests/train_id.csv "
        "--val-manifest datasets/dissertation_v1/manifests/val_id.csv "
        "--test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv "
        "--test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv "
        "--max-train-images 8 --max-test-images 8 --device cpu --seed 42`\n"
        "- [ ] `python scripts/build_dissertation_delivery_bundle.py --strict --out-dir "
        "reports/dissertation_final`\n"
        "- [ ] `python scripts/final_repository_audit.py`\n",
        encoding="utf-8",
    )

    claim_rows = [
        {
            "dissertation claim": claim,
            "evidence file": evidence,
            "script/source": source,
            "figure/table": figure,
            "notes": notes,
        }
        for claim, evidence, source, figure, notes in CLAIMS
    ]
    claim_index = out_dir / "final_claim_evidence_index.md"
    _write_markdown_table(
        claim_index,
        ["dissertation claim", "evidence file", "script/source", "figure/table", "notes"],
        claim_rows,
    )

    appendix_rows = [
        {"path": path, "title": title, "chapter": chapter, "status": status, "why": why}
        for path, title, chapter, status, why in APPENDIX_FIGURES
    ]
    if missing:
        appendix_rows.append(
            {
                "path": "missing evidence",
                "title": "Missing optional files",
                "chapter": "Build diagnostics",
                "status": "backup",
                "why": "; ".join(str(path) for path in missing),
            }
        )
    appendix_index = out_dir / "final_appendix_index.md"
    _write_markdown_table(appendix_index, ["path", "title", "chapter", "status", "why"], appendix_rows)

    # refresh is intentionally a no-op beyond overwriting deterministic outputs; kept for CLI clarity.
    _ = refresh
    return {
        "final_result_summary_md": summary_md,
        "final_result_summary_csv": summary_csv,
        "final_figure_index": figure_index,
        "final_table_index": table_index,
        "final_reproducibility_checklist": checklist,
        "final_claim_evidence_index": claim_index,
        "final_appendix_index": appendix_index,
    }


def _markdown_table_text(columns: list[str], rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final dissertation delivery bundle indexes.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--refresh", action="store_true", help="Overwrite deterministic final indexes.")
    parser.add_argument("--strict", action="store_true", help="Fail if required evidence is missing.")
    parser.add_argument("--skip-missing", action="store_true", help="Write indexes even with missing evidence.")
    args = parser.parse_args()

    try:
        outputs = build_delivery_bundle(
            out_dir=args.out_dir,
            refresh=args.refresh,
            strict=args.strict,
            skip_missing=args.skip_missing,
        )
    except Exception as exc:  # pragma: no cover - exercised by CLI usage
        print(f"delivery bundle failed: {exc}", file=sys.stderr)
        return 1
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

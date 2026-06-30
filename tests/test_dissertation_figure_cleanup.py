"""Regression checks for the final dissertation figure cleanup bundle."""

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_final_paper_ready_figures_have_png_and_pdf_exports():
    figure_paths = [
        "reports/dissertation_figures/figure_stage1_overall_comparison_combined.png",
        "reports/dissertation_figures/figure_patchcore_layer_ablation_combined.png",
        "reports/dissertation_figures/reason_attribution_method_comparison/"
        "figure_stage2_method_comparison_combined.png",
        "reports/dissertation_figures/figure_roc_overall_model_comparison.png",
        "reports/dissertation_figures/figure_pr_overall_model_comparison.png",
        "reports/dissertation_figures/figure_per_ood_type_comparison.png",
        "reports/dissertation_figures/figure_score_distribution_with_threshold.png",
        "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png",
        "reports/dissertation_figures/figure_heatmaps_modality_examples.png",
    ]
    for relative_path in figure_paths:
        png_path = ROOT / relative_path
        pdf_path = png_path.with_suffix(".pdf")
        assert png_path.exists(), relative_path
        assert png_path.stat().st_size > 10_000, relative_path
        assert pdf_path.exists(), str(pdf_path.relative_to(ROOT))
        assert pdf_path.stat().st_size > 1_000, str(pdf_path.relative_to(ROOT))
        with Image.open(png_path) as image:
            assert image.width >= 900
            assert image.height >= 600


def test_stage1_final_comparison_covers_all_expected_methods():
    metrics = pd.read_csv(
        ROOT / "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv"
    )
    schemes = set(metrics.loc[metrics["eval_set"].eq("balanced_by_subtype"), "scheme"])
    assert schemes == {
        "image_statistics",
        "autoencoder",
        "global_feature_knn",
        "mahalanobis_feature",
        "patchcore_l2",
        "patchcore_l3",
        "patchcore_l4",
        "patchcore_l2_l3",
    }


def test_final_figure_docs_prefer_combined_main_text_figures():
    shortlist = (ROOT / "docs/dissertation/final_figure_shortlist.md").read_text(
        encoding="utf-8"
    )
    main_text = shortlist.split("## Appendix Figures", maxsplit=1)[0]
    assert "figure_stage1_overall_comparison_combined.png" in main_text
    assert "figure_patchcore_layer_ablation_combined.png" in main_text
    assert "figure_stage2_method_comparison_combined.png" in main_text
    assert "figure_system_pipeline_overview.png" not in main_text

    revision_report = (
        ROOT / "reports/dissertation_final/figure_revision_report.md"
    ).read_text(encoding="utf-8")
    assert "per-sample score files" in revision_report
    assert "not inferred from scalar AUROC/AUPRC" in revision_report

    captions = (ROOT / "reports/dissertation_final/caption_suggestions.md").read_text(
        encoding="utf-8"
    )
    assert "figure_stage1_overall_comparison_combined.png" in captions
    assert "figure_patchcore_layer_ablation_combined.png" in captions
    assert "figure_stage2_method_comparison_combined.png" in captions
    assert "synthetic FAF fallback" in captions
    assert "not clinical diagnoses" in captions

    subtype_mapping = (
        ROOT / "reports/dissertation_final/subtype_label_mapping.md"
    ).read_text(encoding="utf-8")
    assert "short_code" in subtype_mapping
    assert "RECT" in subtype_mapping
    assert "TXT" in subtype_mapping

# Dissertation Figure Revision Report

No models were retrained. No dataset splits or experimental metrics were changed. All regenerated figures are derived from existing result tables, manifests or per-sample score files.

## Figure Decisions

| old figure | decision | rationale | new/output path | evidence source |
| --- | --- | --- | --- | --- |
| 01 Stage 1 pipeline | removed_from_main_text | Retained only for compatibility; superseded by the two-stage pipeline. | reports/dissertation_figures/figure_system_pipeline_overview.png | generated diagram code |
| 02 Dataset taxonomy | regenerated | British-English display text and clearer ID-only/OOD evaluation boundary. | reports/dissertation_figures/figure_dataset_taxonomy.png | dataset taxonomy script logic |
| 03 Manifest split sizes | regenerated | Added non-additive manifest-row warning and 3,100 packaged image statement. | reports/dissertation_figures/figure_manifest_split_sizes.png | datasets/dissertation_v1/manifests/*.csv |
| 04+05 Stage 1 overall comparison | combined | New two-panel main-text figure combining AUROC/AUPRC and FPR@95%TPR. | reports/dissertation_figures/figure_stage1_overall_comparison_combined.png | reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv |
| 06+07 PatchCore ablation | combined | New two-panel layer-ablation figure with detection and safety-oriented metrics. | reports/dissertation_figures/figure_patchcore_layer_ablation_combined.png | reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.csv |
| 08 Threshold trade-off | regenerated | Deployment and research thresholds use different markers; ID-validation-only note added. | reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png | reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv |
| 09 PCA | moved_to_appendix | Marked as qualitative two-dimensional projection only. | reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png | reports/dissertation_results/robustness_analysis/feature_space_projection.csv |
| 11+12 Stage 2 methods | combined | New two-panel validation/test macro-F1 figure preserving validation-rule family selection. | reports/dissertation_figures/reason_attribution_method_comparison/figure_stage2_method_comparison_combined.png | family_metrics_by_method.csv; subtype_metrics_by_method.csv |
| 13 Family confusion | regenerated | British-English display labels retained with raw counts. | reports/dissertation_figures/reason_attribution_method_comparison/figure_best_reason_family_confusion_matrix.png | best_reason_family_confusion_matrix.csv |
| 14 Subtype confusion | moved_to_appendix | Short labels CF/OCT/TXT/RECT/etc. replace opaque S1-S11 labels. | reports/dissertation_figures/reason_attribution_method_comparison/figure_best_subtype_confusion_matrix.png | best_subtype_confusion_matrix.csv; subtype_label_mapping.md |
| 15/16 ROC/PR | regenerated | Full eight-method ROC/PR curves regenerated from per-sample score files. | reports/dissertation_figures/figure_roc_overall_model_comparison.png; reports/dissertation_figures/figure_pr_overall_model_comparison.png | reports/generated/dissertation_runs/*/runs/*/evaluation/scores.csv |
| 17 Per-family comparison | regenerated | Full eight-method family-level AUROC heatmap regenerated from committed CSV. | reports/dissertation_figures/figure_per_ood_type_comparison.png | reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.csv |
| 18 Per-subtype heatmap | appendix_retained | Retained as detailed appendix diagnostic heatmap. | reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png | reports/dissertation_results/multi_scheme_comparison/per_ood_subtype_by_scheme.csv |
| 19 Score distribution | regenerated | Detector, threshold source and benchmark are now explicit. | reports/dissertation_figures/figure_score_distribution_with_threshold.png | Mahalanobis score file; metrics_by_scheme.csv |
| 20/21 PatchCore qualitative heatmaps | regenerated | Input/anomaly-map/overlay triptychs use representative subtypes from genuine PatchCore heatmap components. | reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png; reports/dissertation_figures/figure_heatmaps_modality_examples.png | PatchCore selected heatmap artifacts generated from existing score files and memory bank |

## Automatic Consistency Checks

| check | result |
| --- | --- |
| Stage 1 expected methods | PASS: eight methods are present in `metrics_by_scheme.csv`. |
| Full ROC/PR evidence | PASS: per-sample score files were available for all eight final Stage 1 configurations. |
| Scalar-to-curve fabrication | PASS: ROC/PR curves are regenerated from score files, not inferred from scalar AUROC/AUPRC. |
| Manifest warning | PASS: `figure_manifest_split_sizes` includes the non-additive row warning. |
| Subtype confusion labels | PASS: short labels are written to `subtype_label_mapping.md` and used in the confusion matrix. |
| Stage boundaries | PASS: generated figure text preserves Stage 1 ID-only detection and Stage 2 optional post-rejection attribution. |
| Clinical overclaim scan | PASS: generated captions state likely technical reasons and synthetic FAF fallback, not clinical diagnoses or validation. |

# Polished Caption Suggestions

## figure_system_pipeline_overview.png

Caption: FAF OOD gatekeeper pipeline. Stage 1 is fitted using ID rows only and separates accepted valid FAF inputs from rejected invalid or OOD inputs before downstream analysis.

## figure_dataset_taxonomy.png

Caption: Dataset v1 taxonomy showing ID FAF rows and the three OOD stress-test families. Stage 2 splits are explanation-only OOD splits; the ID test split uses synthetic FAF fallback.

## figure_manifest_split_sizes.png

Caption: Committed manifest split sizes for ID, OOD evaluation, and Stage 2 reason-attribution rows.

## figure_metrics_by_scheme.png

Caption: Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD evaluation set. AUPRC reflects the OOD-heavy class balance, so the safety-focused FPR@95%TPR comparison should also be considered.

## figure_fpr95_by_scheme.png

Caption: Safety-focused Stage 1 comparison of FPR@95%TPR, where lower values indicate fewer ID false positives at high OOD sensitivity.

## figure_layer_ablation_patchcore.png

Caption: PatchCore layer ablation compatibility figure showing AUROC and AUPRC only; FPR@95%TPR is reported separately because lower values are better.

## figure_patchcore_layer_detection_metrics.png

Caption: PatchCore layer ablation for detection metrics, with AUROC and AUPRC reported together.

## figure_patchcore_layer_safety_metric.png

Caption: PatchCore layer ablation for FPR@95%TPR, where lower values indicate fewer ID false positives.

## figure_per_ood_subtype_by_scheme.png

Caption: Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories.

## figure_per_ood_subtype_comparison.png

Caption: Stable-name subtype-level Stage 1 AUROC heatmap for Overleaf and final dissertation references.

## figure_threshold_policy_tradeoff.png

Caption: Mahalanobis threshold policy trade-off between ID false rejection and OOD recall. Research-only and deployment-style thresholds are shown separately.

## figure_feature_space_pca_by_ood_type.png

Caption: Mahalanobis feature-space PCA by OOD category. This is a qualitative visualization, not the primary quantitative metric.

## figure_two_stage_updated_pipeline.png

Caption: Two-stage pipeline showing Stage 1 ID-only rejection followed by optional Stage 2 reason attribution for rejected inputs. Stage 2 labels are likely explanations, not clinical diagnoses.

## figure_reason_method_family_macro_f1.png

Caption: Stage 2 reason-family method comparison with the PR #24 baseline shown.

## figure_reason_method_accuracy_macro_f1.png

Caption: Stage 2 method comparison showing paired test accuracy and macro-F1 for reason family attribution.

## figure_reason_method_subtype_macro_f1.png

Caption: Stage 2 subtype method comparison using test macro-F1 with the PR #24 baseline shown.

## figure_best_reason_family_confusion_matrix.png

Caption: Count confusion matrix for Stage 2 reason-family attribution using `linear_svm`.

## figure_best_reason_family_confusion_matrix_normalized.png

Caption: Row-normalized confusion matrix for Stage 2 reason-family attribution using `linear_svm`.

## figure_best_subtype_confusion_matrix.png

Caption: Count confusion matrix for Stage 2 subtype attribution using the non-oracle `hierarchical_classifier`. Subtype codes are defined in `subtype_label_mapping.md`.

## figure_best_subtype_confusion_matrix_normalized.png

Caption: Row-normalized confusion matrix for Stage 2 subtype attribution using the non-oracle `hierarchical_classifier`. Subtype codes are defined in `subtype_label_mapping.md`.

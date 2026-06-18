# Polished Caption Suggestions

## figure_system_pipeline_overview.png

Caption: Overall system pipeline for the Stage 1 ID-only FAF OOD gatekeeper. The gatekeeper accepts likely valid FAF inputs and rejects invalid/OOD inputs before downstream analysis.

## figure_dataset_taxonomy.png

Caption: Dataset taxonomy and committed manifest sizes for ID FAF, modality-shift OOD, sensory-artifact OOD, semantic outliers, and Stage 2 reason-attribution splits.

## figure_metrics_by_scheme.png

Caption: Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD evaluation set.

## figure_fpr95_by_scheme.png

Caption: Safety-focused Stage 1 comparison of FPR@95%TPR, where lower values indicate fewer ID false positives at high OOD sensitivity.

## figure_layer_ablation_patchcore.png

Caption: PatchCore layer ablation across evaluated ResNet feature layers.

## figure_per_ood_subtype_by_scheme.png

Caption: Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories.

## figure_per_ood_subtype_comparison.png

Caption: Stable-name subtype-level Stage 1 AUROC heatmap for Overleaf and final dissertation references.

## figure_threshold_policy_tradeoff.png

Caption: Mahalanobis threshold policy trade-off between ID false rejection and OOD recall.

## figure_feature_space_pca_by_ood_type.png

Caption: PCA projection of Mahalanobis feature space by OOD category.

## figure_two_stage_updated_pipeline.png

Caption: Two-stage pipeline showing Stage 1 ID-only rejection followed by optional Stage 2 reason attribution for rejected inputs.

## figure_reason_method_family_macro_f1.png

Caption: Stage 2 reason-family method comparison with the PR #24 baseline shown.

## figure_reason_method_accuracy_macro_f1.png

Caption: Stage 2 method comparison showing paired test accuracy and macro-F1 for reason family attribution.

## figure_best_reason_family_confusion_matrix.png

Caption: Confusion matrix for the selected Stage 2 `linear_svm` family method.

## figure_best_subtype_confusion_matrix.png

Caption: Confusion matrix for the non-oracle Stage 2 hierarchical subtype method.

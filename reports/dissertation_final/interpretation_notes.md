# Figure Interpretation Notes

## figure_system_pipeline_overview.png

Interpretation: This figure fixes the project framing as binary quality control rather than disease classification.

Why it matters: Use early in the dissertation to prevent supervised disease-classifier framing.

## figure_dataset_taxonomy.png

Interpretation: The taxonomy makes clear which data are used for ID-only training and which are used for evaluation or post-rejection explanation.

Why it matters: Supports the dataset chapter and the ID-only/OOD-only boundary.

## figure_manifest_split_sizes.png

Interpretation: The row-count view separates dataset composition from taxonomy.

Why it matters: Keeps the main taxonomy figure uncrowded while preserving manifest evidence.

## figure_metrics_by_scheme.png

Interpretation: Mahalanobis feature distance is the strongest quantitative gatekeeper in the final comparison.

Why it matters: Main quantitative model-selection figure.

## figure_fpr95_by_scheme.png

Interpretation: Mahalanobis has the best FPR@95%TPR among the evaluated Stage 1 methods.

Why it matters: Useful for threshold-safety discussion.

## figure_stage1_overall_comparison_combined.png

Interpretation: Mahalanobis is the strongest quantitative Stage 1 gatekeeper across the combined ranking and safety-oriented view.

Why it matters: Recommended main-text Stage 1 comparison figure.

## figure_roc_overall_model_comparison.png

Interpretation: The regenerated ROC figure is appendix evidence for the full final experiment matrix rather than the older AE/PatchCore-only subset.

Why it matters: Use as an appendix ranking-curve check, not as the main Stage 1 result figure.

## figure_pr_overall_model_comparison.png

Interpretation: The PR curves are useful but less visually discriminative because OOD prevalence is high.

Why it matters: Appendix companion for complete ranking-curve reporting.

## figure_per_ood_type_comparison.png

Interpretation: Global modality and semantic shifts are easier than sensory artefacts for most methods, with Mahalanobis strongest overall.

Why it matters: Replaces the older subset-only per-family figure.

## figure_layer_ablation_patchcore.png

Interpretation: PatchCore L3 is retained as the localization-oriented companion even though Mahalanobis is the strongest quantitative gatekeeper.

Why it matters: Justifies using PatchCore heatmaps in the dissertation.

## figure_patchcore_layer_detection_metrics.png

Interpretation: Layer 3 gives the strongest PatchCore detection trade-off.

Why it matters: Separates higher-is-better metrics from the safety metric.

## figure_patchcore_layer_safety_metric.png

Interpretation: Layer 3 also gives the lowest PatchCore FPR@95%TPR among the evaluated layers.

Why it matters: Prevents mixing metrics with opposite preference directions in one line plot.

## figure_patchcore_layer_ablation_combined.png

Interpretation: PatchCore L3 is the best localisation-oriented PatchCore configuration, but remains a companion to Mahalanobis rather than the strongest quantitative gatekeeper.

Why it matters: Recommended main-text PatchCore ablation figure.

## figure_per_ood_subtype_by_scheme.png

Interpretation: The heatmap shows strong performance on global shifts and weaker behaviour on subtle local artefacts such as text watermark.

Why it matters: Best appendix figure for detailed failure-mode questions.

## figure_per_ood_subtype_comparison.png

Interpretation: The stable comparison copy preserves the same subtype pattern while giving the dissertation a concise figure filename.

Why it matters: Use when a shorter filename is preferred for the main dissertation source.

## figure_score_distribution_with_threshold.png

Interpretation: The figure shows why modality and semantic shifts separate clearly while sensory artefacts overlap more with ID scores.

Why it matters: Optional main-text or appendix support for threshold-policy interpretation.

## figure_threshold_policy_tradeoff.png

Interpretation: The selected prototype policy balances low ID rejection with strong OOD recall but still requires real clinical validation.

Why it matters: Anchors deployment-threshold caveats.

## figure_feature_space_pca_by_ood_type.png

Interpretation: Global semantic and modality shifts separate more clearly than subtle sensory artifacts, explaining the method's strengths and weaknesses.

Why it matters: Connects quantitative results to feature-space intuition.

## figure_two_stage_updated_pipeline.png

Interpretation: Stage 2 explains possible rejection reasons after the Stage 1 decision and does not train or replace the gatekeeper.

Why it matters: Prevents leakage or disease-classifier misinterpretation.

## figure_reason_method_family_macro_f1.png

Interpretation: `linear_svm` is selected as the final family attribution method by validation macro-F1 and holds strong test macro-F1.

Why it matters: Main Stage 2 quantitative comparison.

## figure_stage2_method_comparison_combined.png

Interpretation: The combined Stage 2 figure presents explanation performance without changing the Stage 1 ID-only OOD gatekeeper boundary.

Why it matters: Recommended main-text Stage 2 comparison figure.

## figure_reason_method_accuracy_macro_f1.png

Interpretation: The paired metric view confirms that strong family attribution is not driven by accuracy alone.

Why it matters: Useful as an appendix companion to the main Stage 2 macro-F1 figure.

## figure_reason_method_subtype_macro_f1.png

Interpretation: The non-oracle hierarchical classifier is the selected subtype attribution method.

Why it matters: Main Stage 2 subtype-selection figure.

## figure_best_reason_family_confusion_matrix.png

Interpretation: Most family-level errors occur for semantic outliers, the hardest reason family.

Why it matters: Shows error structure rather than only aggregate performance.

## figure_best_reason_family_confusion_matrix_normalized.png

Interpretation: Percentages make the rare family-level errors easier to compare across rows.

Why it matters: Useful when discussing family-level error rates rather than counts.

## figure_best_subtype_confusion_matrix.png

Interpretation: The subtype classifier performs strongly overall but leaves rectangle annotation as the hardest subtype.

Why it matters: Best figure for fine-grained Stage 2 limitations.

## figure_best_subtype_confusion_matrix_normalized.png

Interpretation: Row percentages make subtype-specific confusion patterns easier to read.

Why it matters: Appendix companion for detailed Stage 2 error analysis.

## figure_heatmaps_sensory_artifact_examples.png

Interpretation: The triptych layout makes it clear which panel is the raw input and which is the heatmap.

Why it matters: Appendix qualitative support for PatchCore as the localisation-oriented companion.

## figure_heatmaps_modality_examples.png

Interpretation: The modality examples provide qualitative contrast to sensory artefacts.

Why it matters: Appendix qualitative support for the localisation discussion.

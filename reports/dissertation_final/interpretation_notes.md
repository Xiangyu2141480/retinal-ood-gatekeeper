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

## figure_grouped_stage2_method_comparison.png

Interpretation: Grouped validation selects `feature_statistics_fusion` for family attribution and the non-oracle `hierarchical_classifier` for subtype attribution. The grouped test scores are retrospective and were not used for selection.

Why it matters: This is the canonical Stage 2 model-comparison figure; Stage 1 remains the sole ID-only unsupervised rejection gate.

## figure_grouped_family_confusion_matrix.png

Interpretation: The selected family method separates all 420 grouped test cases correctly. All three family F1 values are 1.0000, so there is no unique hardest family.

Why it matters: It shows the controlled benchmark result and its fixed sample size, while the synthetic-backed setting prevents a clinical-generalisation claim.

## figure_grouped_subtype_confusion_matrix.png

Interpretation: The non-oracle hierarchy reaches grouped test accuracy 0.9357 and macro-F1 0.9176; `text_watermark` is the weakest subtype at F1 0.7742.

Why it matters: It exposes the remaining fine-grained errors after parent groups are kept within one partition.

## figure_legacy_vs_grouped_stage2_metrics.png

Interpretation: Grouped metrics are slightly higher than legacy row-level metrics in this deterministic allocation. Parent grouping eliminates cross-partition overlap but does not make variants within a split independent.

Why it matters: It frames the rerun as split-sensitivity evidence, not proof of clinical or patient-independent generalisation.

## Legacy row-level Stage 2 figures

The notes below refer to the preserved row-level package and are retained only for historical sensitivity analysis.

## figure_reason_method_family_macro_f1.png

Interpretation: `linear_svm` was selected within the legacy row-level family-attribution protocol; it is not the final grouped family method.

Why it matters: Main Stage 2 quantitative comparison.

## figure_stage2_method_comparison_combined.png

Interpretation: The legacy combined Stage 2 figure preserves the earlier row-level comparison without changing the Stage 1 ID-only OOD gatekeeper boundary.

Why it matters: Recommended main-text Stage 2 comparison figure.

## figure_reason_method_accuracy_macro_f1.png

Interpretation: The paired metric view confirms that strong family attribution is not driven by accuracy alone.

Why it matters: Useful as an appendix companion to the main Stage 2 macro-F1 figure.

## figure_reason_method_subtype_macro_f1.png

Interpretation: The non-oracle hierarchical classifier is the selected subtype attribution method.

Why it matters: Main Stage 2 subtype-selection figure.

## figure_best_reason_family_confusion_matrix.png

Interpretation: In the legacy row-level result, family-level errors were concentrated among semantic outliers; this is not the final grouped error structure.

Why it matters: Shows error structure rather than only aggregate performance.

## figure_best_reason_family_confusion_matrix_normalized.png

Interpretation: Percentages make the rare family-level errors easier to compare across rows.

Why it matters: Useful when discussing family-level error rates rather than counts.

## figure_best_subtype_confusion_matrix.png

Interpretation: In the legacy row-level result, rectangle annotation was the hardest subtype; the grouped protocol instead identifies `text_watermark` as weakest.

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

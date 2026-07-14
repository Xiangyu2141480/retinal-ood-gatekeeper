# Polished Caption Suggestions

## figure_system_pipeline_overview.png

Caption: FAF OOD gatekeeper pipeline. Stage 1 is fitted using ID rows only and separates accepted valid FAF inputs from rejected invalid or OOD inputs before downstream analysis.

## figure_dataset_taxonomy.png

Caption: Dataset v1 taxonomy showing ID FAF rows and the three OOD stress-test families. Stage 2 splits are explanation-only OOD splits; the ID test split uses synthetic FAF fallback.

## figure_manifest_split_sizes.png

Caption: Committed manifest split sizes for ID, OOD evaluation, and Stage 2 reason-attribution rows. Bars report task-specific manifest rows and are not additive; the same packaged image can appear in multiple manifests, and the underlying packaged collection contains 3100 image files.

## figure_metrics_by_scheme.png

Caption: Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD evaluation set. The axis spans 0 to 1; AUPRC reflects the OOD-heavy class balance, so the safety-focused FPR@95%TPR comparison should also be considered.

## figure_fpr95_by_scheme.png

Caption: Safety-focused Stage 1 comparison of FPR@95%TPR, where lower values indicate fewer ID false positives at high OOD sensitivity.

## figure_stage1_overall_comparison_combined.png

Caption: Stage 1 overall method comparison on the balanced-by-subtype benchmark. Panel (a) reports AUROC and AUPRC; panel (b) reports FPR@95%TPR, where lower values are better. AUPRC should be interpreted with the OOD-heavy evaluation prevalence.

## figure_roc_overall_model_comparison.png

Caption: Overall ROC comparison for the final eight Stage 1 OOD gatekeeper configurations on the balanced-by-subtype benchmark. The legend reports AUROC values and includes the final best quantitative model, Mahalanobis feature distance.

## figure_pr_overall_model_comparison.png

Caption: Overall precision-recall comparison for the final eight Stage 1 configurations on the balanced-by-subtype benchmark. High precision-recall values reflect the OOD-heavy evaluation prevalence and should be interpreted with AUROC and threshold-conditioned ID rejection.

## figure_per_ood_type_comparison.png

Caption: Per-OOD-family AUROC heatmap for the final eight Stage 1 configurations on the balanced-by-subtype benchmark.

## figure_layer_ablation_patchcore.png

Caption: PatchCore layer ablation compatibility figure showing AUROC and AUPRC only; FPR@95%TPR is reported separately because lower values are better.

## figure_patchcore_layer_detection_metrics.png

Caption: PatchCore layer ablation for detection metrics, with AUROC and AUPRC reported together.

## figure_patchcore_layer_safety_metric.png

Caption: PatchCore layer ablation for FPR@95%TPR, where lower values indicate fewer ID false positives.

## figure_patchcore_layer_ablation_combined.png

Caption: PatchCore layer ablation showing detection metrics and FPR@95%TPR side by side. Layer 3 provides the strongest PatchCore detection and safety-oriented performance; combining layers 2 and 3 does not improve over layer 3 alone.

## figure_per_ood_subtype_by_scheme.png

Caption: Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories.

## figure_per_ood_subtype_comparison.png

Caption: Stable-name subtype-level Stage 1 AUROC heatmap for Overleaf and final dissertation references.

## figure_score_distribution_with_threshold.png

Caption: Mahalanobis anomaly-score distributions on the balanced-by-subtype benchmark, with the 95th-percentile ID-validation threshold overlaid. The ID split is synthetic FAF fallback, not real clinical FAF validation.

## figure_threshold_policy_tradeoff.png

Caption: Mahalanobis threshold policy trade-off between ID false rejection and OOD recall. Research-only and deployment-style thresholds are shown separately.

## figure_feature_space_pca_by_ood_type.png

Caption: Mahalanobis feature-space PCA by OOD category. This is a qualitative visualization, not the primary quantitative metric.

## figure_two_stage_updated_pipeline.png

Caption: Two-stage pipeline showing Stage 1 ID-only rejection followed by optional Stage 2 reason attribution for rejected inputs. Stage 2 labels are likely explanations, not clinical diagnoses.

## figure_grouped_stage2_method_comparison.png

Caption: Parent-grouped Stage 2 reason-attribution comparison. Grouped validation macro-F1 selects `feature_statistics_fusion` for family attribution and the non-oracle `hierarchical_classifier` for subtype attribution; the displayed grouped test scores are retrospective and do not affect selection.

## figure_grouped_family_confusion_matrix.png

Caption: Count confusion matrix for `feature_statistics_fusion` on the 420-image parent-grouped Stage 2 test split. All three family-level F1 scores are 1.0000, so no family is uniquely hardest. This controlled synthetic-backed result should not be interpreted as clinical generalisation.

## figure_grouped_subtype_confusion_matrix.png

Caption: Count confusion matrix for the non-oracle `hierarchical_classifier` on the parent-grouped Stage 2 test split. Test accuracy is 0.9357 and macro-F1 is 0.9176; `text_watermark` has the lowest subtype F1 (0.7742).

## figure_legacy_vs_grouped_stage2_metrics.png

Caption: Sensitivity comparison between the legacy row-stratified and final parent-grouped Stage 2 protocols. Parent grouping eliminates cross-partition parent/group overlap, but variants from a common parent remain dependent within one split; metric differences from this single deterministic allocation are not evidence of clinical generalisation.

## Legacy row-level Stage 2 figures

The captions below describe the preserved row-level sensitivity package and must not be used as the final grouped headline evidence.

## figure_reason_method_family_macro_f1.png

Caption: Legacy row-level Stage 2 reason-family method comparison with the PR #24 baseline shown. Linear SVM was selected within that historical protocol using validation macro-F1 and the predefined simplicity/tie-breaking rule.

## figure_stage2_method_comparison_combined.png

Caption: Legacy row-level Stage 2 reason-attribution method comparison for rejected inputs. This figure is retained for split-sensitivity context and is superseded by the parent-grouped comparison.

## figure_reason_method_accuracy_macro_f1.png

Caption: Stage 2 method comparison showing paired test accuracy and macro-F1 for reason family attribution.

## figure_reason_method_subtype_macro_f1.png

Caption: Stage 2 subtype method comparison using test macro-F1 with the PR #24 baseline shown.

## figure_best_reason_family_confusion_matrix.png

Caption: Legacy row-level count confusion matrix for Stage 2 reason-family attribution using `linear_svm`.

## figure_best_reason_family_confusion_matrix_normalized.png

Caption: Legacy row-level row-normalized confusion matrix for Stage 2 reason-family attribution using `linear_svm`.

## figure_best_subtype_confusion_matrix.png

Caption: Legacy row-level count confusion matrix for Stage 2 subtype attribution using the non-oracle `hierarchical_classifier`. Short labels are defined in `subtype_label_mapping.md`.

## figure_best_subtype_confusion_matrix_normalized.png

Caption: Legacy row-level row-normalized confusion matrix for Stage 2 subtype attribution using the non-oracle `hierarchical_classifier`. Short labels are defined in `subtype_label_mapping.md`.

## figure_heatmaps_sensory_artifact_examples.png

Caption: Representative PatchCore L3 sensory-artefact examples shown as input image, anomaly map and overlay. These maps are qualitative localisation evidence only and should not be interpreted as pixel-level clinical ground truth.

## figure_heatmaps_modality_examples.png

Caption: Representative PatchCore L3 modality-shift examples shown as input image, anomaly map and overlay. These examples contrast broader modality differences with local artefact cases.

# Caption Suggestions

## figure_dataset_taxonomy.png

Dataset taxonomy diagram showing ID, modality shift, sensory artifact, and semantic outlier categories while preserving binary accept/reject modelling.

## figure_experiment_workflow.png

Experiment workflow diagram from dataset package through ID-only training, OOD evaluation, aggregation, and figure generation.

## figure_failure_modes_by_scheme.png

Observed OOD miss-rate patterns by broad OOD type, derived from completed primary predictions.

## figure_fpr95_by_scheme.png

Safety-focused view of FPR@95%TPR for the primary evaluation split; lower values indicate fewer ID false rejections at high OOD recall.

## figure_heatmaps_modality_examples.png

Representative modality-shift heatmap examples for the selected PatchCore model.

## figure_heatmaps_sensory_artifact_examples.png

Representative sensory artifact heatmap examples for the selected PatchCore model.

## figure_layer_ablation_patchcore.png

Original PatchCore layer-ablation figure from the primary result package.

## figure_methods_family_diagram.png

Method-family overview grouping reconstruction, global feature distance, statistical feature distance, and patch-level memory-bank approaches.

## figure_metrics_by_scheme.png

Primary balanced-by-subtype comparison of AUROC, AUPRC, and FPR@95%TPR across completed schemes. `test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation.

## figure_metrics_by_scheme_and_eval_set.png

AUROC stability comparison across the primary, secondary, and stress OOD evaluation sets.

## figure_model_selection_radar_or_table.png

Trade-off summary used to motivate Mahalanobis feature as the selected discussion model while retaining safety caveats.

## figure_patchcore_layer_ablation.png

PatchCore layer ablation showing the effect of feature layer selection; runtime-limited variants are marked explicitly rather than imputed.

## figure_patchcore_method.png

PatchCore method diagram showing feature extraction, ID patch memory, nearest-neighbor scoring, and thresholding.

## figure_per_ood_subtype_by_scheme.png

Subtype-level AUROC heatmap across the 11 OOD subtypes used for stress testing.

## figure_per_ood_subtype_comparison.png

Original per-OOD-subtype comparison for the selected PatchCore result package.

## figure_per_ood_type_by_scheme.png

Per-OOD-type AUROC heatmap for modality shifts, sensory artifacts, and semantic outliers.

## figure_per_ood_type_comparison.png

Original per-OOD-type comparison for the AE/PatchCore primary result package.

## figure_pr_overall_model_comparison.png

Overall precision-recall comparison for the original AE/PatchCore model set on the primary evaluation split. `test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation.

## figure_roc_overall_model_comparison.png

Overall ROC comparison for the original AE/PatchCore model set on the primary evaluation split. `test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation.

## figure_scheme_comparison_matrix.png

Matrix of evaluated unsupervised OOD gatekeeper schemes, showing training type, feature space, anomaly score, localization support, sensitivity expectations, compute cost, and completion status.

## figure_score_distribution_id_vs_ood.png

ID versus OOD anomaly-score distribution for the selected PatchCore model.

## figure_score_distribution_with_threshold.png

Score distribution for the selected PatchCore model with the validation-ID threshold overlaid.

## figure_system_pipeline_overview.png

System pipeline overview showing the OOD gatekeeper as an upstream binary accept/reject component.

## figure_threshold_policy.png

Original threshold-policy visualization for the selected PatchCore model.

## figure_threshold_policy_comparison.png

Deployment threshold trade-off comparing ID false rejection rate and OOD recall at the validation-ID threshold.

## figure_workload_summary.png

Summary of project workload across schemes, PatchCore variants, evaluation sets, OOD categories, subtypes, and evaluated images.

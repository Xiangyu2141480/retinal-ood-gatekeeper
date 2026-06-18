| path | title | chapter | placement | status | why |
| --- | --- | --- | --- | --- | --- |
| reports/dissertation_figures/figure_system_pipeline_overview.png | Overall pipeline | Introduction / Methods | main text | must_include | Defines the binary Stage 1 gatekeeper and prevents disease-classifier framing. |
| reports/dissertation_figures/figure_dataset_taxonomy.png | Dataset taxonomy and manifest sizes | Dataset | main text | must_include | Shows ID rows, OOD evaluation rows, and Stage 2 reason-attribution splits. |
| reports/dissertation_figures/figure_metrics_by_scheme.png | Stage 1 method comparison | Results | main text | must_include | Shows Mahalanobis as the strongest quantitative Stage 1 method. |
| reports/dissertation_figures/figure_fpr95_by_scheme.png | Stage 1 FPR@95%TPR comparison | Results / Safety | main text | must_include | Highlights the safety-oriented ranking where lower FPR@95%TPR is better. |
| reports/dissertation_figures/figure_layer_ablation_patchcore.png | PatchCore layer ablation | Methods / Results | main text | recommended | Supports PatchCore L3 as the localization-oriented companion. |
| reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png | Threshold policy trade-off | Threshold Safety | main text | must_include | Shows the ID false rejection versus OOD recall trade-off. |
| reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png | Feature-space PCA by OOD type | Discussion | main text | must_include | Explains why global shifts separate more strongly than subtle artifacts. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_two_stage_updated_pipeline.png | Two-stage reason-attribution pipeline | Methods / Phase 2 | main text | must_include | Shows Stage 2 as optional post-rejection explanation only. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_reason_method_family_macro_f1.png | Stage 2 family method comparison | Phase 2 Results | main text | must_include | Shows `linear_svm` as the selected family attribution method. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_best_reason_family_confusion_matrix.png | Best family confusion matrix | Phase 2 Results | main text | must_include | Shows family-level errors for the selected `linear_svm` method. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_best_subtype_confusion_matrix.png | Best subtype confusion matrix | Phase 2 Results | main text or appendix | recommended | Shows subtype-level errors for the non-oracle hierarchical method. |
| reports/dissertation_figures/figure_roc_overall_model_comparison.png | Overall ROC comparison | Results | appendix | backup | Supporting Stage 1 discrimination view. |
| reports/dissertation_figures/figure_pr_overall_model_comparison.png | Overall PR comparison | Results | appendix | backup | Supporting Stage 1 precision-recall view. |
| reports/dissertation_figures/figure_per_ood_type_comparison.png | Per-OOD-type comparison | Results | appendix | backup | Broad OOD category breakdown. |
| reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png | Per-subtype heatmap | Failure Analysis | appendix | backup | Detailed Stage 1 subtype-level diagnostic view. |
| reports/dissertation_figures/figure_score_distribution_with_threshold.png | Score distribution with threshold | Threshold Safety | appendix | backup | Shows anomaly scores and the decision threshold. |
| reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png | Sensory artifact heatmaps | Discussion | appendix | optional | Qualitative localization evidence for artifact cases. |
| reports/dissertation_figures/figure_heatmaps_modality_examples.png | Modality heatmaps | Discussion | appendix | optional | Qualitative localization evidence for modality shifts. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_reason_method_per_subtype_f1.png | Stage 2 subtype F1 heatmap | Phase 2 Results | appendix | backup | Fine-grained Stage 2 diagnostic figure. |
| reports/dissertation_figures/reason_attribution_method_comparison/figure_unknown_threshold_tradeoff.png | Stage 2 unknown-threshold trade-off | Discussion | appendix | optional | Supports cautious `unknown_ood` confidence wording. |

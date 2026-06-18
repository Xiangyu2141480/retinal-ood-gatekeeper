# Final Thesis Figure Shortlist

Use this as the practical dissertation figure menu. Main-text figures should tell the core story:
Stage 1 gatekeeper framing, Stage 1 method comparison, threshold/failure interpretation, and the
optional Stage 2 reason-attribution extension.

## Main Text Figures

| filename | suggested chapter | placement | one-line message |
| --- | --- | --- | --- |
| `reports/dissertation_figures/figure_system_pipeline_overview.png` | Introduction / Methods | main text | The project is an upstream binary FAF OOD gatekeeper, not a disease classifier. |
| `reports/dissertation_figures/figure_dataset_taxonomy.png` | Dataset | main text | Dataset v1 separates ID FAF training/evaluation rows from OOD stress-test and Stage 2 reason splits. |
| `reports/dissertation_figures/figure_metrics_by_scheme.png` | Results | main text | Mahalanobis is the strongest Stage 1 quantitative gatekeeper by AUROC/AUPRC. |
| `reports/dissertation_figures/figure_fpr95_by_scheme.png` | Results / Safety | main text | Mahalanobis also gives the best FPR@95%TPR among the evaluated Stage 1 methods. |
| `reports/dissertation_figures/figure_layer_ablation_patchcore.png` | Methods / Results | main text | PatchCore L3 is the selected localization-oriented companion. |
| `reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png` | Threshold Safety | main text | The deployment-style threshold trades ID false rejection against OOD recall. |
| `reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png` | Discussion | main text | Global modality and semantic shifts separate more clearly than subtle local artifacts. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_two_stage_updated_pipeline.png` | Methods / Phase 2 | main text | Stage 2 is optional post-rejection explanation and does not change Stage 1. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_reason_method_family_macro_f1.png` | Phase 2 Results | main text | `linear_svm` is the selected family attribution method. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_best_reason_family_confusion_matrix.png` | Phase 2 Results | main text | Family-level errors are rare, with `semantic_outlier` remaining hardest. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_best_subtype_confusion_matrix.png` | Phase 2 Results / Appendix | main text or appendix | The non-oracle hierarchical subtype method is strong but leaves `rectangle_annotation` hardest. |

## Appendix Figures

| filename | suggested chapter | placement | one-line message |
| --- | --- | --- | --- |
| `reports/dissertation_figures/figure_roc_overall_model_comparison.png` | Results | appendix | ROC comparison is a supporting view of Stage 1 discrimination. |
| `reports/dissertation_figures/figure_pr_overall_model_comparison.png` | Results | appendix | Precision-recall comparison complements ROC under the OOD-heavy evaluation setting. |
| `reports/dissertation_figures/figure_per_ood_type_comparison.png` | Results | appendix | Broad OOD category results show model strengths and weaknesses. |
| `reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png` | Failure Analysis | appendix | Subtype-level heatmap gives the most detailed Stage 1 failure-mode view. |
| `reports/dissertation_figures/figure_score_distribution_with_threshold.png` | Threshold Safety | appendix | Score distributions make the ID-calibrated decision threshold visible. |
| `reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png` | Discussion | appendix | PatchCore heatmaps provide qualitative localization evidence for sensory artifacts. |
| `reports/dissertation_figures/figure_heatmaps_modality_examples.png` | Discussion | appendix | Modality-shift heatmaps provide qualitative contrast to artifact cases. |
| `reports/dissertation_figures/robustness/figure_bootstrap_ci_main_metrics.png` | Robustness | appendix | Bootstrap confidence intervals support cautious model-ranking claims. |
| `reports/dissertation_figures/robustness/figure_method_disagreement_examples.png` | Failure Analysis | appendix | Disagreement cases explain why a localizable companion remains useful. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_reason_method_per_subtype_f1.png` | Phase 2 Results | appendix | Fine-grained Stage 2 subtype F1 is useful for viva and limitations discussion. |
| `reports/dissertation_figures/reason_attribution_method_comparison/figure_unknown_threshold_tradeoff.png` | Discussion | appendix | The `unknown_ood` trade-off supports cautious explanation confidence wording. |
| `reports/dissertation_figures/figure_experiment_workflow.png` | Reproducibility | appendix | Workflow diagram connects manifests, ID-only fitting, OOD evaluation, and figure generation. |

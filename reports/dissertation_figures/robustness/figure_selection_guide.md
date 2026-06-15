# Robustness Figure Selection Guide

Use `top_figures_for_thesis.md` for the final 8-12 figure story across the main and robustness packages. This
robustness-only guide keeps extra backup figures available for viva, appendix, or failure-analysis discussion. The
highest-priority robustness figures are bootstrap AUROC, threshold trade-off, artifact reject rate, feature-space PCA
by OOD type, method-disagreement examples, false-negative OOD examples, and runtime versus performance.

| Figure | Purpose | Section | Status |
| --- | --- | --- | --- |
| figure_bootstrap_ci_main_metrics.png | Bootstrap AUROC uncertainty; Mahalanobis ranking stability | Results | thesis_shortlist |
| figure_bootstrap_ci_fpr95.png | Bootstrap FPR@95%TPR uncertainty; wider/overlapping safety intervals | Safety | backup |
| figure_train_size_sensitivity_auroc.png | Train-size AUROC sensitivity; Mahalanobis data efficiency | Results | optional |
| figure_train_size_sensitivity_fpr95.png | Train-size FPR@95 sensitivity | Safety | backup |
| figure_threshold_policy_tradeoff.png | ID-calibrated threshold policy trade-off | Safety | thesis_shortlist |
| figure_id_rejection_vs_ood_recall.png | ID rejection versus OOD recall | Safety | optional |
| figure_threshold_policy_by_ood_type.png | OOD-type recall by threshold policy | Safety | backup |
| figure_artifact_severity_scores.png | Artifact severity score response, including non-monotonic cases | Stress tests | optional |
| figure_artifact_severity_reject_rate.png | Artifact severity reject rate and hard artifact families | Stress tests | thesis_shortlist |
| figure_artifact_severity_examples.png | Visual examples of generated artifact severities | Stress tests | optional |
| figure_method_disagreement_matrix.png | Method disagreement case counts | Failure analysis | optional |
| figure_method_disagreement_examples.png | Representative Mahalanobis/PatchCore disagreement examples | Failure analysis | thesis_shortlist |
| figure_failure_case_grid.png | Representative failure cases | Failure analysis | backup |
| figure_false_positive_id_examples.png | Synthetic-ID false-positive examples | Failure analysis | backup |
| figure_false_negative_ood_examples.png | OOD false negatives; text watermark failure mode | Failure analysis | thesis_shortlist |
| figure_feature_space_pca_by_ood_type.png | Feature-space PCA by OOD type; explains Mahalanobis strength | Feature interpretation | thesis_shortlist |
| figure_feature_space_pca_by_subtype.png | Feature-space PCA by subtype | Feature interpretation | optional |
| figure_feature_space_pca_by_score.png | Feature-space PCA by Mahalanobis score | Feature interpretation | optional |
| figure_runtime_vs_performance.png | Runtime versus performance; local smoke timing estimate | Deployment practicality | thesis_shortlist |
| figure_method_tradeoff_table.png | Method trade-off table | Deployment practicality | optional |
| figure_subtype_influence.png | Subtype influence diagnostics | Results | optional |

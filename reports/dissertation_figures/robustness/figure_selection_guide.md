# Robustness Figure Selection Guide

| Figure | Purpose | Section | Status |
| --- | --- | --- | --- |
| figure_bootstrap_ci_main_metrics.png | Bootstrap AUROC uncertainty | Results | must_include |
| figure_bootstrap_ci_fpr95.png | Bootstrap FPR@95%TPR uncertainty | Safety | must_include |
| figure_train_size_sensitivity_auroc.png | Train-size AUROC sensitivity | Results | must_include |
| figure_train_size_sensitivity_fpr95.png | Train-size FPR@95 sensitivity | Safety | optional |
| figure_threshold_policy_tradeoff.png | Threshold policy trade-off | Safety | must_include |
| figure_id_rejection_vs_ood_recall.png | ID rejection versus OOD recall | Safety | must_include |
| figure_threshold_policy_by_ood_type.png | OOD-type recall by threshold policy | Safety | optional |
| figure_artifact_severity_scores.png | Artifact severity score response | Stress tests | must_include |
| figure_artifact_severity_reject_rate.png | Artifact severity reject rate | Stress tests | must_include |
| figure_artifact_severity_examples.png | Visual examples of generated artifact severities | Stress tests | optional |
| figure_method_disagreement_matrix.png | Method disagreement case counts | Failure analysis | must_include |
| figure_method_disagreement_examples.png | Representative method disagreement examples | Failure analysis | must_include |
| figure_failure_case_grid.png | Representative failure cases | Failure analysis | optional |
| figure_false_positive_id_examples.png | ID false-positive examples | Failure analysis | optional |
| figure_false_negative_ood_examples.png | OOD false-negative examples | Failure analysis | optional |
| figure_feature_space_pca_by_ood_type.png | Feature-space PCA by OOD type | Feature interpretation | must_include |
| figure_feature_space_pca_by_subtype.png | Feature-space PCA by subtype | Feature interpretation | optional |
| figure_feature_space_pca_by_score.png | Feature-space PCA by Mahalanobis score | Feature interpretation | must_include |
| figure_runtime_vs_performance.png | Runtime versus performance | Deployment practicality | must_include |
| figure_method_tradeoff_table.png | Method trade-off table | Deployment practicality | optional |
| figure_subtype_influence.png | Subtype influence diagnostics | Results | optional |

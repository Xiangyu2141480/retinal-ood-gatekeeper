# Robustness Figure Caption Suggestions

These captions are thesis-facing drafts from the final PR #22 QA pass. They preserve the core framing:
unsupervised binary FAF OOD gatekeeper, ID-only training, OOD evaluation/stress testing only, synthetic fallback ID
test split, and proof-of-concept rather than clinical deployment validation.

## figure_bootstrap_ci_main_metrics.png

Bootstrap AUROC confidence intervals for the main OOD gatekeeper methods. Mahalanobis feature distance remains the strongest quantitative model in this evaluation, with AUROC 0.9724 [0.9638, 0.9800]. Training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_bootstrap_ci_fpr95.png

Bootstrap FPR@95%TPR confidence intervals for threshold-sensitive safety. Mahalanobis has the lowest estimate, but FPR intervals are wider and overlap with Global feature kNN, so this should be interpreted as directional safety evidence rather than clinical statistical proof. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_train_size_sensitivity_auroc.png

Train-size AUROC sensitivity. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_train_size_sensitivity_fpr95.png

Train-size FPR@95 sensitivity. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_threshold_policy_tradeoff.png

Threshold-policy trade-off between synthetic-ID false rejection and OOD recall. The research 95% TPR threshold uses OOD labels and is evaluation-only; deployment-style thresholds must be calibrated from validation ID scores. Mahalanobis `val_id_quantile_95` is the balanced prototype policy in PR #22.

## figure_id_rejection_vs_ood_recall.png

ID rejection versus OOD recall. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_threshold_policy_by_ood_type.png

OOD-type recall by threshold policy. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_artifact_severity_scores.png

Artifact severity score response for generated stress-test variants. Score curves are not universally monotonic: Mahalanobis border crop and autoencoder rectangle annotation are highly monotonic, while autoencoder blur decreases with severity. These are proof-of-concept stress tests, not clinical prevalence estimates.

## figure_artifact_severity_reject_rate.png

Deployment reject rate under generated artifact severity stress tests. Rectangle annotation and border crop are detected earliest, while text watermark and JPEG compression remain hard. The response is intentionally reported with weak/non-monotonic cases visible rather than smoothed away.

## figure_artifact_severity_examples.png

Visual examples of generated artifact severities. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_method_disagreement_matrix.png

Method disagreement case counts. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_method_disagreement_examples.png

Representative method-disagreement cases comparing Mahalanobis feature distance with PatchCore L3. Mahalanobis is stronger quantitatively, while PatchCore L3 remains useful for localized visual evidence and catches many Mahalanobis text-watermark misses.

## figure_failure_case_grid.png

Representative failure cases. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_false_positive_id_examples.png

Synthetic-ID false-positive examples for the primary Mahalanobis gatekeeper. These examples should be interpreted cautiously because `test_id_synthetic_fallback.csv` is synthetic FAF fallback, not real clinical FAF validation.

## figure_false_negative_ood_examples.png

OOD false-negative examples for the primary Mahalanobis gatekeeper. Text watermark dominates the missed OOD cases, accounting for 129 of 152 Mahalanobis false negatives, and is the most useful subtype for dissertation failure-mode discussion.

## figure_feature_space_pca_by_ood_type.png

PCA projection of Mahalanobis feature space by OOD type. Semantic outliers and OCT-like modality shifts separate clearly from ID, while sensory artifacts overlap more with ID. This visualization helps explain why Mahalanobis performs well on global shifts but struggles with subtle local artifacts.

## figure_feature_space_pca_by_subtype.png

Feature-space PCA by subtype. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_feature_space_pca_by_score.png

PCA projection colored by Mahalanobis score. Higher scores align with globally shifted OOD groups, while text watermark remains close to ID in feature space. This supports Mahalanobis as the primary gatekeeper and motivates local artifact handling as future work.

## figure_runtime_vs_performance.png

Runtime versus AUROC for candidate gatekeepers. Measurements are local smoke scoring estimates over 22 images with existing PR #21 artifacts reused; fit time was not remeasured. Mahalanobis provides the best performance/resource trade-off, while PatchCore L3 adds heatmap/localization support at higher cost.

## figure_method_tradeoff_table.png

Method trade-off table. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

## figure_subtype_influence.png

Subtype influence diagnostics. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation.

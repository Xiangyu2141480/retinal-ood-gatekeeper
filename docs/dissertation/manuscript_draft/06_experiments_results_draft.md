# Experiments and Results Draft

## Stage 1 OOD Gatekeeper Results

The main multi-scheme comparison evaluates ID-trained unsupervised OOD gatekeepers on dataset v1. Mahalanobis feature distance is the strongest quantitative model in the final comparison, with AUROC 0.9724, AUPRC 0.9975, and FPR@95%TPR 0.2000. The best balanced prototype threshold is Mahalanobis `val_id_quantile_95`, with ID false rejection 0.0467 and OOD recall 0.9079.

Robustness analysis supports the Mahalanobis ranking for AUROC and AUPRC, while FPR@95%TPR confidence intervals remain wider and overlap with Global feature kNN. The failure analysis identifies text watermark as the hardest Mahalanobis subtype. PatchCore L3 is quantitatively weaker than Mahalanobis but remains useful as a localizable heatmap companion, especially for method-disagreement examples.

## Optional Phase 2 Reason Attribution Results

The optional Phase 2 experiment evaluates rejected-input reason attribution. It is separate from the Stage 1 OOD gatekeeper: Stage 1 remains ID-only and unsupervised, and OOD labels are used only as Stage 2 explanation targets after rejection.

The best reason-family method is `linear_svm`. On the held-out reason test split, it achieves family test accuracy 0.9881 and family test macro-F1 0.9901. This improves over the PR #24 family baseline by +0.0954 macro-F1. The hardest family is `semantic_outlier`.

The best subtype method is the non-oracle `hierarchical_classifier`. It achieves subtype accuracy 0.9238 and subtype macro-F1 0.9059, improving over the PR #24 subtype baseline by +0.2335 macro-F1. The hierarchy routes by predicted family at test time, not ground-truth family. The hardest subtype is `rectangle_annotation`.

The family-level scores are plausible because modality shifts, sensory artifacts, and semantic outliers create separable image-derived feature patterns in the current taxonomy. The subtype result is also plausible because the hierarchy first separates broad families before making finer within-family decisions. However, the reason splits are disjoint by `image_path` but not by `parent_image_hash`, especially for generated sensory-artifact variants. These scores should therefore be described as held-out image-file performance, not parent-independent generalization.

## Figures and Tables

The Phase 2 main-text figure set should include the two-stage pipeline, the reason method comparison, the best reason-family confusion matrix, and optionally the best subtype confusion matrix. Reason attribution examples may be used qualitatively if space allows.

The Phase 2 main tables should include `best_method_summary.md` and, where space allows, the family method comparison table. The leakage sanity check belongs in the appendix because it documents feature exclusions, split disjointness, parent-hash overlap, and the non-oracle hierarchy check.

## Interpretation

The reason-attribution outputs should be described as likely explanations for rejected inputs. They are not clinical diagnoses and should not be used to claim disease classification, clinical triage, or deployment readiness.

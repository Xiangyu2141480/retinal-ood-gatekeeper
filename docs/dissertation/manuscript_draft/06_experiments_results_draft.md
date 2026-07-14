# Experiments and Results Draft

## Stage 1 OOD Gatekeeper Results

The main multi-scheme comparison evaluates ID-trained unsupervised OOD gatekeepers on dataset v1. Mahalanobis feature distance is the strongest quantitative model in the final comparison, with AUROC 0.9724, AUPRC 0.9975, and FPR@95%TPR 0.2000. The best balanced prototype threshold is Mahalanobis `val_id_quantile_95`, with ID false rejection 0.0467 and OOD recall 0.9079.

Robustness analysis supports the Mahalanobis ranking for AUROC and AUPRC, while FPR@95%TPR confidence intervals remain wider and overlap with Global feature kNN. The failure analysis identifies text watermark as the hardest Mahalanobis subtype. PatchCore L3 is quantitatively weaker than Mahalanobis but remains useful as a localizable heatmap companion, especially for method-disagreement examples.

## Optional Phase 2 Reason Attribution Results

The optional Phase 2 experiment evaluates rejected-input reason attribution. It is separate from the Stage 1 OOD gatekeeper: Stage 1 remains ID-only and unsupervised, and OOD labels are used only as Stage 2 explanation targets after rejection.

The final Stage 2 protocol groups all variants from a common synthetic parent into one partition. It contains 1260 training, 420 validation, and 420 test rows, with zero cross-partition image-path and group-ID overlap. Related variants remain dependent within their assigned split. Grouped validation family macro-F1 selects `feature_statistics_fusion` (0.9925); its retrospective grouped test accuracy and macro-F1 are both 1.0000. All three family test F1 values are 1.0000, so there is no unique hardest family.

Grouped validation subtype macro-F1 selects the non-oracle `hierarchical_classifier` (0.9059). Its retrospective grouped test accuracy is 0.9357 and macro-F1 is 0.9176. The hierarchy routes by predicted family at test time, not ground-truth family. The hardest grouped-test subtype is `text_watermark` (F1 0.7742).

Relative to the preserved legacy row-level protocol, grouped-minus-legacy test macro-F1 is +0.0099 for family attribution and +0.0116 for subtype attribution. These increases are split-allocation sensitivity results from one deterministic seed, not evidence that grouping removed within-split variant dependence or established clinical generalisation. The controlled family taxonomy is visually separable, and the hierarchy benefits from decomposing the 11-way subtype task into predicted-family routes.

## Figures and Tables

The Phase 2 main-text figure set should include the two-stage pipeline, `figure_grouped_stage2_method_comparison`, and `figure_grouped_family_confusion_matrix`; `figure_grouped_subtype_confusion_matrix` and the legacy-versus-grouped sensitivity figure can be placed in the appendix.

The Phase 2 main tables should include `reports/stage2_grouped/summary.md` and the compact grouped method summary. The grouped split audit and legacy-versus-grouped table belong in the appendix because they document feature isolation, zero cross-partition overlap, remaining within-split dependence, and the non-oracle hierarchy.

## Interpretation

The reason-attribution outputs should be described as likely explanations for rejected inputs. They are not clinical diagnoses and should not be used to claim disease classification, clinical triage, or deployment readiness.

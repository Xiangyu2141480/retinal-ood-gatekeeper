# Discussion Draft

## Main Interpretation

The completed project supports an unsupervised binary FAF OOD gatekeeper, not a disease classifier. Stage 1 learns from ID FAF data only and rejects inputs that look invalid or out-of-distribution relative to that ID reference. The strongest quantitative Stage 1 method is Mahalanobis feature distance, while PatchCore L3 remains valuable as a localizable qualitative companion.

The Stage 1 results should be interpreted as proof-of-concept stress-test evidence. Dataset v1 includes synthetic ID fallback and curated OOD groups, so the results do not establish clinical deployment readiness or clinical prevalence behavior.

## Phase 2 Interpretation

The optional Stage 2 reason-attribution module extends the system after rejection. It answers a different question from Stage 1: not whether to reject, but what likely reason family or subtype explains an already rejected input. This preserves the core unsupervised gatekeeper formulation because OOD labels are not used to fit Stage 1.

Grouped validation selects `feature_statistics_fusion` for Stage 2 family attribution (validation macro-F1 0.9925). Its retrospective grouped test accuracy and macro-F1 are both 1.0000. Because every family has test F1 1.0000, there is no unique hardest family. Grouped validation selects the non-oracle `hierarchical_classifier` for subtype attribution (validation macro-F1 0.9059); grouped test accuracy is 0.9357 and macro-F1 is 0.9176.

The hardest grouped-test subtype is `text_watermark` (F1 0.7742), rather than `rectangle_annotation` in the legacy row-level method comparison or `jpeg_compression` in the earlier PR #24 baseline. This change shows that residual error rankings depend on the split protocol and selected method, so they should be tied to their exact experiment rather than presented as intrinsic clinical difficulty.

## Limitations

The final Phase 2 splits are disjoint by both `image_path` and group ID, so no synthetic parent crosses train, validation, or test. This removes the specific cross-partition overlap found in the legacy protocol, but it does not make variants from one parent independent: those related variants remain together within one split. The benchmark also remains controlled, closed-set, and synthetic-backed, so grouped performance is not patient-, device-, or site-independent clinical evidence.

Reason labels are likely explanations, not clinical diagnoses. The Stage 2 module does not classify disease, grade pathology, infer biomarkers, or provide clinical decisions. It is best described as a post-rejection quality-control explanation layer.

## Future Work

Future work should validate Stage 1 on real clinical FAF data, calibrate thresholds prospectively, and test whether Stage 2 explanations help human quality-control workflows. For Phase 2, priorities are repeated grouped evaluations across seeds, real rejection cases from independent patients/devices/sites, open-world reason handling, and calibrated explanation confidence.

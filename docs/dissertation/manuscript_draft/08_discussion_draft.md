# Discussion Draft

## Main Interpretation

The completed project supports an unsupervised binary FAF OOD gatekeeper, not a disease classifier. Stage 1 learns from ID FAF data only and rejects inputs that look invalid or out-of-distribution relative to that ID reference. The strongest quantitative Stage 1 method is Mahalanobis feature distance, while PatchCore L3 remains valuable as a localizable qualitative companion.

The Stage 1 results should be interpreted as proof-of-concept stress-test evidence. Dataset v1 includes synthetic ID fallback and curated OOD groups, so the results do not establish clinical deployment readiness or clinical prevalence behavior.

## Phase 2 Interpretation

The optional Stage 2 reason-attribution module extends the system after rejection. It answers a different question from Stage 1: not whether to reject, but what likely reason family or subtype explains an already rejected input. This preserves the core unsupervised gatekeeper formulation because OOD labels are not used to fit Stage 1.

The selected Stage 2 family method, `linear_svm`, reaches family test accuracy 0.9881 and macro-F1 0.9901. The selected subtype method, the non-oracle `hierarchical_classifier`, reaches subtype accuracy 0.9238 and macro-F1 0.9059. These results suggest that deterministic image-derived features are strong enough to support likely explanation labels in the current taxonomy.

The hardest family is `semantic_outlier`, and the hardest subtype is `rectangle_annotation`. This differs from the earlier PR #24 baseline, where `jpeg_compression` was hardest, because the stronger Phase 2 comparison improves JPEG-compression attribution and shifts the residual weakness toward rectangle-style overlays.

## Limitations

The Phase 2 scores should be framed cautiously. The reason splits are disjoint by `image_path` but not by `parent_image_hash`, especially for generated sensory artifacts. This parent-hash overlap may make Stage 2 scores optimistic for generated artifact variants. A future grouped split by `parent_image_hash` is required before claiming parent-independent reason-attribution robustness.

Reason labels are likely explanations, not clinical diagnoses. The Stage 2 module does not classify disease, grade pathology, infer biomarkers, or provide clinical decisions. It is best described as a post-rejection quality-control explanation layer.

## Future Work

Future work should validate Stage 1 on real clinical FAF data, calibrate thresholds prospectively, and test whether Stage 2 explanations help human quality-control workflows. For Phase 2 specifically, the next experiment should regenerate reason train/validation/test splits grouped by `parent_image_hash` and repeat the method comparison.

# Reason Attribution Extension Draft

## Methodology

This extension adds an optional second stage to the retinal FAF OOD gatekeeper. The first stage remains the primary safety mechanism: an ID-only unsupervised OOD detector trained on valid FAF-like images and used to produce a binary `ACCEPT` or `REJECT` decision. The second stage is invoked only after the first stage rejects an input.

The first-stage gatekeeper remains an ID-only unsupervised OOD detector. The second-stage reason attribution module is a post-hoc supervised explanation layer applied only after rejection.

For a rejected image `x`, the explanation module extracts an image-derived feature vector `z(x) = g(f_psi(x))`. A lightweight classifier then estimates `p(r | x) = softmax(W z(x) + b)`, where `r` is the reason family. The predicted reason is `r_hat = argmax_r p(r | x)`. If the maximum probability is below a threshold `gamma`, the module reports `unknown_ood` rather than forcing a known explanation.

The reason families are `modality_shift`, `sensory_artifact`, and `semantic_outlier`. An optional subtype classifier predicts finer categories such as `colour_fundus`, `oct_screenshot`, `text_watermark`, `arrow_annotation`, `gaussian_noise`, and `cifar10_natural`.

## Experiments

The reason attribution manifests were derived from the dataset v1 OOD evaluation split only. No ID rows were included in the Stage 2 reason training, validation, or test manifests. Each row has `label=1`, a reason family in `ood_type`, and a subtype in `ood_subtype`. Patient identifiers, `Eye_ID`, clinical labels, disease labels, and biomarker labels are excluded.

The initial PR #24 baseline used deterministic image-derived features and a balanced logistic-regression classifier on the legacy row-level manifests. It is retained as historical context rather than final evidence. On that split, the reason family classifier achieved:

- accuracy: `0.8810`
- macro-F1: `0.8947`
- hardest reason family by F1: `modality_shift`
- `unknown_ood` rate at `gamma=0.5`: `0.0357`

With the optional subtype classifier enabled, the subtype results were:

- accuracy: `0.7429`
- macro-F1: `0.6724`
- hardest subtype by F1: `jpeg_compression`

The subtype result is weaker than the family-level result, which is expected because fine-grained artifact categories can share similar low-level visual statistics.

### Final parent-grouped multi-method comparison

The final comparison uses `reason_grouped_train`, `reason_grouped_val`, and
`reason_grouped_test`. A row is grouped by non-empty `parent_image_hash`, otherwise by its own
`image_path`. Seed 42 produces 1260/420/420 train/validation/test rows, and all pairwise image-path
and group-ID overlaps are zero. All transformed variants from a common parent remain together in
one partition; grouping does not make those variants independent within that partition.

All eight candidates were rerun: image-statistics logistic regression, global-feature k-nearest
neighbours, nearest centroid, logistic regression, linear SVM, random forest, feature-statistics
fusion, and the hierarchical family-to-subtype classifier. Inputs are image statistics `h(x)`,
pooled global features `z(x)`, or their concatenation. Metadata, targets, and Stage 1 anomaly scores
are excluded. Scalers and estimators fit grouped training rows only. Family and subtype methods are
selected independently by grouped validation macro-F1 with the predefined complexity/name
tie-breaks; test predictions are computed only after both choices are frozen.

Grouped validation selects `feature_statistics_fusion` for family attribution (accuracy 0.9929,
macro-F1 0.9925). Its retrospective grouped test accuracy and macro-F1 are both 1.0000. Each of the
three family test F1 values is 1.0000, so no family is uniquely hardest. Grouped validation selects
the non-oracle `hierarchical_classifier` for subtype attribution (accuracy 0.9238, macro-F1
0.9059). It reaches grouped test accuracy 0.9357 and macro-F1 0.9176; `text_watermark` is the
hardest subtype (F1 0.7742). The hierarchy routes with its predicted family rather than an oracle
ground-truth family.

The legacy row-level protocol is preserved as sensitivity evidence. Grouped-minus-legacy test
macro-F1 is +0.0099 for family attribution and +0.0116 for subtype attribution. The increase means
that this deterministic grouped allocation was not harder on those aggregate metrics; it does not
show that related variants became independent, nor does it establish patient, device, site, or
clinical generalisation.

This comparison preserves the conceptual boundary of the dissertation. Stage 1 remains the
ID-only unsupervised gatekeeper and alone decides acceptance or rejection. Stage 2 is optional,
supervised, and post-rejection. It is not a disease classifier, and its reason labels are likely
technical explanations rather than clinical diagnoses.

## Discussion

The two-stage design preserves the original unsupervised OOD formulation. Stage 1 decides whether the input should be rejected. Stage 2 explains the likely reason for rejection after that decision has already been made. Therefore, OOD taxonomy labels are not used to train the gatekeeper and do not convert the project into a supervised four-class classifier.

The main value of the extension is interpretability. A binary rejection is useful for guarding downstream FAF analysis, but a reason family can help an operator understand whether the rejection appears to be caused by an imaging modality mismatch, an acquisition or overlay artifact, or a semantic outlier. The `unknown_ood` threshold is included to avoid overconfident explanations when the Stage 2 classifier has low confidence.

These outputs should be described as likely explanations rather than clinical findings. The module is not a disease classifier, does not predict diagnosis, and should not be treated as clinical deployment validation.

## Future Work

Future work should validate the reason attribution layer on real clinical FAF rejection cases, not only the synthetic/public OOD taxonomy used here. A stronger explanation model could use richer frozen visual representations, calibrated uncertainty estimates, and open-world recognition methods. Human review studies could also test whether the reason labels are useful for quality-control workflows and whether `unknown_ood` improves safety by reducing forced explanations.

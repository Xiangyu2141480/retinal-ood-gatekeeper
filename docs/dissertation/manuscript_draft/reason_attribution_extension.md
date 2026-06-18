# Reason Attribution Extension Draft

## Methodology

This extension adds an optional second stage to the retinal FAF OOD gatekeeper. The first stage remains the primary safety mechanism: an ID-only unsupervised OOD detector trained on valid FAF-like images and used to produce a binary `ACCEPT` or `REJECT` decision. The second stage is invoked only after the first stage rejects an input.

The first-stage gatekeeper remains an ID-only unsupervised OOD detector. The second-stage reason attribution module is a post-hoc supervised explanation layer applied only after rejection.

For a rejected image `x`, the explanation module extracts an image-derived feature vector `z(x) = g(f_psi(x))`. A lightweight classifier then estimates `p(r | x) = softmax(W z(x) + b)`, where `r` is the reason family. The predicted reason is `r_hat = argmax_r p(r | x)`. If the maximum probability is below a threshold `gamma`, the module reports `unknown_ood` rather than forcing a known explanation.

The reason families are `modality_shift`, `sensory_artifact`, and `semantic_outlier`. An optional subtype classifier predicts finer categories such as `colour_fundus`, `oct_screenshot`, `text_watermark`, `arrow_annotation`, `gaussian_noise`, and `cifar10_natural`.

## Experiments

The reason attribution manifests were derived from the dataset v1 OOD evaluation split only. No ID rows were included in the Stage 2 reason training, validation, or test manifests. Each row has `label=1`, a reason family in `ood_type`, and a subtype in `ood_subtype`. Patient identifiers, `Eye_ID`, clinical labels, disease labels, and biomarker labels are excluded.

The Stage 2 model was trained with deterministic image-derived features and a balanced logistic-regression classifier. On the held-out reason test split, the reason family classifier achieved:

- accuracy: `0.8810`
- macro-F1: `0.8947`
- hardest reason family by F1: `modality_shift`
- `unknown_ood` rate at `gamma=0.5`: `0.0357`

With the optional subtype classifier enabled, the subtype results were:

- accuracy: `0.7429`
- macro-F1: `0.6724`
- hardest subtype by F1: `jpeg_compression`

The subtype result is weaker than the family-level result, which is expected because fine-grained artifact categories can share similar low-level visual statistics.

### Multi-method comparison for rejected-input reason attribution

To avoid presenting the Phase 2 explanation layer as a single unchallenged classifier, a systematic
multi-method comparison was run on the finalized `reason_train`, `reason_val`, and `reason_test`
splits. The comparison kept the first-stage OOD gatekeeper fixed and unchanged. Stage 1 remains an
ID-only unsupervised detector that decides whether an input is accepted or rejected; OOD taxonomy
labels are used only by the optional second-stage explanation module after rejection.

The compared Stage 2 methods included image-statistics logistic regression, k-nearest neighbours,
nearest centroid, logistic regression, linear SVM, random forest, feature-statistics fusion, and a
hierarchical family-to-subtype classifier. The committed run used deterministic image-derived
features only: low-level image statistics `h(x)`, pooled-pixel global features `z(x)`, and their
concatenation. Metadata fields such as file name, source dataset, notes, `ood_type`, and
`ood_subtype` were not used as input features; reason labels served only as Stage 2 targets.

The pre-specified selection rule was validation reason-family macro-F1, with simpler methods
preferred on ties. Under this rule, `linear_svm` was selected as the family-level explanation
method. It achieved validation family macro-F1 `0.9908` and held-out test family macro-F1 `0.9901`
with test accuracy `0.9881`, known coverage `0.9952` at `gamma=0.5`, and an `unknown_ood` rate
of `0.0048`. This improved over the PR #24 family macro-F1 baseline of `0.8947` by `+0.0954`.
The fused and hierarchical methods had a slightly higher raw test family macro-F1 (`0.9939`), but
they were not selected as the final family method because model selection was based on validation
performance rather than post-hoc test-set ranking.

For subtype attribution, the best method was the `hierarchical_classifier`. It achieved validation
subtype macro-F1 `0.8671` and test subtype macro-F1 `0.9059`, improving over the PR #24 subtype
baseline of `0.6724` by `+0.2335`. This result suggests that fine-grained reason attribution
benefits from first separating broad reason families and then routing to a family-specific subtype
model. The hierarchy is evaluated using the predicted family at test time, not the ground-truth
family, so it does not rely on oracle family labels during prediction.

The hardest family for the selected `linear_svm` family method was `semantic_outlier`
(`F1=0.9848`). The hardest subtype for the best hierarchical subtype method was
`rectangle_annotation` (`F1=0.7241`), consistent with the observation that some visual artifact
subtypes overlap in simple low-level appearance cues. These subtype predictions should therefore
be interpreted as likely rejection explanations rather than definitive labels.

This comparison preserves the conceptual boundary of the dissertation. The system remains an
unsupervised FAF OOD gatekeeper in Stage 1, with an optional post-hoc Stage 2 explanation module.
It is not a disease classifier, and the reason-attribution experiment should not be described as
clinical deployment validation.

## Discussion

The two-stage design preserves the original unsupervised OOD formulation. Stage 1 decides whether the input should be rejected. Stage 2 explains the likely reason for rejection after that decision has already been made. Therefore, OOD taxonomy labels are not used to train the gatekeeper and do not convert the project into a supervised four-class classifier.

The main value of the extension is interpretability. A binary rejection is useful for guarding downstream FAF analysis, but a reason family can help an operator understand whether the rejection appears to be caused by an imaging modality mismatch, an acquisition or overlay artifact, or a semantic outlier. The `unknown_ood` threshold is included to avoid overconfident explanations when the Stage 2 classifier has low confidence.

These outputs should be described as likely explanations rather than clinical findings. The module is not a disease classifier, does not predict diagnosis, and should not be treated as clinical deployment validation.

## Future Work

Future work should validate the reason attribution layer on real clinical FAF rejection cases, not only the synthetic/public OOD taxonomy used here. A stronger explanation model could use richer frozen visual representations, calibrated uncertainty estimates, and open-world recognition methods. Human review studies could also test whether the reason labels are useful for quality-control workflows and whether `unknown_ood` improves safety by reducing forced explanations.

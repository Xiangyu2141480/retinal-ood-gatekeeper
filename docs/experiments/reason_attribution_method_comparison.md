# Reason Attribution Method Comparison

## Motivation

The Phase 2 reason-attribution module explains rejected inputs after the Stage 1 gatekeeper
has already made the binary `ACCEPT` / `REJECT` decision. PR #24 introduced a single
deterministic image-statistics logistic-regression baseline. This experiment compares several
lightweight Stage 2 alternatives on the same finalized `reason_train`, `reason_val`, and
`reason_test` splits so the dissertation can report a selected method rather than a single
uncontested baseline.

Stage 1 remains unchanged: it is an ID-only unsupervised OOD gatekeeper trained only to reject
invalid or OOD inputs. OOD taxonomy labels are used only for this optional Stage 2 explanation
experiment.

## Methods Compared

The comparison script evaluated:

- `image_statistics_logreg`: balanced logistic regression on low-level image statistics `h(x)`.
- `global_feature_knn`: k-nearest neighbours on deterministic pooled image features `z(x)`.
- `nearest_centroid`: prototype baseline in the same pooled feature space.
- `logistic_regression`: balanced logistic regression on pooled image features.
- `linear_svm`: balanced linear SVM on pooled image features.
- `random_forest_or_gradient_boosting`: random forest on pooled image features.
- `feature_statistics_fusion`: balanced logistic regression on concatenated `z(x)` and `h(x)`.
- `hierarchical_classifier`: family-first logistic model followed by subtype models routed by the
  predicted family, not by the ground-truth family.

No required method was skipped. The optional RBF SVM was not run by default because the required
dissertation command is intended to remain bounded and reproducible.

## Feature Extraction Policy

All inputs to the Stage 2 classifiers are image-derived. The script uses:

- `h(x)`: deterministic image statistics, including intensity moments, entropy, edge density,
  border ratio, and histogram features.
- `z(x)`: deterministic pooled-pixel image features. This committed run did not use a pretrained
  CNN backbone, so the figures and tables should be described as lightweight image-derived
  features rather than CNN features.
- `z(x) || h(x)`: feature-statistics fusion.

Manifest metadata is never concatenated into model features. `image_path` is used only to load
the image pixels. The following fields are explicitly excluded from model input features:
`image_path`, `filename`, `file_name`, `source`, `source_dataset`, `source_url`,
`license_status`, `source_split`, `source_image_hash`, `parent_image_hash`, `notes`, `label`,
`ood_type`, `ood_subtype`, `synthetic_transform`, `severity`, and `split`. The `ood_type` and
`ood_subtype` columns are targets only. The tests include a metadata perturbation check to verify
that changing metadata fields does not change the feature matrix.

## Metrics

Family-level evaluation reports accuracy, macro-F1, balanced accuracy, per-family precision,
recall and F1, confusion matrices, coverage at `gamma=0.5`, accuracy excluding `unknown_ood`,
and coverage-accuracy curves. Subtype evaluation reports top-1 accuracy, macro-F1, per-subtype
precision, recall and F1, and a subtype confusion matrix.

The predeclared model-selection rule is validation reason-family macro-F1, with simpler and more
interpretable methods preferred on ties. Test metrics are reported only after this validation
selection.

## Results

The selected family-level method is `linear_svm`, with validation family macro-F1 `0.9908`.
On the held-out reason test split it achieved:

- family accuracy: `0.9881`
- family macro-F1: `0.9901`
- family balanced accuracy: `0.9872`
- known coverage at `gamma=0.5`: `0.9952`
- `unknown_ood` rate at `gamma=0.5`: `0.0048`
- accuracy excluding unknown outputs: `0.9928`

Relative to the PR #24 family macro-F1 baseline of `0.8947`, the selected validation method
improved by `+0.0954` on the test split. The strongest raw test family macro-F1 was slightly
higher for `feature_statistics_fusion` and `hierarchical_classifier` (`0.9939`), but those were
not selected because the selection rule was fixed on validation performance before reading the
test ranking.

The best subtype method is `hierarchical_classifier`, with validation subtype macro-F1 `0.8671`
and test subtype macro-F1 `0.9059`. This improves over the PR #24 subtype baseline of `0.6724`
by `+0.2335`.

The hardest selected family result was `semantic_outlier` (`F1=0.9848`) for the selected
family method. The hardest subtype for the best subtype method was `rectangle_annotation`
(`F1=0.7241`).

## Best Selected Method

For dissertation reporting, use `linear_svm` as the selected family-level Stage 2 reason
attribution method because it won the validation macro-F1 selection criterion. Use
`hierarchical_classifier` as the best subtype attribution method because the family-routed
subtype models substantially improved fine-grained subtype macro-F1.

This distinction is important: family reason selection and subtype attribution are related but
not identical objectives.

## Limitations

This experiment does not change the Stage 1 OOD gatekeeper and does not validate a clinical
deployment system. Stage 2 is a supervised post-hoc explanation layer trained on OOD taxonomy
labels after rejection. It is not a disease classifier and is not a supervised replacement for
OOD detection.

The comparison uses the finalized synthetic/public OOD taxonomy manifests. High performance on
these controlled reason labels should not be interpreted as real clinical FAF validation.
The reason splits are disjoint by `image_path`, but a final leakage sanity gate found overlap by
`parent_image_hash` across train/validation/test for generated sensory-artifact variants. This
means method-comparison scores may be optimistic for generated artifact subtypes and should be
reported as split-level rather than parent-independent generalization. A grouped-by-parent split
is recommended before making stronger claims about artifact-subtype robustness.

The pooled-pixel representation is deliberately lightweight and reproducible, but richer frozen
visual encoders may be worth testing later if runtime, dependency, and external validation
constraints are acceptable.

## Dissertation Interpretation

The result supports the claim that a lightweight optional Stage 2 module can provide useful
reason-family and subtype explanations for rejected inputs while preserving the original
unsupervised FAF OOD gatekeeper. The method comparison strengthens the dissertation by showing
that the final explanation method was selected systematically on a validation split rather than
chosen ad hoc after test evaluation.

Primary artifacts:

- Results: `reports/dissertation_results/reason_attribution_method_comparison/`
- Figures: `reports/dissertation_figures/reason_attribution_method_comparison/`
- Script: `scripts/compare_reason_attribution_methods.py`

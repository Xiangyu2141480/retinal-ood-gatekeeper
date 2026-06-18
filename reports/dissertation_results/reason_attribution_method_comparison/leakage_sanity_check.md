# Reason Attribution Leakage Sanity Check

## Scope

This sanity gate applies to PR #25, the optional Stage 2 rejected-input reason-attribution
comparison. Stage 1 remains the ID-only unsupervised FAF OOD gatekeeper and is not trained or
modified by this comparison. OOD taxonomy labels are used only after rejection for Stage 2 reason
family and subtype attribution.

## Feature Inputs Used

The compared Stage 2 methods use image-derived features only:

- `h(x)`: low-level image statistics, including intensity moments, entropy, edge density, border
  ratio, and histogram features.
- `z(x)`: deterministic pooled-pixel image features used as a lightweight global image
  representation.
- `z(x) || h(x)`: concatenated pooled-pixel and image-statistics features for the fusion and
  hierarchical methods.

`image_path` is used only to locate and load the image pixels. It is not encoded as a feature.

## Fields Excluded From Features

The following manifest fields are explicitly excluded from model input features:

- `image_path`
- `filename`
- `file_name`
- `source`
- `source_dataset`
- `source_url`
- `license_status`
- `source_split`
- `source_image_hash`
- `parent_image_hash`
- `notes`
- `label`
- `ood_type`
- `ood_subtype`
- `synthetic_transform`
- `severity`
- `split`

`ood_type` and `ood_subtype` are targets only. Tests perturb metadata fields and verify that the
feature matrix remains unchanged.

## Train/Validation/Test Disjointness

The existing reason manifests are OOD-only:

| split | rows | label values | reason families |
| --- | ---: | --- | --- |
| train | 1260 | `1` only | `modality_shift`, `semantic_outlier`, `sensory_artifact` |
| validation | 420 | `1` only | `modality_shift`, `semantic_outlier`, `sensory_artifact` |
| test | 420 | `1` only | `modality_shift`, `semantic_outlier`, `sensory_artifact` |

`image_path` overlap check:

| split pair | overlapping image paths |
| --- | ---: |
| train-validation | 0 |
| train-test | 0 |
| validation-test | 0 |

## Parent Image Hash Overlap

`parent_image_hash` overlap exists across the reason splits:

| split pair | overlapping parent hashes |
| --- | ---: |
| train-validation | 127 |
| train-test | 128 |
| validation-test | 108 |

The number of unique non-empty parent hashes is 150 in train, 127 in validation, and 128 in test.
There are 108 parent hashes shared by all three splits.

Rows whose parent appears in another split:

| split | rows with parent in another split | total rows |
| --- | ---: | ---: |
| train | 696 | 1260 |
| validation | 240 | 420 |
| test | 240 | 420 |

The overlap is concentrated in generated sensory-artifact variants. In validation and test, all
eight sensory-artifact subtypes contribute 30 overlapping-parent rows each. This means the PR #25
scores should be interpreted as performance on disjoint image files, not as parent-independent
generalization across generated variants.

## Hierarchical Classifier Oracle Check

The reported `hierarchical_classifier` is non-oracle. During training, subtype models are fitted
within training-set reason families. At validation/test time, subtype routing uses the predicted
family from the family model (`family_argmax`), not the ground-truth family. No oracle hierarchy is
reported as a deployed or selected subtype method.

The selected final family method is `linear_svm`. The selected final subtype method is the
non-oracle `hierarchical_classifier`.

## Confusion Matrix Sanity

The selected `linear_svm` family confusion matrix has three errors on the test split:

| true family | predicted modality_shift | predicted sensory_artifact | predicted semantic_outlier |
| --- | ---: | ---: | ---: |
| modality_shift | 80 | 0 | 0 |
| sensory_artifact | 0 | 240 | 0 |
| semantic_outlier | 1 | 2 | 97 |

The selected family score is plausible because broad reason families are visually separable in
the current taxonomy: modality shifts, synthetic/public sensory artifacts, and semantic outliers
produce different global and low-level image cues. However, the parent-hash overlap for generated
sensory artifacts likely inflates artifact-family and artifact-subtype performance.

The best subtype confusion matrix shows strong performance for modality and semantic subtypes and
remaining confusion among sensory-artifact subtypes. The hierarchy reaches subtype macro-F1
`0.9059` because family routing first separates broad reason families, then subtype models focus
on within-family decisions. This is especially helpful compared with a single flat subtype model.

`semantic_outlier` is the hardest selected family (`F1=0.9848`) because it has fewer test rows
than sensory artifacts and its natural-image outliers can share global intensity or texture cues
with other non-FAF inputs.

`rectangle_annotation` is the hardest subtype for PR #25 (`F1=0.7241`). Its errors are mainly
within the sensory-artifact family, where rectangular overlays can resemble arrow annotations,
text/overlay artifacts, or noise/crop patterns in simple image-derived features. This differs
from PR #24, where `jpeg_compression` was hardest under the flat image-statistics baseline. The
hierarchical/fusion comparison improves JPEG-compression attribution, shifting the residual
weakness toward rectangle-style annotations.

## Limitations

- The Stage 2 comparison is supervised post-hoc explanation and does not replace Stage 1 OOD
  detection.
- Stage 1 remains ID-only and unsupervised.
- No disease labels or clinical targets are predicted.
- The reason splits are disjoint by image path but not disjoint by `parent_image_hash`.
- Parent-hash overlap is a leakage limitation for generated sensory-artifact variants and may
  make the high Stage 2 scores optimistic.
- A future grouped-by-`parent_image_hash` split should be generated before claiming
  parent-independent reason-attribution robustness.

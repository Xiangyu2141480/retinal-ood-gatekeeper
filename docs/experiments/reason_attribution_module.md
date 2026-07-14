# Rejected-Input Reason Attribution Module

## Motivation

The original FAF OOD gatekeeper answers a binary safety question: should the image be accepted as valid FAF-like input, or rejected as invalid/OOD? That first-stage decision is intentionally conservative and does not attempt to diagnose why an input was rejected.

This Phase 2 module adds an optional post-rejection explanation layer. It estimates the likely reason family for an already rejected input, such as `modality_shift`, `sensory_artifact`, or `semantic_outlier`, and can optionally predict the finer `ood_subtype`. Low-confidence outputs are mapped to `unknown_ood`.

## Two-Stage Architecture

Stage 1 remains:

```text
input image -> ID-only unsupervised OOD gatekeeper -> ACCEPT valid FAF / REJECT OOD
```

Stage 2 runs only after rejection:

```text
rejected image -> reason attribution module -> reason family + optional subtype + unknown_ood
```

The first-stage gatekeeper remains an ID-only unsupervised OOD detector. The second-stage reason attribution module is a post-hoc supervised explanation layer applied only after rejection.

## Mathematical Formulation

The reason module extracts an image-derived feature vector:

```text
z(x) = g(f_psi(x)) in R^D
```

where `f_psi` is a frozen image feature extractor or deterministic image-feature transform, `g` is global pooling or feature aggregation, and `z(x)` is the Stage 2 image feature vector.

For reason family prediction:

```text
p(r | x) = softmax(W z(x) + b)
r_hat = argmax_r p(r | x)
```

Low-confidence unknown handling is:

```text
if max_r p(r | x) < gamma:
    output unknown_ood
else:
    output r_hat
```

The default `gamma` is `0.5`.

## Training Data

Stage 1 uses ID-only training data such as `datasets/dissertation_v1/manifests/train_id.csv`. It is not trained with `ood_type` or `ood_subtype`.

The final Stage 2 evaluation uses OOD taxonomy labels from:

- `datasets/dissertation_v1/manifests/reason_grouped_train.csv`
- `datasets/dissertation_v1/manifests/reason_grouped_val.csv`
- `datasets/dissertation_v1/manifests/reason_grouped_test.csv`

These reason manifests are built from the OOD evaluation manifest only. All rows have `label=1`, an explicit reason family in `ood_type`, and an explicit subtype in `ood_subtype`. Forbidden identifier columns are not written to the reason manifests.

The older `reason_train.csv`, `reason_val.csv`, and `reason_test.csv` are retained unchanged as
legacy row-stratified sensitivity manifests. In the final grouped protocol, a non-empty
`parent_image_hash` defines the group and `image_path` is the fallback group. Cross-partition
image-path and group-ID overlap is zero, although related variants remain dependent within the
single split containing their parent.

## Evaluation

The Stage 2 module is evaluated with:

- top-1 accuracy
- macro-F1
- per-class precision, recall, and F1
- reason family confusion matrix
- optional subtype accuracy and macro-F1
- optional subtype confusion matrix
- `unknown_ood` coverage and accuracy as `gamma` changes

Historical PR #24 baseline results using deterministic image-statistics features and logistic regression:

- reason family accuracy: `0.8810`
- reason family macro-F1: `0.8947`
- hardest reason family: `modality_shift`
- subtype accuracy: `0.7429`
- subtype macro-F1: `0.6724`
- hardest subtype: `jpeg_compression`
- `unknown_ood` rate at `gamma=0.5`: `0.0357`

The compact result files are under `reports/dissertation_results/reason_attribution/`. Figures are under `reports/dissertation_figures/reason_attribution/`.

## Limitations

Stage 2 is supervised explanation, not unsupervised detection. It does not replace the Stage 1 gatekeeper and does not decide whether the image should be accepted or rejected.

Reason labels are likely explanations, not clinical diagnoses. The classifier is limited to known OOD families and subtypes represented in the Stage 2 taxonomy. The `unknown_ood` threshold handles low-confidence cases but is not a full open-world recognition solution. True open-world OOD explanation may require broader data, richer visual reasoning, and external validation.

## Dissertation Framing

Use this wording:

> The first-stage gatekeeper remains an ID-only unsupervised OOD detector. The second-stage reason attribution module is a post-hoc supervised explanation layer applied only after rejection.

Also state that this project is not a disease classifier and that OOD labels are used only for the explanation module.

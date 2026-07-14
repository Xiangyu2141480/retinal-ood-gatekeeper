# Parent-grouped Stage 2 reason attribution

## Purpose and system boundary

The initial Stage 2 manifests were disjoint by `image_path`, but transformed
sensory-artifact variants from the same synthetic parent could occur in more
than one partition. The legacy audit measured 127 shared parents between train
and validation, 128 between train and test, 108 between validation and test,
and 108 across all three splits. This dependence could make attribution results
optimistic.

The pre-remediation evidence is preserved verbatim in
`docs/dissertation/vincent_feedback_code_audit.md`,
`reports/audit/parent_overlap_summary.csv`, and
`reports/audit/parent_overlap_examples.csv`.

The final protocol eliminates cross-partition parent/group overlap without
changing Stage 1. It does not make variants from a common parent independent:
those variants remain dependent within the single partition to which their
parent is assigned. Stage 1 remains the ID-only unsupervised binary FAF OOD
gatekeeper. Stage 2 is an optional supervised explanation module invoked only
after rejection. OOD family/subtype labels are Stage 2 targets only; Stage 1
anomaly scores are not Stage 2 features. Neither stage is a disease classifier.

## Group definition and deterministic split

For each OOD row:

```python
group_id = parent_image_hash if parent_image_hash is non-empty else image_path
```

Empty strings, whitespace, `None`, and NaN parent hashes are treated as absent.
All eight sensory-artifact variants sharing a parent hash are assigned to one
partition. `colour_fundus`, `oct_screenshot`, and `cifar10_natural` rows have no
parent hash and are singleton image-path groups. The custom deterministic
stratification uses seed 42 and 60/20/20 targets.

| split | rows | unique groups | modality shift | sensory artifact | semantic outlier |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1260 | 630 | 240 | 720 | 300 |
| validation | 420 | 210 | 80 | 240 | 100 |
| test | 420 | 210 | 80 | 240 | 100 |

Every sensory subtype has 90/30/30 train/validation/test rows. Colour fundus
and OCT screenshot each have 120/40/40; CIFAR-10 natural has 300/100/100.

Final overlap audit:

| scope | shared image paths | shared group IDs |
| --- | ---: | ---: |
| train-validation | 0 | 0 |
| train-test | 0 | 0 |
| validation-test | 0 | 0 |
| all three | 0 | 0 |

Evidence: `reports/stage2_grouped/split_audit.md` and
`reports/stage2_grouped/group_overlap_summary.csv`.

## Manifests and lineage

- Final grouped training: `datasets/dissertation_v1/manifests/reason_grouped_train.csv`
- Final grouped validation: `datasets/dissertation_v1/manifests/reason_grouped_val.csv`
- Final grouped test: `datasets/dissertation_v1/manifests/reason_grouped_test.csv`
- Legacy row-stratified manifests retained unchanged:
  `reason_train.csv`, `reason_val.csv`, and `reason_test.csv`

The grouped manifest SHA-256 values are recorded in
`reports/stage2_grouped/split_audit.md`. Manifest `image_path` values beginning
`images/...` are resolved using `root_dir=data`, so they point to
`data/images/...`.

## Model selection and isolation

All eight pre-specified candidates were rerun. Every scaler and estimator is
fit on grouped training rows only. Family selection uses grouped validation
family macro-F1; subtype selection uses grouped validation subtype macro-F1.
The existing lower-complexity and method-name tie-breaks are applied within a
1e-6 tolerance. The selected methods are frozen before any test predictions are
computed.

The hierarchical classifier is non-oracle: at inference time it routes subtype
prediction using its own predicted family, never ground-truth family.
`StandardScaler` is inside each training pipeline. Metadata fields and Stage 1
scores are excluded from features.

## Final grouped results

Family selection:

- selected method: `feature_statistics_fusion`;
- validation accuracy: 0.9928571428571429;
- validation macro-F1: 0.9924836011792534;
- grouped test accuracy: 1.0;
- grouped test macro-F1: 1.0;
- all three family test F1 values are 1.0, so there is no unique hardest family.

Subtype selection:

- selected method: non-oracle `hierarchical_classifier`;
- validation accuracy: 0.9238095238095239;
- validation macro-F1: 0.9058507271445498;
- grouped test accuracy: 0.9357142857142857;
- grouped test macro-F1: 0.9175854801338393;
- hardest subtype: `text_watermark`, test F1 0.7741935483870969.

These are retrospective grouped test measurements for validation-selected
models, not test-selected winners.

## Legacy sensitivity comparison

| metric | legacy row-level | parent-grouped | grouped minus legacy |
| --- | ---: | ---: | ---: |
| selected family validation macro-F1 | 0.990774462414277 | 0.9924836011792534 | +0.001709138764976359 |
| family test accuracy | 0.988095238095238 | 1.0 | +0.011904761904761973 |
| family test macro-F1 | 0.9900756867155014 | 1.0 | +0.00992431328449861 |
| selected subtype validation macro-F1 | 0.8671040060661988 | 0.9058507271445498 | +0.03874672107835109 |
| subtype test accuracy | 0.923809523809524 | 0.9357142857142856 | +0.01190476190476164 |
| subtype test macro-F1 | 0.9059435494826286 | 0.9175854801338392 | +0.011641930651210664 |

The grouped scores did not decrease. This does not invalidate the overlap
concern: the allocation contains different unseen parent groups, and a single
deterministic split can be easier or harder by chance. The defensible conclusion
is that attribution remained strong after explicit parent isolation on this
controlled benchmark, not that clinical generalisation was established.

## Reproduction

```bash
python scripts/build_reason_attribution_manifests.py --input datasets/dissertation_v1/manifests/test_ood_full.csv --out-dir datasets/dissertation_v1/manifests --split-mode grouped --output-prefix reason_grouped --seed 42
python scripts/audit_stage2_grouped.py --input datasets/dissertation_v1/manifests/test_ood_full.csv --train datasets/dissertation_v1/manifests/reason_grouped_train.csv --val datasets/dissertation_v1/manifests/reason_grouped_val.csv --test datasets/dissertation_v1/manifests/reason_grouped_test.csv --out-dir reports/stage2_grouped --seed 42
python scripts/generate_stage2_grouped_report.py --root-dir data --train-manifest datasets/dissertation_v1/manifests/reason_grouped_train.csv --val-manifest datasets/dissertation_v1/manifests/reason_grouped_val.csv --test-manifest datasets/dissertation_v1/manifests/reason_grouped_test.csv --legacy-dir reports/dissertation_results/reason_attribution_method_comparison --out-dir reports/stage2_grouped --figures-dir reports/dissertation_figures/stage2_grouped --seed 42
```

## Evidence and figures

- `reports/stage2_grouped/selected_models.json`
- `reports/stage2_grouped/family_metrics.csv`
- `reports/stage2_grouped/subtype_metrics.csv`
- `reports/stage2_grouped/predictions_test.csv`
- `reports/stage2_grouped/legacy_vs_grouped_comparison.csv`
- `reports/stage2_grouped/generation_provenance.json`
- `reports/dissertation_figures/stage2_grouped/`

## Remaining limitations

Parent grouping eliminates transformed-parent overlap between Stage 2
partitions, but related variants from each parent remain dependent within one
partition. The task is still supervised, closed-set, and synthetic-backed. It
does not establish patient-independent or device-independent clinical
validation. Reason labels are likely technical explanations for rejection, not
clinical diagnoses. The perfect family test confusion matrix should be reported
with the fixed 420-image controlled test size and not extrapolated to deployment.

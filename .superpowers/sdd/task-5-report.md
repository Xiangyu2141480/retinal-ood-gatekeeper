# Task 5 Report: Grouped Stage 2 Documentation and Evidence Index

## Scope completed

Canonical repository navigation, dissertation evidence indexes, manuscript
drafts, figure/table shortlists, captions, interpretation notes, result/model
summaries, and claims/limitations matrices now use the parent-grouped Stage 2
evaluation as headline evidence. The historical row-stratified experiment is
retained and explicitly labelled as legacy sensitivity evidence.

Stage 1 remains unchanged: it is the ID-only unsupervised FAF OOD gatekeeper
and alone decides acceptance or rejection. Stage 2 remains an optional,
supervised, OOD-labelled post-rejection explanation module. Stage 2 does not
use Stage 1 anomaly scores, is not a disease classifier, and its outputs are
likely technical rejection reasons rather than clinical diagnoses.

## Canonical grouped evidence

- Split sizes: train 1260, validation 420, test 420.
- Split seed: 42.
- Group rule: non-empty `parent_image_hash`, otherwise `image_path`.
- Cross-partition image-path overlap: 0 for train-validation, train-test,
  validation-test, and all-three scopes.
- Cross-partition group-ID overlap: 0 for the same four scopes.
- Variants sharing a parent remain dependent within their assigned split.

Family selection:

- method: `feature_statistics_fusion`;
- grouped validation accuracy: 0.9928571428571429;
- grouped validation macro-F1: 0.9924836011792534;
- grouped test accuracy: 1.0;
- grouped test macro-F1: 1.0;
- all three selected-model family F1 values are 1.0, so there is no unique
  hardest family.

Subtype selection:

- method: non-oracle `hierarchical_classifier` using predicted-family routing;
- grouped validation accuracy: 0.9238095238095239;
- grouped validation macro-F1: 0.9058507271445498;
- grouped test accuracy: 0.9357142857142857;
- grouped test macro-F1: 0.9175854801338393;
- hardest grouped-test subtype: `text_watermark`, F1 0.7741935483870969.

The family and subtype models were selected independently using grouped
validation macro-F1. Test results are retrospective measurements after both
selections were frozen.

## Legacy sensitivity comparison

Grouped-minus-legacy differences from
`reports/stage2_grouped/legacy_vs_grouped_comparison.csv`:

- selected family validation macro-F1: +0.001709138764976359;
- family test accuracy: +0.011904761904761973;
- family test macro-F1: +0.00992431328449861;
- selected subtype validation macro-F1: +0.03874672107835109;
- subtype test accuracy: +0.01190476190476164;
- subtype test macro-F1: +0.011641930651210664.

These increases describe sensitivity to one deterministic split allocation.
They do not establish patient-, device-, site-, or clinical generalisation and
do not imply that variants within a partition are independent.

## Audit artifact copies

Source and destination SHA-256 values matched byte-for-byte:

- `docs/dissertation/vincent_feedback_code_audit.md`:
  `917d9341bc3c3814b95963e1893ace916a2395182c840b433491ed30cf97a58a`;
- `reports/audit/parent_overlap_summary.csv`:
  `3057e4d76c93eca02f7c6120c3de33576ed64db163dd019ebca92ac1dfb107b7`;
- `reports/audit/parent_overlap_examples.csv`:
  `f1baba775d9648b05548e67326b16e0fcbb6cd71324b3b28c7fc202233928b64`.

## Current grouped output hashes

- `method_comparison.csv`:
  `97510658111c7a6c730d7b5928ef5c4996a05c964abb83e8b66ffb3233164c43`;
- `family_metrics.csv`:
  `0a12143dd42b16b1744429d592b5e12d959f485fd55f9bb128c1f758c2219c66`;
- `subtype_metrics.csv`:
  `eb415349ce6f5266aaedc6d3a68a111ad01cc45e925708a447ee6fb9dd27f0b0`;
- `selected_models.json`:
  `12f3f625a1f1891fd93f084f028c79f6af5db23481e9e395599d97d71406ab24`;
- `predictions_test.csv`:
  `e608bcfd115fbbe3134992c940d9ff933ff37278982e98a27b82b18b242f865d`;
- `legacy_vs_grouped_comparison.csv`:
  `67d62ffb4db0161907ecbabade0ff26d04915a389fca9a0c0da239abba2837ec`;
- `summary.md`:
  `ce0e9d3a3d53b1c565dd18512b26423b7ab0b7f4afe9a648c7b22e4f346aa3bc`.

The `family_metrics.csv` and `summary.md` hashes supersede the pre-review-fix
values recorded before commit `1498f73`; the selected methods and headline
metrics are unchanged.

## Stash safety

`stash@{0}` (`task5-docs-wip-before-task4-review-fix`) was inspected without
applying or dropping it. Its four files were already restored and then refined
in the working tree. Applying it would reintroduce less precise wording, so the
stash was retained until the documentation commit could be verified.

## Searches and validation

Searches covered non-legacy documentation for old headline values
`0.9881`, `0.9901`, `0.9238`, `0.9059`, and `0.8671`; legacy selected-model
names; final `reason_train.csv`/`reason_val.csv`/`reason_test.csv` references;
parent-overlap wording; and hardest-family claims. Old values remain only in
labelled legacy/pre-remediation audit contexts or as values that happen to
round to a valid grouped validation metric. No canonical document reports a
unique grouped-test hardest family.

Commands and outcomes:

```text
pytest -q tests/test_dissertation_delivery_bundle.py tests/test_dissertation_figure_cleanup.py tests/test_stage2_grouped_report.py
11 passed, 11 non-functional Matplotlib/Pyparsing deprecation warnings

ruff check .
All checks passed

git diff --check
Passed; Git only reported expected LF-to-CRLF working-copy warnings
```

The first documentation run exposed one stale assertion that still required
the legacy combined Stage 2 comparison in the main-text shortlist. The test was
updated to require the canonical grouped method-comparison figure; the focused
test then passed and the fix was committed separately as `e326327`.

## Self-review and remaining limitations

- The grouped protocol eliminates cross-partition parent/group overlap, not
  dependence among variants within a partition.
- The Stage 2 benchmark remains supervised, controlled, closed-set, and
  synthetic-backed.
- Parent grouping is not patient-independent clinical validation.
- The perfect family result is reported with its fixed 420-image test size and
  is not extrapolated to deployment.
- Legacy reports, figures, and row-level manifests remain unchanged for
  reproducibility and sensitivity comparison.

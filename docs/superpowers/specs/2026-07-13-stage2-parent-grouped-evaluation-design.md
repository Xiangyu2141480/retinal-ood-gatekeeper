# Stage 2 Parent-Grouped Evaluation Design

## Purpose

Replace the dissertation's headline Stage 2 row-stratified evaluation with a
deterministic parent-independent evaluation. Stage 1 remains unchanged: it is
an ID-only unsupervised OOD gatekeeper, and Stage 2 remains an optional,
supervised, closed-set explanation module invoked only after rejection.

## Repository Boundary

The implementation repository and dissertation source are separate Git
repositories. The GitHub pull request contains code, manifests, generated
results, figures, documentation, and an exact patch of the dissertation
changes. The authoritative `main.tex` is updated and compiled in an isolated
worktree of `D:\UCL-Dissertation`, whose remote is Overleaf rather than GitHub.
The two histories are not merged or rewritten.

## Split Design

Each row receives a normalized group identifier:

```python
group_id = parent_image_hash if parent_image_hash is non-empty else image_path
```

`None`, `NaN`, and whitespace-only strings are empty. Groups are stratified by
their complete subtype-count profile. This makes each eight-variant sensory
parent an atomic group while making modality-shift and semantic-outlier images
singleton groups. Within each profile, groups are sorted, deterministically
shuffled with seed 42 and a stable profile-derived offset, and allocated to
train/validation/test at 60/20/20. The current dataset therefore yields exact
subtype counts of 90/30/30 for each sensory subtype, 120/40/40 for each
modality subtype, and 300/100/100 for `cifar10_natural`.

The legacy `reason_train.csv`, `reason_val.csv`, and `reason_test.csv` remain
unchanged. New files use the `reason_grouped_{train,val,test}.csv` names.

## Evaluation Design

All existing required candidates are fitted on grouped training data only:

- `image_statistics_logreg`
- `global_feature_knn`
- `nearest_centroid`
- `logistic_regression`
- `linear_svm`
- `random_forest_or_gradient_boosting`
- `feature_statistics_fusion`
- `hierarchical_classifier`

The comparison is explicitly phased. First, every fitted method is evaluated
on grouped validation data. Family and subtype selections are frozen using
validation macro-F1, then existing complexity and method-name tie breakers.
Only after selection is frozen is grouped test evaluation performed. The
unknown threshold remains the existing pre-specified 0.5 policy, with its
validation trade-off curve reported; test data never chooses a method,
threshold, feature representation, or estimator setting.

Feature extraction remains image-only. Scalers and estimators are fitted by
scikit-learn pipelines on grouped training features only. Stage 1 scores and
manifest metadata are excluded. Hierarchical test-time subtype routing uses
the predicted family and is non-oracle.

## Outputs

The canonical package under `reports/stage2_grouped/` contains split audit
tables, method metrics, frozen selection metadata, selected-model test
predictions, confusion matrices, a legacy comparison, hashes, and a narrative
summary. Figures are generated under
`reports/dissertation_figures/stage2_grouped/` from the canonical CSV files.
No legacy report or figure is overwritten.

The dissertation adopts grouped results as its main Stage 2 evidence and keeps
row-level results only as a sensitivity comparison. It states that grouped
splitting removes parent reuse in this controlled synthetic-backed benchmark,
not that it establishes patient-independent or clinical generalisation.

## Verification

Tests cover group normalization, exact counts, zero path/group overlap,
eight-variant co-location, deterministic output, validation-only selection,
test-evaluation ordering, non-oracle routing, and report consistency. Data
validation checks file existence, corruption, duplicate content, and hashes.
The dissertation is compiled from its isolated Overleaf worktree and checked
for missing references, figures, and citations. Repository lint, tests,
hygiene audit, PR checks, and mergeability must all pass before squash merge.


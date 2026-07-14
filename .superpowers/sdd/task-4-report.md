# Task 4 Report: Parent-grouped Stage 2 experiment package

## Scope

- Runner: `scripts/generate_stage2_grouped_report.py`
- Tests: `tests/test_stage2_grouped_report.py`
- Results: `reports/stage2_grouped/`
- Figures: `reports/dissertation_figures/stage2_grouped/`

The runner resolves manifest values such as `images/...` relative to the
project `data` directory, so the effective image paths are
`data/images/dissertation_v1/...`. It does not look for a repository-root
`images/` directory.

## TDD evidence

Initial RED command:

```powershell
pytest -q tests/test_stage2_grouped_report.py
```

Result before implementation: `2 failed`; both failed because
`scripts/generate_stage2_grouped_report.py` did not exist.

The tie-aware regression was also observed RED before its fix:

```powershell
pytest -q tests/test_stage2_grouped_report.py::test_hardest_class_reports_a_tie_without_arbitrary_single_winner
```

Result: `1 failed`; the old helper returned arbitrary `alpha` instead of
`alpha, beta (tie)`.

Final focused verification:

```powershell
pytest -q tests/test_stage2_grouped_report.py \
  tests/test_reason_attribution_method_comparison.py \
  tests/test_stage2_grouped_audit.py \
  tests/test_reason_attribution_manifests.py
ruff check scripts/generate_stage2_grouped_report.py tests/test_stage2_grouped_report.py
git diff --check
```

Results: `31 passed`; Ruff passed; diff check passed. The only pytest output
was the existing Matplotlib/Pyparsing deprecation warning set.

## Real experiment command and runtime

```powershell
python scripts/generate_stage2_grouped_report.py `
  --root-dir data `
  --train-manifest datasets/dissertation_v1/manifests/reason_grouped_train.csv `
  --val-manifest datasets/dissertation_v1/manifests/reason_grouped_val.csv `
  --test-manifest datasets/dissertation_v1/manifests/reason_grouped_test.csv `
  --legacy-dir reports/dissertation_results/reason_attribution_method_comparison `
  --out-dir reports/stage2_grouped `
  --figures-dir reports/dissertation_figures/stage2_grouped `
  --seed 42
```

- Comparison settings: seed 42, unknown threshold 0.5, global image size 24,
  statistics image size 224.
- Required methods: all eight specified methods; `rbf_svm_optional` absent.
- Measured package runtime: 82.26890459982678 seconds (shell wall time 85.3 s).
- Evaluation order: fit all methods, evaluate all validation partitions,
  freeze both selected models, then evaluate test.

## Parent isolation and split sizes

- Train/validation/test rows: 1260 / 420 / 420.
- Train-validation image paths/groups: 0 / 0 shared.
- Train-test image paths/groups: 0 / 0 shared.
- Validation-test image paths/groups: 0 / 0 shared.
- All-three image paths/groups: 0 / 0 shared.
- Test predictions: 420 rows.
- Family confusion total: 420.
- Subtype confusion total: 420.

Grouped manifest SHA-256:

- Train: `7c4ccfce72d1e169351633c341af68b69e1e65a6a9c28b9cb2f45dfe243a4b55`
- Validation: `2250c2694b6f8479523b7fce62cf63476cb0881abb20878946a1b76194644f61`
- Test: `0f53228034975967e1ed15dba214e71bc89e2d770ea0e8f087f10aaaef059367`

## Validation-selected methods and grouped metrics

Family winner: `feature_statistics_fusion`.

- Validation accuracy: 0.9928571428571429.
- Validation macro-F1: 0.9924836011792534.
- Test accuracy: 1.0.
- Test macro-F1: 1.0.
- All three family test F1 scores are 1.0; there is no unique hardest family.

Subtype winner: `hierarchical_classifier`.

- Validation accuracy: 0.9238095238095239.
- Validation macro-F1: 0.9058507271445498.
- Test accuracy: 0.9357142857142857.
- Test macro-F1: 0.9175854801338393.
- Hardest subtype: `text_watermark`, test F1 0.7741935483870969.

The hierarchical classifier routes subtype predictions using its predicted
family, not ground-truth family. Stage 1 scores are not Stage 2 inputs.

## Legacy sensitivity comparison

Differences below are grouped minus legacy and were calculated from source
CSV values without rounded constants:

- Selected family validation macro-F1: +0.001709138764976359.
- Family test accuracy: +0.011904761904761973.
- Family test macro-F1: +0.00992431328449861.
- Selected subtype validation macro-F1: +0.03874672107835109.
- Subtype test accuracy: +0.01190476190476164.
- Subtype test macro-F1: +0.011641930651210664.

Every legacy result file was SHA-256 snapshotted before and after generation.
The full before/after dictionaries in `generation_provenance.json` are equal.

## Key output hashes

- `method_comparison.csv`: `97510658111c7a6c730d7b5928ef5c4996a05c964abb83e8b66ffb3233164c43`
- `family_metrics.csv`: `0a12143dd42b16b1744429d592b5e12d959f485fd55f9bb128c1f758c2219c66`
- `subtype_metrics.csv`: `eb415349ce6f5266aaedc6d3a68a111ad01cc45e925708a447ee6fb9dd27f0b0`
- `selected_models.json`: `12f3f625a1f1891fd93f084f028c79f6af5db23481e9e395599d97d71406ab24`
- `predictions_test.csv`: `e608bcfd115fbbe3134992c940d9ff933ff37278982e98a27b82b18b242f865d`
- `legacy_vs_grouped_comparison.csv`: `67d62ffb4db0161907ecbabade0ff26d04915a389fca9a0c0da239abba2837ec`
- `summary.md`: `ce0e9d3a3d53b1c565dd18512b26423b7ab0b7f4afe9a648c7b22e4f346aa3bc`

## Figures

Stable PNG and PDF pairs were generated for:

- grouped family confusion matrix;
- grouped subtype confusion matrix;
- grouped family/subtype method comparison;
- legacy versus grouped metric comparison.

Visual QA confirmed white backgrounds, readable labels, non-overlapping
legends, and stable filenames. Family confusion display labels use spaces and
line breaks while the source CSV retains canonical underscore names.

## Commits

1. `c4d6e21` `test: add grouped stage2 report invariants`
2. `9e37d95` `feat: rerun stage2 attribution on grouped splits`

## Self-review and concerns

- Family and subtype winners are independently selected on grouped validation
  macro-F1; the perfect family test result did not participate in selection.
- Full-precision source values are retained in CSV/JSON and the Markdown
  evidence summary. Figure labels use presentation rounding only.
- The grouped score increase over legacy is plausible because the deterministic
  parent allocation changed which parent groups appear in each partition; it
  is not evidence of clinical generalisation.
- The evaluation remains controlled, closed-set, synthetic-backed, supervised
  Stage 2 attribution. Stage 1 remains the unchanged ID-only unsupervised OOD
  gatekeeper.

## Review fix: aligned grouped evidence artifacts

Review found that the canonical summary was tie-aware but the compatibility
summary and selection-summary figure still used a first-row `idxmin` result.
The fix adds a regression for the compatibility helper and makes tied minima
explicit while preserving the original unique-minimum format for legacy runs.

Additional review fixes:

- `best_method_summary.csv` and `.md` now state
  `no unique hardest; modality_shift, semantic_outlier, sensory_artifact tied
  (F1=1.0000)`;
- `figure_reason_method_selection_summary.png` carries the same tie-aware
  wording and was visually checked for clipping;
- canonical limitations now state that grouping eliminates cross-partition
  parent overlap while variants remain dependent within a split;
- grouped `figure_index.md` now lists all four primary PNG/PDF pairs (eight
  rows) and labels compatibility figures separately;
- real grouped outputs were regenerated with the exact Task 4 command; selected
  methods and headline metrics were unchanged;
- legacy before/after hash dictionaries remain equal.

Review-fix RED command:

```powershell
pytest -q \
  tests/test_reason_attribution_method_comparison.py::test_compatibility_summary_reports_tied_hardest_classes_without_idxmin_bias \
  tests/test_stage2_grouped_report.py::test_report_package_uses_generated_evidence_and_preserves_legacy \
  tests/test_stage2_grouped_report.py::test_committed_grouped_outputs_are_complete_and_consistent
```

RED result: `3 failed`; one failure for each review finding. Final expanded
focused gate: `32 passed`, Ruff passed, and `git diff --check` passed. Existing
Matplotlib/Pyparsing deprecation warnings remain non-functional.

Review-fix commit: `1498f73` `fix: align grouped stage2 evidence artifacts`.

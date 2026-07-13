# Stage 2 Parent-Grouped Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic parent-grouped Stage 2 protocol, rerun every attribution candidate without test-set selection, and replace the dissertation's headline Stage 2 evidence with the grouped results while preserving legacy outputs.

**Architecture:** A focused grouped-split module supplies normalized group IDs, deterministic profile-stratified allocation, and overlap audits. The existing comparison engine is changed to freeze validation selections before any test evaluation and to emit traceable predictions and selection metadata. A thin grouped-report command assembles canonical CSV/JSON/Markdown outputs and figures; the Overleaf source is updated in its separate worktree and represented in the GitHub branch by an exact patch and evidence notes.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, Pillow, Matplotlib, pytest, Ruff, LaTeX/latexmk-compatible tooling, Git, GitHub CLI.

## Global Constraints

- Branch: `codex/stage2-parent-grouped-evaluation`, based on latest `origin/main`.
- Seed: `42`; train/validation/test ratios: `0.6/0.2/0.2`.
- Group ID: non-empty `parent_image_hash`, otherwise `image_path`; `None`, `NaN`, and whitespace are empty.
- Legacy `reason_train.csv`, `reason_val.csv`, `reason_test.csv` and legacy reports remain unchanged.
- New manifests: `reason_grouped_train.csv`, `reason_grouped_val.csv`, `reason_grouped_test.csv`.
- Stage 1 code, manifests, thresholds, and results must not change.
- Stage 2 uses OOD-only labels for optional post-rejection explanation; it is not disease classification.
- No test-set model, threshold, hyperparameter, or feature selection.
- No metadata or Stage 1 anomaly score enters Stage 2 feature matrices.
- Hierarchical subtype routing uses predicted family, never ground-truth family.
- Do not claim patient-independent or clinical generalisation.
- Do not commit checkpoints, caches, LaTeX auxiliaries, private paths, patient identifiers, or unrelated changes.

---

### Task 1: Parent-Grouped Split Invariants

**Files:**
- Modify: `tests/test_reason_attribution_manifests.py`
- Create: `src/retinal_ood/reason_attribution/grouped_split.py`
- Modify: `scripts/build_reason_attribution_manifests.py`

**Interfaces:**
- Produces: `normalize_group_id(parent_image_hash, image_path) -> str` and `parent_grouped_stratified_split(frame, seed, train_ratio, val_ratio, test_ratio) -> dict[str, pd.DataFrame]`.
- Preserves: `build_reason_manifests(...)` row-stratified default behavior.

- [ ] **Step 1: Add failing group-normalization tests**

Add parameterized assertions covering a real parent hash, empty string,
whitespace, `None`, and `numpy.nan`. Sensory rows must resolve to the parent;
singleton OOD rows must resolve to `image_path`.

- [ ] **Step 2: Add failing grouped-split integration tests**

Construct 150 synthetic parent groups with all eight sensory subtypes plus
200/200/500 singleton rows for the other subtypes. Assert 1260/420/420 rows,
the requested subtype table, zero pairwise image-path and group overlap, all
eight variants co-located, and identical CSV bytes across two seed-42 runs.

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest -q tests/test_reason_attribution_manifests.py -k "group or grouped"
```

Expected: failure because grouped APIs and grouped output names do not exist.

- [ ] **Step 4: Implement minimal grouped splitting**

Create a small module that normalizes group IDs, derives a stable subtype-count
profile per group, shuffles groups deterministically within profiles, and
allocates complete groups using exact ratio counts when divisible. Extend the
builder with `split_mode="row"|"grouped"` and `output_prefix`, plus CLI flags
`--split-mode` and `--output-prefix`; keep row mode and legacy names as defaults.

- [ ] **Step 5: Run GREEN tests and regression tests**

Run:

```bash
pytest -q tests/test_reason_attribution_manifests.py
```

Expected: all manifest tests pass, including unchanged legacy behavior.

- [ ] **Step 6: Commit**

```bash
git add tests/test_reason_attribution_manifests.py src/retinal_ood/reason_attribution/grouped_split.py scripts/build_reason_attribution_manifests.py
git commit -m "feat: add deterministic parent-grouped reason splits"
```

### Task 2: Grouped Split Audit and Final Manifests

**Files:**
- Create: `scripts/audit_stage2_grouped.py`
- Create: `tests/test_stage2_grouped_audit.py`
- Create: `datasets/dissertation_v1/manifests/reason_grouped_train.csv`
- Create: `datasets/dissertation_v1/manifests/reason_grouped_val.csv`
- Create: `datasets/dissertation_v1/manifests/reason_grouped_test.csv`
- Create: `reports/stage2_grouped/split_summary.csv`
- Create: `reports/stage2_grouped/group_overlap_summary.csv`
- Create: `reports/stage2_grouped/subtype_distribution.csv`
- Create: `reports/stage2_grouped/split_audit.md`

**Interfaces:**
- Consumes: grouped split APIs from Task 1.
- Produces: `audit_grouped_manifests(...)` and content/hash evidence used by later reports.

- [ ] **Step 1: Add failing audit tests**

Assert split/family/subtype counts, unique groups, zero path/group overlap,
SHA-256 fields for input/output manifests, seed 42, algorithm name, and an
explicit failure when a group crosses partitions.

- [ ] **Step 2: Run RED tests**

Run:

```bash
pytest -q tests/test_stage2_grouped_audit.py
```

Expected: failure because the audit script does not exist.

- [ ] **Step 3: Implement the audit command**

Read only the supplied manifests, recompute normalized group IDs, validate
pairwise disjointness, and write deterministic CSV/Markdown reports with input
and output SHA-256 hashes. The script must exit non-zero on any overlap or
distribution mismatch.

- [ ] **Step 4: Generate final manifests and audits**

Run:

```bash
python scripts/build_reason_attribution_manifests.py --input datasets/dissertation_v1/manifests/test_ood_full.csv --out-dir datasets/dissertation_v1/manifests --seed 42 --split-mode grouped --output-prefix reason_grouped --reports-dir reports/stage2_grouped
python scripts/audit_stage2_grouped.py --input datasets/dissertation_v1/manifests/test_ood_full.csv --train datasets/dissertation_v1/manifests/reason_grouped_train.csv --val datasets/dissertation_v1/manifests/reason_grouped_val.csv --test datasets/dissertation_v1/manifests/reason_grouped_test.csv --out-dir reports/stage2_grouped --seed 42
```

Expected: exact 1260/420/420 rows and zero shared paths/groups.

- [ ] **Step 5: Verify tests and reproducibility**

Run the builder twice into temporary directories and compare SHA-256 hashes;
then run `pytest -q tests/test_stage2_grouped_audit.py`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit_stage2_grouped.py tests/test_stage2_grouped_audit.py datasets/dissertation_v1/manifests/reason_grouped_*.csv reports/stage2_grouped/split_summary.csv reports/stage2_grouped/group_overlap_summary.csv reports/stage2_grouped/subtype_distribution.csv reports/stage2_grouped/split_audit.md
git commit -m "feat: add audited parent-independent stage2 manifests"
```

### Task 3: Validation-First Comparison and Isolation Tests

**Files:**
- Modify: `src/retinal_ood/reason_attribution/comparison.py`
- Modify: `tests/test_reason_attribution_method_comparison.py`

**Interfaces:**
- Produces: a comparison result that records frozen validation selections, test-evaluation phase order, selected test predictions, and canonical JSON/CSV artifacts.
- Preserves: existing candidates, features, tie-breaking, and legacy CLI defaults.

- [ ] **Step 1: Add failing phase-order and isolation tests**

Instrument prediction calls to assert all validation predictions occur before
the first test prediction, selection is made only from `split="val"`, scaler
and estimators fit only training rows, metadata/Stage 1 scores remain excluded,
and hierarchical routing uses predicted family.

- [ ] **Step 2: Add failing output-consistency tests**

Require `selected_models.json`, `predictions_test.csv`, canonical family and
subtype metrics, and confusion matrices whose totals equal test size. Assert
the selected method names match validation comparison rows and JSON records
the pre-specified unknown threshold.

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest -q tests/test_reason_attribution_method_comparison.py
```

Expected: failures for phase ordering and missing canonical outputs.

- [ ] **Step 4: Refactor to validation then test phases**

Fit all candidates from training features, evaluate all candidates on
validation, freeze family/subtype selection, then evaluate all fitted
candidates on test. Write selected-model metadata and selected test
predictions without encoding metadata as features. Keep threshold 0.5 as the
pre-specified policy and report its validation curve.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
pytest -q tests/test_reason_attribution_method_comparison.py
```

Expected: all comparison and leakage tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/retinal_ood/reason_attribution/comparison.py tests/test_reason_attribution_method_comparison.py
git commit -m "test: enforce validation-first stage2 selection"
```

### Task 4: Grouped Experiment Package and Figures

**Files:**
- Create: `scripts/generate_stage2_grouped_report.py`
- Create: `tests/test_stage2_grouped_report.py`
- Create/Update: `reports/stage2_grouped/*`
- Create: `reports/dissertation_figures/stage2_grouped/*`

**Interfaces:**
- Consumes: canonical comparison files and legacy committed metrics.
- Produces: required grouped output names, legacy comparison, summary, and traceable figures.

- [ ] **Step 1: Add failing package tests**

Assert the required files exist, legacy differences equal grouped minus legacy,
figures read canonical CSVs, selection JSON agrees with validation metrics,
test prediction count is 420, and family/subtype confusion totals each equal
420.

- [ ] **Step 2: Run RED tests**

Run:

```bash
pytest -q tests/test_stage2_grouped_report.py
```

Expected: failure because grouped package generation is absent.

- [ ] **Step 3: Implement the report generator**

Run all eight required candidates against grouped manifests, produce
`method_comparison.csv`, `family_metrics.csv`, `subtype_metrics.csv`,
`selected_models.json`, `predictions_test.csv`, both confusion matrices,
`legacy_vs_grouped_comparison.csv`, and `summary.md`. Generate grouped family
and subtype confusion matrices, grouped method comparison, and legacy-versus-
grouped comparison from CSV inputs in the existing academic style.

- [ ] **Step 4: Execute the full grouped experiment**

Run:

```bash
python scripts/generate_stage2_grouped_report.py --root-dir data --train-manifest datasets/dissertation_v1/manifests/reason_grouped_train.csv --val-manifest datasets/dissertation_v1/manifests/reason_grouped_val.csv --test-manifest datasets/dissertation_v1/manifests/reason_grouped_test.csv --legacy-dir reports/dissertation_results/reason_attribution_method_comparison --out-dir reports/stage2_grouped --figures-dir reports/dissertation_figures/stage2_grouped --seed 42
```

Record wall-clock output and actual selected methods without assuming legacy
winners remain selected.

- [ ] **Step 5: Verify package consistency**

Run `pytest -q tests/test_stage2_grouped_report.py` and manually inspect PNGs
for readable labels, white backgrounds, and non-empty content.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_stage2_grouped_report.py tests/test_stage2_grouped_report.py reports/stage2_grouped reports/dissertation_figures/stage2_grouped
git commit -m "feat: rerun stage2 attribution on grouped splits"
```

### Task 5: Repository Documentation and Evidence

**Files:**
- Create: `docs/experiments/reason_attribution_parent_grouped.md`
- Modify: `docs/dissertation/README.md`
- Modify: `docs/dissertation/final_evidence_index.md`
- Modify: `README.md`
- Modify: affected `docs/dissertation/manuscript_draft/*.md`
- Modify: affected `reports/dissertation_final/*`
- Copy: `docs/dissertation/vincent_feedback_code_audit.md`
- Copy: `reports/audit/parent_overlap_summary.csv`
- Copy: `reports/audit/parent_overlap_examples.csv`

**Interfaces:**
- Consumes: actual grouped CSV/JSON metrics from Task 4.
- Produces: final repository narrative and clean-environment reproduction commands.

- [ ] **Step 1: Import the completed audit artifacts mechanically**

Copy the three existing untracked audit outputs from `D:\UCL-Dissertation`
into the corresponding GitHub paths without changing their contents.

- [ ] **Step 2: Update documentation from generated metrics**

Describe the overlap motivation, algorithm, seed, exact commands, grouped
manifests, validation-selected methods, grouped metrics, legacy comparison,
and remaining synthetic/closed-set limitations. Replace headline legacy
Stage 2 claims while retaining legacy numbers only as sensitivity evidence.

- [ ] **Step 3: Add reproducibility and evidence links**

Provide a clean sequence that rebuilds grouped manifests, audits, experiments,
and figures. Ensure every numeric claim links to CSV/JSON evidence and that
Stage 1 remains explicitly unchanged.

- [ ] **Step 4: Run documentation consistency searches**

Search all Markdown/CSV/JSON for legacy Stage 2 headline numbers and future-
work claims that grouped splitting is unimplemented; retain only labelled
legacy-comparison occurrences.

- [ ] **Step 5: Commit**

```bash
git add README.md docs reports/dissertation_final reports/audit
git commit -m "docs: report parent-grouped stage2 evaluation"
```

### Task 6: Dissertation Source Update and Compile

**Files:**
- Modify in Overleaf worktree: `main.tex`
- Update/copy in Overleaf worktree: relevant grouped figures
- Create in GitHub worktree: `docs/dissertation/overleaf_stage2_grouped_main_tex.patch`
- Create in GitHub worktree: `docs/dissertation/overleaf_stage2_grouped_build.md`

**Interfaces:**
- Consumes: grouped CSV/JSON and figures from Task 4.
- Produces: compiled dissertation and exact GitHub-reviewable paper patch.

- [ ] **Step 1: Update dataset/split methodology**

Revise Section 4.2/Table 3, Sections 4.6-4.7, and Stage 2 methodology with
group definition, 60/20/20 counts, seed 42, validation-only selection,
predicted-family routing, metadata exclusions, and Stage 1 boundary.

- [ ] **Step 2: Update results and figures from generated evidence**

Replace headline Stage 2 tables, prose, captions, discussion, conclusion, and
all other old metric occurrences with actual grouped values. Label legacy
values only in sensitivity comparison and discuss affected classes from the
grouped confusion matrices.

- [ ] **Step 3: Preserve limitations accurately**

State that parent overlap is eliminated for this benchmark, while the Stage 2
task remains supervised, closed-set, synthetic-backed, non-clinical, and not
patient-independent clinical validation. State that Stage 1 was never affected.

- [ ] **Step 4: Compile and inspect**

Run the available `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
compatible command. Fail on undefined references, missing figures, missing
citations, or compile errors; inspect Table 3 and Stage 2 pages for overflow.

- [ ] **Step 5: Export exact patch and build evidence**

Generate a unified `main.tex` diff into the GitHub worktree and write the build
command/result. Do not commit LaTeX auxiliary files.

- [ ] **Step 6: Commit both histories**

Commit the Overleaf worktree on its isolated feature branch, then commit the
reviewable patch/build evidence in the GitHub branch:

```bash
git add docs/dissertation/overleaf_stage2_grouped_main_tex.patch docs/dissertation/overleaf_stage2_grouped_build.md
git commit -m "docs: update dissertation with parent-grouped stage2 results"
```

### Task 7: Full Validation, Hygiene, and Review

**Files:**
- Modify only files required to fix validation or review findings.

**Interfaces:**
- Verifies all prior tasks and produces merge evidence.

- [ ] **Step 1: Run targeted tests**

```bash
pytest -q tests/test_reason_attribution_manifests.py tests/test_reason_attribution_method_comparison.py tests/test_stage2_grouped_audit.py tests/test_stage2_grouped_report.py
```

- [ ] **Step 2: Run related and full tests**

```bash
pytest -q tests/test_dataset_schema.py tests/test_dissertation_dataset_builder.py tests/test_reason_attribution_manifests.py tests/test_reason_attribution_method_comparison.py tests/test_lightweight_baselines.py tests/test_patchcore.py tests/test_autoencoder.py tests/test_stage2_grouped_audit.py tests/test_stage2_grouped_report.py
pytest -q
```

- [ ] **Step 3: Validate manifests and images**

Run `scripts/validate_manifests.py`, `scripts/audit_dataset_images.py` with
corrupt/duplicate-content failure gates, and the grouped audit command.

- [ ] **Step 4: Run lint and repository audit**

```bash
ruff check .
git diff --check origin/main...HEAD
python scripts/final_repository_audit.py
```

- [ ] **Step 5: Run hygiene scans**

Check private paths, patient identifiers, secrets, checkpoints, tracked
`runs/`, caches, LaTeX auxiliaries, and unexpected large files. Confirm Stage 1
guard-file hashes match the pre-run snapshot.

- [ ] **Step 6: Review the full diff**

Check zero parent overlap, validation-before-test sequencing, no test-set
selection, no overwritten legacy artifacts, no stale headline metrics, and no
clinical overclaim. Fix every Critical/Important finding and rerun covering
tests.

### Task 8: PR, CI, and Squash Merge

**Files:**
- No repository file changes unless CI/review requires a tested fix.

**Interfaces:**
- Produces: GitHub PR and verified squash commit on `main`.

- [ ] **Step 1: Push and open PR**

Push `codex/stage2-parent-grouped-evaluation` and create a PR titled
`Add parent-grouped Stage 2 evaluation and update dissertation results` with
summary, zero-overlap evidence, real legacy/grouped metrics, interpretation,
Stage 1 boundary, limitations, and validation output.

- [ ] **Step 2: Review remote diff and comments**

Confirm only task-related files, correct base/head branches, no conflicts, and
no unresolved review threads.

- [ ] **Step 3: Wait for required checks**

Run `gh pr checks --watch`; diagnose and fix failures without bypassing branch
protection or force-pushing `main`.

- [ ] **Step 4: Squash merge and delete branch**

When all local and remote gates pass, run:

```bash
gh pr merge --squash --delete-branch
```

- [ ] **Step 5: Verify merged main**

Fetch and update local `main`, record the squash hash, and verify grouped
manifests, reports, figures, and paper patch are present in the merge commit.

## Self-Review

- Every requirement in the supplied specification maps to Tasks 1-8.
- No placeholders or deferred implementation steps remain.
- The group-ID, file-name, command, method-name, and metric interfaces are
  consistent across tasks.
- The plan preserves separate Git histories while making the dissertation
  changes exactly reviewable from the GitHub PR.


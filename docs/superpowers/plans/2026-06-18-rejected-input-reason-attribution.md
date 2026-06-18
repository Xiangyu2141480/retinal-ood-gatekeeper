# Rejected-Input Reason Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional second-stage module that explains the likely reason for an input already rejected by the Stage 1 ID-only OOD gatekeeper.

**Architecture:** Stage 1 remains unchanged: an unsupervised binary FAF OOD gatekeeper trained only on ID rows. Phase 2 adds separate reason manifests, frozen image-feature extraction, lightweight supervised classifiers for OOD reason family and optional subtype, post-hoc evaluation/reporting, figures, docs, and a demo that invokes Stage 2 only after rejection.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, PyTorch/Torchvision feature extractors already present in `retinal_ood`, matplotlib, pytest, ruff.

---

### Task 1: Reason Attribution Manifests

**Files:**
- Create: `scripts/build_reason_attribution_manifests.py`
- Test: `tests/test_reason_attribution_manifests.py`

- [ ] Write tests proving deterministic stratified splits, OOD-only rows, explicit `ood_type` and `ood_subtype`, private-path rejection, and removal/rejection of patient/clinical identifier columns.
- [ ] Implement manifest validation and stratified subtype splitting from `test_ood_full.csv`.
- [ ] Preserve safe metadata columns and write `reason_train.csv`, `reason_val.csv`, `reason_test.csv`.
- [ ] Write `reason_manifest_audit.md` and `.csv` under `reports/dissertation_results/reason_attribution/`.

### Task 2: Image-Derived Features and Classifier

**Files:**
- Create: `src/retinal_ood/reason_attribution/__init__.py`
- Create: `src/retinal_ood/reason_attribution/features.py`
- Create: `src/retinal_ood/reason_attribution/classifier.py`
- Test: `tests/test_reason_attribution_classifier.py`

- [ ] Write tests for image-derived feature extraction with injectable feature extractors.
- [ ] Write tests proving metadata columns are not accepted as input features.
- [ ] Write tests for family classifier training, optional subtype classifier behavior, metrics generation, confusion matrix shape, and `unknown_ood` thresholding.
- [ ] Implement frozen global-pooled feature extraction as `z(x) = g(f_psi(x))`.
- [ ] Implement logistic-regression classifier with deterministic seed and nearest-centroid fallback.
- [ ] Implement save/load using compact JSON/NPZ artifacts that are generated outputs, not committed model checkpoints.

### Task 3: Training, Evaluation, Demo CLIs

**Files:**
- Create: `scripts/train_reason_attribution.py`
- Create: `scripts/evaluate_reason_attribution.py`
- Create: `scripts/run_two_stage_demo.py`

- [ ] Add `--help`-friendly CLIs matching the requested commands.
- [ ] Train family classifier from `reason_train.csv` and `reason_val.csv`; train subtype classifier only with `--train-subtype-classifier`.
- [ ] Evaluate on `reason_test.csv`, writing family/subtype metrics, confusion matrices, predictions, unknown-threshold sweep, and end-to-end caveat files.
- [ ] Implement demo JSON output that returns Stage 2 explanation only when Stage 1 decision is `REJECT`.

### Task 4: Figures and Dissertation Docs

**Files:**
- Create: `reports/dissertation_figures/reason_attribution/figure_two_stage_pipeline.png`
- Create: `reports/dissertation_figures/reason_attribution/figure_reason_family_confusion_matrix.png`
- Create: `reports/dissertation_figures/reason_attribution/figure_reason_subtype_confusion_matrix.png`
- Create: `reports/dissertation_figures/reason_attribution/figure_reason_attribution_examples.png`
- Create: `reports/dissertation_figures/reason_attribution/figure_reason_unknown_threshold.png`
- Create: `reports/dissertation_figures/reason_attribution/figure_index.md`
- Create: `reports/dissertation_figures/reason_attribution/caption_suggestions.md`
- Create: `reports/dissertation_figures/reason_attribution/figure_selection_guide.md`
- Create: `docs/experiments/reason_attribution_module.md`
- Create: `docs/dissertation/manuscript_draft/reason_attribution_extension.md`

- [ ] Generate clean white-background academic figures from compact CSV/prediction outputs.
- [ ] Document the exact Stage 1/Stage 2 boundary and required wording: “The first-stage gatekeeper remains an ID-only unsupervised OOD detector. The second-stage reason attribution module is a post-hoc supervised explanation layer applied only after rejection.”
- [ ] State limitations: supervised explanation, not clinical diagnosis, known OOD labels only, low-confidence `unknown_ood`, not deployment-ready.

### Task 5: Experiment, Validation, PR

**Files:**
- Generated/committed compact outputs under `datasets/dissertation_v1/manifests/`, `reports/dissertation_results/reason_attribution/`, and `reports/dissertation_figures/reason_attribution/`.

- [ ] Run manifest builder, train/evaluate scripts, and figure generation on dataset v1.
- [ ] Run `pip install -e ".[dev]"`, `pytest -q`, `ruff check .`, `git diff --check`, CLI `--help` checks, and hygiene scans.
- [ ] Stage only intended files; do not use `git add .`.
- [ ] Commit with `feat: add rejected-input reason attribution module`.
- [ ] Push `codex/rejected-input-reason-attribution` and open PR into `main` with actual metrics and safety notes.

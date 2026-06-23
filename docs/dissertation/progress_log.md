# Dissertation Project Progress Log

This is the progressive task-and-conclusion log requested for dissertation supervision. Each
entry records what was completed, the result that matters, and where the supporting evidence is
stored. Future completed tasks should add a short factual entry using the same fields.

## 1. Dataset Preparation

- **Purpose:** Create the manifest-driven ID/OOD dataset required for the gatekeeper study.
- **Work completed:** Packaged 3,100 curated images through Git LFS, including 1,000 synthetic FAF
  ID images and 2,100 OOD images across three families and 11 subtypes.
- **Key result:** The final Stage 1 splits contain 700 train ID, 150 validation ID, 150 synthetic
  fallback test ID, and 2,100 full OOD rows.
- **Main conclusion:** The dataset supports reproducible ID-only fitting and structured OOD
  evaluation, but not real clinical FAF validation.
- **Evidence path:** [`datasets/dissertation_v1/`](../../datasets/dissertation_v1/),
  [`dataset documentation`](../datasets/dissertation_dataset_v1.md).
- **Status:** Complete.

## 2. Dataset Audit and Validation

- **Purpose:** Check manifest structure, image integrity, split safety, privacy fields, and
  reproducible packaging.
- **Work completed:** Added manifest validation, image audit, checksums, Git LFS checks, and
  repository safety rules.
- **Key result:** Stage 1 train and validation manifests contain ID rows only; patient identifiers
  and prohibited clinical target fields are absent from the committed package.
- **Main conclusion:** Dataset v1 satisfies the repository's proof-of-concept integrity and hygiene
  gates.
- **Evidence path:** [`dataset card`](../datasets/dissertation_dataset_card_v1.md),
  [`validation instructions`](../datasets/dissertation_dataset_v1.md).
- **Status:** Complete.

## 3. Stage 1 Baseline Implementation

- **Purpose:** Establish diverse unsupervised baselines for binary FAF input gatekeeping.
- **Work completed:** Implemented image statistics, autoencoder reconstruction, global feature kNN,
  Mahalanobis feature distance, and PatchCore variants.
- **Key result:** Eight Stage 1 configurations can be fitted using ID data and evaluated under a
  common reporting pipeline.
- **Main conclusion:** The project compares low-level, reconstruction, global-distance, and
  patch-level approaches without using OOD labels for Stage 1 fitting.
- **Evidence path:** [`scheme overview`](../../reports/dissertation_results/multi_scheme_comparison/scheme_overview.md).
- **Status:** Complete.

## 4. Stage 1 Multi-Method Comparison

- **Purpose:** Identify the strongest quantitative gatekeeper on the final benchmark.
- **Work completed:** Evaluated all eight configurations using AUROC, AUPRC, FPR@95%TPR, threshold
  recall, and category-level breakdowns.
- **Key result:** Mahalanobis achieved AUROC 0.9724, AUPRC 0.9975, and FPR@95%TPR 0.2000.
- **Main conclusion:** Mahalanobis feature distance is the primary quantitative Stage 1 method.
- **Evidence path:** [`metrics by scheme`](../../reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.md).
- **Status:** Complete.

## 5. PatchCore Layer Ablation

- **Purpose:** Select the most useful PatchCore feature layer for local anomaly evidence.
- **Work completed:** Compared L2, L3, and L2+L3 using ranking and safety metrics.
- **Key result:** PatchCore L3 achieved the strongest PatchCore AUROC (0.8819) and AUPRC (0.9880).
- **Main conclusion:** PatchCore L3 is the selected localisation-oriented companion, not the
  primary quantitative gatekeeper.
- **Evidence path:** [`layer ablation table`](../../reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.md).
- **Status:** Complete.

## 6. Threshold-Policy Analysis

- **Purpose:** Distinguish research thresholds from deployable ID-calibrated policies.
- **Work completed:** Compared research 95% TPR, validation-ID quantiles, and fixed ID rejection
  policies.
- **Key result:** Mahalanobis q95 gives ID false rejection 0.0467 and OOD recall 0.9079.
- **Main conclusion:** Threshold selection is an explicit safety trade-off; q95 is the balanced
  prototype policy, while real clinical calibration remains future work.
- **Evidence path:** [`threshold policy sweep`](../../reports/dissertation_results/robustness_analysis/threshold_policy_sweep.md).
- **Status:** Complete.

## 7. Robustness and Failure Analysis

- **Purpose:** Test uncertainty, data sensitivity, artifact severity, disagreements, failure modes,
  feature structure, and resource behaviour.
- **Work completed:** Added bootstrap intervals, train-size sensitivity, severity stress tests,
  disagreement/failure cases, PCA, subtype influence, and runtime/resource summaries.
- **Key result:** Mahalanobis remains strongest for aggregate quantitative performance; text
  watermark dominates its false negatives, and PatchCore catches many of those misses.
- **Main conclusion:** The primary ranking is stable on the benchmark, but subtle local artifacts
  and threshold-dependent safety remain important limitations.
- **Evidence path:** [`robustness takeaway table`](../../reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md).
- **Status:** Complete.

## 8. Stage 2 Reason Attribution

- **Purpose:** Test whether a rejected input can receive a likely, non-diagnostic explanation.
- **Work completed:** Built OOD-only reason train/validation/test manifests and a supervised
  post-rejection family/subtype attribution module.
- **Key result:** Stage 2 runs after `REJECT` and does not change or train the Stage 1 decision.
- **Main conclusion:** Reason attribution is a useful optional extension but remains conceptually
  separate from unsupervised OOD gatekeeping.
- **Evidence path:** [`reason-attribution experiment`](../experiments/reason_attribution_method_comparison.md).
- **Status:** Complete.

## 9. Stage 2 Multi-Method Comparison

- **Purpose:** Select family and subtype attribution methods and run leakage/sanity checks.
- **Work completed:** Compared image-statistics, global, fusion, tree, linear, nearest-centroid,
  and hierarchical approaches.
- **Key result:** Linear SVM family accuracy is 0.9881 with macro-F1 0.9901; the non-oracle
  hierarchical classifier subtype accuracy is 0.9238 with macro-F1 0.9059.
- **Main conclusion:** Stage 2 is strong on the current controlled splits, but parent-image-hash
  overlap may make parent-independent generalisation optimistic.
- **Evidence path:** [`best method summary`](../../reports/dissertation_results/reason_attribution_method_comparison/best_method_summary.md),
  [`leakage sanity check`](../../reports/dissertation_results/reason_attribution_method_comparison/leakage_sanity_check.md).
- **Status:** Complete.

## 10. Final Figure and Reporting Polish

- **Purpose:** Make the completed evidence directly usable in a dissertation.
- **Work completed:** Standardised academic plotting style, refined pipelines and comparisons,
  separated opposing metric directions, added normalised confusion matrices, and updated captions
  and figure shortlists.
- **Key result:** A concise set of stable, high-resolution dissertation figures is indexed with
  suggested chapters, placement, messages, and captions.
- **Main conclusion:** The visual evidence now communicates the Stage 1/Stage 2 boundary and main
  findings without relying on engineering-oriented plots.
- **Evidence path:** [`final figure shortlist`](final_figure_shortlist.md),
  [`figure index`](../../reports/dissertation_final/final_figure_index.md).
- **Status:** Complete.

## 11. Final Repository Audit and Reproducibility Bundle

- **Purpose:** Consolidate evidence and verify that the repository is reproducible and safe to
  hand off.
- **Work completed:** Added the final result/evidence indexes, reproduction runbooks, provenance,
  automated audit, tests, linting, and artifact/privacy hygiene checks.
- **Key result:** The final bundle maps claims to committed manifests, tables, figures, scripts,
  and limitations.
- **Main conclusion:** The repository is ready for dissertation writing and supervisor review,
  while clinical validation remains future work.
- **Evidence path:** [`final evidence index`](final_evidence_index.md),
  [`reproducibility runbook`](reproducibility_runbook.md),
  [`final audit`](../../scripts/final_repository_audit.py).
- **Status:** Complete.

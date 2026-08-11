# Dissertation Documentation Index

This folder contains manuscript-facing documentation that supports the final
dissertation text without changing the experimental benchmark or reported
metrics.

## Provenance Audit

- `docs/dissertation/dataset_source_provenance.md` records the public-source
  provenance audit for the benchmark.
- `reports/audit/dataset_source_mapping.csv` gives component-level source
  mapping.
- `reports/audit/imported_ood_image_provenance.csv` gives row-level provenance
  status for the 900 imported OOD images.
- `reports/audit/dataset_source_provenance_validation.md` records validation
  findings and required checks.

## Interpretation Boundary

The Stage 1 gatekeeper remains an ID-only unsupervised OOD detector. OOD labels
are used for evaluation and optional Stage 2 post-rejection attribution, not
for Stage 1 fitting. The provenance audit updates source documentation only; it
does not change manifests, models, thresholds, figures, or metrics.

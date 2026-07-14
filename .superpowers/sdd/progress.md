# Stage 2 Parent-Grouped Evaluation Progress

Plan: docs/superpowers/plans/2026-07-13-stage2-parent-grouped-evaluation.md
Base: cb4b975ea1a3f4711743881fed1c0fd009852348
Plan commit: 2887a603fa4aed473d8fe6b44503be7c25a054cb

Task 1: complete (commits 2887a60..ca1d49a, review clean)
Task 2: complete (commits ca1d49a..72d7edf, review clean)
Task 3: complete (commits 72d7edf..f8afc47, review clean after two fixes)
Task 4: complete (commits f8afc47..1498f73, review clean after three fixes)
Task 5: complete (documentation/evidence migration; validation recorded in task-5-report.md)
## Task 4: Grouped experiment package and figures

- Status: complete
- RED commit: `c4d6e21` `test: add grouped stage2 report invariants`
- GREEN commit: `9e37d95` `feat: rerun stage2 attribution on grouped splits`
- Verification: 31 focused tests passed; Ruff and diff check passed.
- Selected family method: `feature_statistics_fusion` (validation macro-F1
  0.9924836011792534; test accuracy/macro-F1 1.0/1.0).
- Selected subtype method: `hierarchical_classifier` (validation macro-F1
  0.9058507271445498; test accuracy/macro-F1
  0.9357142857142857/0.9175854801338393).
- Parent/image overlap: zero for every split pair and all-three scope.
- Review notes: tie-aware family interpretation and non-overlapping figure
  legend fixed after visual QA; legacy report hashes preserved.
- Task 4 review fix: compatibility summaries/figure made tie-aware, dependency
  wording made precise, and grouped figure index expanded to four PNG/PDF
  pairs. Expanded focused gate: 32 passed; Ruff/diff check passed.

## Task 5: Grouped documentation and evidence migration

- Status: complete.
- Canonical documentation now uses the parent-grouped Stage 2 protocol and
  validation-selected methods as headline evidence.
- Legacy row-level metrics remain only in labelled historical, audit, or
  sensitivity-comparison contexts.
- The pre-remediation audit artifacts were copied byte-for-byte and verified
  by SHA-256.
- A stale figure-cleanup assertion was updated in commit `e326327` to require
  the grouped Stage 2 comparison figure in main-text documentation.
- Documentation gate: 11 passed; Ruff and diff check passed.

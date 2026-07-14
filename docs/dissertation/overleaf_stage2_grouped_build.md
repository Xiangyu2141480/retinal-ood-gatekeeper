# Overleaf manuscript: parent-grouped Stage 2 build evidence

## Repository boundary

The dissertation manuscript is maintained in a separate Git repository from
the GitHub code and evidence package:

- manuscript remote: `https://git.overleaf.com/6a31238a019dd712e3417241`;
- manuscript branch: `codex/stage2-parent-grouped-evaluation`;
- manuscript base: `822c99a`;
- manuscript content commit: `e14e24b51e8600133094e466240eabd136d862ec`;
- canonical-figure sync commit: `d2f590a9c58c42c90539f03d930adf1c7a5f52d2`.

The repositories have different remotes and histories. No cross-repository
cherry-pick was used.

## Manuscript changes

The grouped result migration updates:

- Abstract and Introduction terminology and Stage 1/Stage 2 boundaries;
- dataset lineage table and parent-grouped split protocol;
- Section 4.6 integrity/provenance and Section 4.7 limitations;
- the fixed `k=1` explanation for global kNN and PatchCore;
- exact Stage 1 scoring functions;
- Stage 2 features, train-only preprocessing, validation-only selection, and
  predicted-family non-oracle routing;
- Stage 2 results, confusion matrices, legacy sensitivity comparison,
  Discussion, Limitations, Conclusion, and Reproducibility.

The validation-selected family method is the feature--statistics fusion
logistic-regression model. The validation-selected subtype method is the
non-oracle hierarchical classifier. Grouped test values are retrospective and
did not participate in model selection.

## Canonical manuscript figure hashes

The committed manuscript PDFs are byte-identical to the canonical GitHub
evidence figures:

- `figure_grouped_family_confusion_matrix.pdf`:
  `2f5e4b688e13658c96a8601d7f0426b1f9e5310e28ee946b543c2438e35d6dce`;
- `figure_grouped_stage2_method_comparison.pdf`:
  `51ad21816401f3ec77ded8f601d69a9cd14f65ac506f35cf8dcaed021f86072a`;
- `figure_grouped_subtype_confusion_matrix.pdf`:
  `1111134fe5a0fbab704ded7b56e80eb5d86d4519aec02ef1502d66e6c78d846d`;
- `figure_legacy_vs_grouped_stage2_metrics.pdf`:
  `2f51a2546fa2b1f769f2bfb9270e20e77e1a737979ccffca09cf3b62c4969748`;
- `figure_two_stage_updated_pipeline.png`:
  `49f2f3907699a74b04d9a3c9de8049405fe18ff0e5d33b0d0a76faae6879d974`.

## Validation

The final manuscript command was:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Outcome:

- exit status 0;
- 62-page PDF;
- no undefined references;
- no undefined citations;
- no missing figures;
- no missing characters;
- no grouped table overflow;
- the long grouped-manifest path was made breakable, removing the new 96 pt
  overfull box found during the first final build.

Remaining overfull warnings are pre-existing long equations/paragraphs; none
is attached to the grouped result tables. Key pages 25, 26, 48, 49, 50, and 56
were rendered to PNG and visually checked for clipping, overlap, table width,
legend readability, figure placement, and manifest-path wrapping.

The final repository verification cited by the manuscript is `208 passed`.
The only exact old-value search hit in `main.tex` is `0.9059`, where it denotes
the new grouped validation subtype macro-F1. The legacy test value is not used
as headline evidence.

## Interpretation boundary

Parent grouping removes cross-partition parent/group overlap. It does not make
variants within one partition independent and does not establish patient-,
device-, site-, or clinical generalisation. Stage 1 remains unchanged and
ID-only; Stage 2 remains optional, supervised, post-rejection, closed-set, and
non-diagnostic.

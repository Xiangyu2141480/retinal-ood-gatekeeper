# Parent-grouped Stage 2 evaluation

## Selection and results

- Family method selected by validation macro-F1: `feature_statistics_fusion` (0.9924836011792534).
- Grouped test family accuracy: 1; macro-F1: 1.
- Subtype method selected by validation macro-F1: `hierarchical_classifier` (0.9058507271445498).
- Grouped test subtype accuracy: 0.9357142857142856; macro-F1: 0.9175854801338392.
- No unique hardest family; all families tied at test F1 1.0000: `modality_shift, semantic_outlier, sensory_artifact`.
- Hardest subtype: `text_watermark` (test F1 0.7741935483870969).

All candidate models were fitted on grouped training data. Model selection was frozen from
grouped validation macro-F1 before test evaluation began. The hierarchical classifier uses
non-oracle predicted-family routing at inference time. Stage 1 remains unchanged: it is the
ID-only unsupervised OOD gatekeeper, and no Stage 2 labels or scores are Stage 1 inputs.

## Legacy sensitivity comparison

- Family validation macro-F1 difference: +0.001709138764976359.
- Family test accuracy difference: +0.01190476190476197.
- Family test macro-F1 difference: +0.009924313284498609.
- Subtype validation macro-F1 difference: +0.03874672107835109.
- Subtype test accuracy difference: +0.01190476190476164.
- Subtype test macro-F1 difference: +0.01164193065121066.

The legacy row-level results are retained only as a sensitivity comparison. In the final split,
train-validation, train-test, validation-test, and all-three image-path and group-ID overlaps are
all zero.

## Scope and limitations

This optional supervised Stage 2 module explains a rejected input after Stage 1; it does not
decide rejection and is not a disease classifier. The grouped protocol removes dependence between
derived variants of the same synthetic parent within this benchmark, but remains a controlled,
closed-set, synthetic-backed, non-clinical evaluation. Reason labels are likely rejection
explanations, not clinical diagnoses. Parent grouping does not establish patient-independent or
device-independent clinical generalisation.

# Dissertation Robustness Results Interpretation

## Summary of Robustness Findings

The robustness analyses extend the merged multi-scheme comparison without changing the core task. Training remains ID-only, OOD examples remain evaluation-only, and the system remains a binary FAF OOD gatekeeper rather than a disease classifier.

## Bootstrap CI Interpretation

Bootstrap confidence intervals were computed from sample-level primary score outputs. The top AUROC estimate remains `Mahalanobis feature`, supporting the PR #21 finding that feature-distance methods are strong on the current synthetic-backed evaluation.

## Training-Size Sensitivity Interpretation

At the largest tested ID subset (700 images), the strongest train-size sensitivity row is `Mahalanobis feature` with AUROC 0.9724. These curves show whether performance saturates before all 700 ID training images are used and provide motivation for future real FAF collection.

## Threshold-Safety Interpretation

Threshold sweeps distinguish research thresholds from deployment-style ID validation quantiles. The strongest deployment-style row by OOD recall in this run is `Mahalanobis feature` / `fixed_id_rejection_20` with OOD recall 0.9388 and ID false rejection rate 0.1200. This is threshold analysis, not deployment approval.

## Artifact-Severity Stress-Test Interpretation

Artifact severity tests were generated from ID fallback images for evaluation only. The highest monotonic severity-score relationship in the compact summary is `autoencoder` on `rectangle_annotation` with Spearman 0.9685.

## Method Disagreement and Failure Cases

The disagreement tables compare Mahalanobis, PatchCore L3, autoencoder, and image statistics decisions at their selected thresholds. These cases support discussion of why a quantitatively strong global/statistical feature method and a localizable PatchCore method can fail on different images.

## Feature-Space Visualization Interpretation

PCA figures use the same feature family as the Mahalanobis baseline. They are intended to show whether semantic outliers, modality shifts, and sensory artifacts form separable feature-space structure and whether that separation explains Mahalanobis performance.

## Runtime and Deployment Practicality

The fastest smoke scoring row is `Autoencoder` at 10.99 ms/image on this local CPU run. Runtime figures should be interpreted as local engineering measurements, not hardware-independent deployment guarantees.

## Limitations

`test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation. OOD stress tests are not clinical prevalence estimates. Research thresholds use OOD labels for analysis only. Real clinical FAF validation, calibration, and prospective workflow testing remain future work.

## Suggested Dissertation Paragraphs

The follow-up robustness experiments show that the ranking observed in the main comparison is not solely a single point estimate. Bootstrap intervals, train-size curves, threshold sweeps, and severity stress tests provide complementary evidence about uncertainty, calibration sensitivity, and failure modes. The analysis also clarifies a practical distinction: Mahalanobis feature distance is quantitatively strong and compact, while PatchCore L3 remains useful for localizing and discussing image-level artifact evidence.

# Dissertation Robustness and Failure-Analysis Plan

## Motivation

PR #21 established the main multi-scheme comparison for the unsupervised retinal FAF OOD gatekeeper. This follow-up PR adds dissertation evidence that goes beyond headline metrics: uncertainty, threshold safety, training-data sensitivity, artifact severity stress tests, method disagreement, feature-space structure, and runtime practicality.

The system remains a binary unsupervised gatekeeper:

```text
input image -> OOD gatekeeper -> ACCEPT valid FAF / REJECT invalid or OOD
```

It is not a disease classifier. Training remains ID-only using `train_id.csv`, `label=0`, and `ood_type=id`. OOD data remains evaluation-only for grouping, stress testing, and interpretation.

## What PR #21 Already Answered

PR #21 answered the main method-comparison questions:

- Which completed scheme performed best quantitatively: Mahalanobis feature distance.
- Which PatchCore variant was best among completed layer variants: PatchCore L3.
- How image statistics, autoencoder, global kNN, Mahalanobis, and PatchCore variants compare on AUROC, AUPRC, FPR@95%TPR, per-type metrics, and per-subtype metrics.
- Which dissertation figures are most useful for the main method-comparison chapter.

The PR also documented that `test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation.

## What This PR Adds

This PR answers the next layer of dissertation questions:

- How stable are the results under bootstrap resampling?
- How much ID training data is needed before performance saturates?
- How sensitive are accept/reject decisions to threshold choice?
- Which artifact severities are detected earliest?
- Where do Mahalanobis and PatchCore disagree?
- Which cases fail under different methods?
- How separable are ID/OOD groups in feature space?
- What are the runtime, memory, and interpretability trade-offs?

## Experiments Included

1. Bootstrap confidence intervals for AUROC, AUPRC, FPR@95%TPR, ID false rejection, and OOD recall.
2. Train-size sensitivity using deterministic ID-only subsets of 50, 100, 250, 500, and 700 images.
3. Threshold policy sweeps comparing research thresholds and deployment-style ID validation quantiles.
4. Artifact severity stress tests for text watermark, Gaussian noise, JPEG compression, blur, border crop, and rectangle annotation.
5. Method disagreement and failure-case selection across Mahalanobis, PatchCore L3, autoencoder, and image statistics.
6. Feature-space PCA visualizations for Mahalanobis/global feature embeddings.
7. Runtime and resource analysis for the main completed schemes.
8. Optional subtype influence diagnostics where existing score files make this inexpensive.

## Expected Dissertation Sections Supported

- Methods: threshold policies, robustness protocol, artifact stress-test protocol, runtime measurement protocol.
- Results: bootstrap uncertainty, train-size sensitivity, threshold trade-offs, severity response curves, feature-space plots.
- Discussion: method disagreement, failure cases, deployment practicality, why Mahalanobis and PatchCore serve different roles.
- Limitations: synthetic fallback ID validation, non-clinical stress-test prevalence, missing PatchCore L1 full run, and need for real FAF validation.

## Limitations

The robustness analyses still use the dissertation v1 synthetic-backed dataset. OOD stress tests are evaluation-only perturbations and are not clinical prevalence estimates. Threshold sweeps include research thresholds that use OOD labels for analysis; only ID-validation quantile thresholds are deployment-style. No result in this PR should be framed as clinical deployment readiness.

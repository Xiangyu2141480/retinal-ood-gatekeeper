# Dissertation Outline Draft

## Working Title

Unsupervised out-of-distribution gatekeeping for retinal FAF image quality control.

## Core Thesis Framing

The dissertation studies an upstream binary OOD gatekeeper for retinal Fundus Autofluorescence (FAF) images. Stage 1 decides whether an input should be accepted as valid FAF or rejected as invalid/OOD. Stage 1 is trained only on ID FAF rows and remains unsupervised with respect to OOD labels.

The completed system contains:

1. Stage 1 ID-only OOD gatekeeper.
2. Multi-scheme OOD model comparison.
3. Robustness and failure analysis.
4. Optional Stage 2 rejected-input reason attribution.
5. Reproducible dataset and Git LFS package.
6. Dissertation-ready figures and evidence index.

The project is not disease classification. It does not infer diagnoses, disease grades, biomarkers, or patient labels.

## Chapter Plan

### Chapter 1: Introduction

Introduce the problem of invalid or out-of-distribution inputs in retinal FAF workflows. Frame the contribution as a pre-diagnostic quality-control gatekeeper rather than a diagnostic model.

#### Research Question and Objectives

The primary research question is:

**How can an ID-only unsupervised out-of-distribution detection system be designed and evaluated
as an upstream quality-control gatekeeper for retinal fundus autofluorescence imaging, so that
invalid or unexpected inputs can be rejected before they reach a downstream diagnostic model?**

The primary objective is to design and evaluate Stage 1 as an ID-only unsupervised binary
gatekeeper. The optional Stage 2 extension runs only after rejection and attributes a likely
reason family or subtype; it does not change the Stage 1 decision.

Research objectives:

1. Construct and audit a manifest-driven ID/OOD benchmark for FAF input-quality control.
2. Compare unsupervised Stage 1 method families using aggregate and safety-oriented metrics.
3. Analyse OOD family/subtype behaviour, threshold policies, robustness, and failure cases.
4. Assess whether PatchCore provides useful localisation-oriented evidence.
5. Evaluate optional post-rejection reason attribution while preserving the Stage 1 boundary.

The canonical accessible version of the research question, strategy, results, and limitations is
`docs/dissertation/project_overview.md`.

### Chapter 2: Background and Related Work

Cover retinal FAF quality-control needs, OOD detection, feature-distance methods, reconstruction baselines, PatchCore-style local anomaly evidence, threshold calibration, and the difference between rejection decisions and post-hoc explanations.

### Chapter 3: Dataset

Describe dataset v1, including ID FAF rows, modality-shift OOD, sensory artifacts, semantic outliers, and synthetic ID fallback. State clearly that `test_id_synthetic_fallback.csv` is synthetic FAF fallback, not real clinical FAF validation.

### Chapter 4: Methods

Describe Stage 1 models: image statistics, autoencoder, global feature kNN, Mahalanobis feature distance, and PatchCore variants. All Stage 1 fitting uses ID-only training data.

Describe the optional Stage 2 module separately. Stage 2 is invoked only after Stage 1 rejects an input. It uses OOD reason labels as explanation targets for rejected-input reason attribution, not for Stage 1 OOD fitting.

### Chapter 5: Experiments and Results

Report the multi-scheme Stage 1 comparison, threshold analysis, robustness analysis, and failure analysis. Mahalanobis is the strongest quantitative Stage 1 model on dataset v1, while PatchCore L3 provides useful heatmap/localisation evidence.

Report the optional Phase 2 method comparison as a separate post-rejection explanation experiment. Under the final parent-grouped protocol, grouped validation selects `feature_statistics_fusion` for reason-family attribution and the non-oracle `hierarchical_classifier` for subtype attribution.

### Chapter 6: Discussion

Discuss why global feature-distance methods perform strongly, why local artifacts remain challenging, and why threshold interpretation must stay cautious. Explain that Stage 2 reason labels are likely explanations, not clinical diagnoses.

### Chapter 7: Limitations and Future Work

Limitations include synthetic ID fallback, curated OOD stress-test distributions, lack of real clinical FAF validation, no prospective threshold calibration, and a controlled closed-set Stage 2 taxonomy. Parent grouping eliminates cross-partition parent/group overlap, but variants from a common synthetic parent remain dependent within one split; this is not patient-, device-, or site-independent validation.

Future work should validate on real clinical FAF data, calibrate thresholds prospectively, expand subtle local artifact coverage, repeat grouped Stage 2 evaluation across seeds, and assess explanations on real rejection cases.

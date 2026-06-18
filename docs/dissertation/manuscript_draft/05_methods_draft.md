# Methods Draft

## Stage 1: ID-Only OOD Gatekeeper

The primary system is a binary out-of-distribution gatekeeper for retinal Fundus Autofluorescence (FAF) images. Given an input image, Stage 1 computes an anomaly score and applies a threshold to output either `ACCEPT: valid FAF` or `REJECT: invalid or OOD`.

Stage 1 remains ID-only and unsupervised. The fitting data are valid FAF rows from `train_id.csv`, all with `label=0` and `ood_type=id`. OOD labels are not used to train, tune, or fit the Stage 1 models. OOD rows are used for held-out evaluation, stress-test grouping, and result interpretation only.

The Stage 1 comparison includes image-statistics baselines, an autoencoder reconstruction baseline, global feature kNN, Mahalanobis feature distance, and PatchCore feature-layer variants. Mahalanobis feature distance is treated as the primary quantitative gatekeeper because it gives the strongest overall AUROC and AUPRC on dataset v1. PatchCore L3 is retained as a useful localizable companion because it provides qualitative heatmap evidence.

## Thresholding

The dissertation distinguishes research thresholds from deployment-style prototype thresholds. Research thresholds such as FPR@95%TPR use OOD labels to summarize held-out performance. Deployment-style thresholds use validation ID score quantiles only. The strongest balanced prototype policy is Mahalanobis `val_id_quantile_95`.

## Optional Stage 2: Rejected-Input Reason Attribution

Stage 2 is an optional explanation layer that runs only after Stage 1 has already rejected an input. It does not alter the Stage 1 decision boundary. It is not part of the unsupervised OOD gatekeeper training loop.

For rejected inputs, Stage 2 predicts a likely rejection reason family, such as `modality_shift`, `sensory_artifact`, or `semantic_outlier`, and may also predict a finer subtype. These labels are explanation targets for rejected inputs. They are not disease labels, clinical diagnoses, biomarkers, or patient-level targets.

The Phase 2 comparison uses image-derived features only. Metadata fields such as `image_path`, `filename`, `source`, `source_dataset`, `source_url`, `license_status`, `notes`, `label`, `ood_type`, `ood_subtype`, `synthetic_transform`, and `severity` are not encoded as input features. `ood_type` and `ood_subtype` are targets only.

The selected reason-family method is `linear_svm`. The selected subtype method is the non-oracle `hierarchical_classifier`. At validation and test time, the hierarchical subtype model routes by predicted family, not ground-truth family, so it should not be described as an oracle hierarchy.

## Scope

The system is not a disease classifier. Its outputs are quality-control decisions and, optionally, likely explanations for rejected inputs. The reported experiments are proof-of-concept stress-test evidence, not clinical deployment validation.

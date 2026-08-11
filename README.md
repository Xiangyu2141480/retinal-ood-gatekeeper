# Retinal FAF OOD Gatekeeper

**Unsupervised out-of-distribution detection for quality control in retinal imaging**

This repository contains the research code, benchmark package, experiment outputs, figures, and reproducibility material for the UCL MSc dissertation:

> **Unsupervised Out-of-Distribution Detection for Quality Control in Retinal Imaging**

The project implements a pre-diagnostic gatekeeper for fundus autofluorescence (FAF) imaging. Its purpose is to identify unsupported or invalid inputs before they reach a downstream analysis system.

The gatekeeper is an input-validity model, not a disease classifier. In-distribution (ID) means supported by the intended FAF input domain; it does not mean healthy. Rejection indicates insufficient support from the learned reference distribution and does not imply disease.

> **Research prototype only.** This repository is not a clinical device, and the reported results
> are not clinical deployment validation. Real clinical FAF evaluation and prospective threshold
> calibration are required before any clinical use could be considered.

## Dissertation Submission Version

The code and evidence corresponding to the submitted PHAS0077 MSc dissertation are preserved in
the GitHub release [`v1.0-dissertation-submission`](https://github.com/Xiangyu2141480/retinal-ood-gatekeeper/releases/tag/v1.0-dissertation-submission).

## Project at a Glance

The completed prototype has two separate stages:

1. **Stage 1 — ID-only OOD detection**  
   Fits only nominal synthetic FAF-like images and produces an anomaly score followed by an `ACCEPT` or `REJECT` decision.

2. **Stage 2 — post-rejection attribution**  
   Optionally assigns a rejected image to a known OOD family and subtype. Stage 2 is supervised, closed-set, and cannot alter the Stage 1 decision.

```text
Input image
    |
    v
Stage 1: ID-only anomaly detector
    |
    +-- ACCEPT --> external downstream analysis
    |
    +-- REJECT --> optional Stage 2 family/subtype attribution
```

## Research Contributions

- A manifest-driven, ID-only Stage 1 evaluation pipeline spanning low-level statistics,
  reconstruction, global feature distance, and patch-level anomaly detection.
- A controlled benchmark organised into modality shift, sensory artefact, and semantic-outlier
  families, with per-subtype reporting and explicit dataset audits.
- A threshold analysis that separates retrospective ranking metrics from an ID-validation-only
  prototype operating threshold.
- PatchCore localisation evidence and failure analysis alongside the quantitatively stronger
  Mahalanobis detector.
- An optional, separately evaluated Stage 2 experiment for closed-set family and subtype
  attribution after rejection.

## Main Results

### Stage 1: binary input-validity detection

| Method | AUROC | AUPRC | FPR@95% TPR | OOD recall at ID-calibrated `tau_95` |
| --- | ---: | ---: | ---: | ---: |
| **Mahalanobis feature distance** | **0.9724** | **0.9975** | **0.2000** | **0.9079** |
| Global feature kNN | 0.9458 | 0.9949 | 0.3800 | 0.8024 |
| PatchCore L3 | 0.8819 | 0.9880 | 0.6333 | 0.7097 |

At the 95th-percentile ID-validation threshold, Mahalanobis rejected **90.79%** of the OOD benchmark while falsely rejecting **4.67%** of the synthetic ID fallback set.

AUROC, AUPRC, and FPR@95% TPR are retrospective ranking/safety metrics; the 95th-percentile
ID-validation threshold is the prototype decision policy. The high AUPRC is also influenced by
the OOD-heavy class balance of the evaluation set and should not be read as a prevalence-neutral
clinical estimate.

Subtype analysis showed complementary behaviour. Mahalanobis was strongest for broad shifts and
several diffuse corruptions, whereas PatchCore L3 was useful for local overlays and spatial
anomaly maps, especially text watermarks. Mild blur and high-quality JPEG compression remained
difficult for maximum-patch scoring.

### Stage 2: post-rejection reason attribution

| Task | Validation-selected method | Validation macro-F1 | Grouped-test accuracy | Grouped-test macro-F1 |
| --- | --- | ---: | ---: | ---: |
| OOD family | `feature_statistics_fusion` | 0.9925 | 1.0000 | 1.0000 |
| OOD subtype | `hierarchical_classifier` | 0.9059 | 0.9357 | 0.9176 |

The hardest Stage 2 subtype was `text_watermark`, with test F1 **0.7742**. These are results from
a separate closed-set attribution experiment, not end-to-end gatekeeper performance or clinical
validation.

## Benchmark

The manifest-defined benchmark contains **3,100 unique images**:

| Component | Images | Role |
| --- | ---: | --- |
| Synthetic FAF-like ID | 1,000 | Stage 1 fitting, threshold calibration, and fallback ID testing |
| Imported modality-shift OOD | 400 | 200 colour fundus images and 200 OCT screenshots |
| Imported semantic-outlier OOD | 500 | Natural-image stress tests |
| Derived sensory artefacts | 1,200 | Eight artefact subtypes generated from held-out synthetic parents |

The OOD taxonomy contains three operational families and 11 subtypes:

- **Modality shift:** colour fundus, OCT screenshot
- **Sensory artefact:** text watermark, rectangle annotation, arrow annotation, composite layout, blur, border crop, Gaussian noise, JPEG compression
- **Semantic outlier:** natural image

### Experimental views

- `train_id.csv`: 700 synthetic ID training images
- `val_id.csv`: 150 synthetic ID images for threshold calibration
- `test_id_synthetic_fallback.csv`: 150 synthetic fallback ID test images
- `test_ood_full.csv`: 2,100 OOD stress-test images
- `test_ood_balanced_by_subtype.csv`: 1,650 OOD images used for the primary balanced comparison
- Stage 2 parent-grouped split: 1,260 training, 420 validation, and 420 test OOD images

The final Stage 2 split audit reports zero cross-partition image-path overlap and zero overlap in
normalised group IDs. Grouping uses the parent-image hash where available and the image path
otherwise. Variants from a shared parent may still be dependent within the same split, so this is
not patient-independent clinical validation.

## Methods Evaluated

Stage 1 compares eight completed configurations across five method families:

| Method family | Configuration(s) | Image-level anomaly score |
| --- | --- | --- |
| Image statistics | 24-dimensional descriptor | RMS standardised deviation from the ID centre |
| Reconstruction | Convolutional autoencoder | Mean squared reconstruction error |
| Global pretrained features | kNN | Distance to the nearest nominal pooled embedding |
| Global pretrained features | Mahalanobis | Covariance-normalised distance from the nominal mean |
| Local pretrained features | PatchCore L2, L3, L4, L2+L3 | Maximum nearest-memory distance over query patches |

All Stage 1 methods are fitted using ID images only. OOD labels are reserved for held-out evaluation. The optional Stage 2 module uses labelled OOD examples only for post-rejection attribution.

## Repository Structure

| Location | Contents |
| --- | --- |
| `data/images/dissertation_v1/` | Versioned benchmark image assets managed through Git LFS |
| `datasets/dissertation_v1/` | Dataset manifests, grouped splits, checksums, and metadata |
| `src/retinal_ood/` | Core data, model, evaluation, and utility modules |
| `scripts/` | Training, evaluation, auditing, reporting, and UI entry points |
| `configs/` | Reproducible experiment configurations |
| `reports/dissertation_results/` | Stage 1 comparison, robustness, and legacy Stage 2 results |
| `reports/stage2_grouped/` | Final parent-grouped Stage 2 audit, metrics, and predictions |
| `reports/dissertation_figures/` | Dissertation-ready figures and qualitative evidence |
| `reports/dissertation_final/` | Final summary tables, interpretation notes, and claim mapping |
| `docs/dissertation/` | Project overview, evidence index, runbooks, and limitations guidance |

## Start Here

- [Project overview](docs/dissertation/project_overview.md)
- [Final evidence index](docs/dissertation/final_evidence_index.md)
- [Final figure shortlist](docs/dissertation/final_figure_shortlist.md)
- [Final table shortlist](docs/dissertation/final_table_shortlist.md)
- [Reproducibility runbook](docs/dissertation/reproducibility_runbook.md)
- [School server runbook](docs/dissertation/school_server_runbook.md)
- [Claims and limitations matrix](docs/dissertation/claims_and_limitations_matrix.md)
- [Dissertation handoff index](docs/dissertation/README.md)
- [Submitted dissertation source snapshot](docs/dissertation/submission_source/README.md)
- [Final result summary](reports/dissertation_final/final_result_summary.md)

## Quick Start

### 1. Clone the repository and retrieve Git LFS assets

```bash
git clone https://github.com/Xiangyu2141480/retinal-ood-gatekeeper.git
cd retinal-ood-gatekeeper
git lfs install
git lfs pull
```

### 2. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Run the repository checks

```bash
pytest
ruff check .
python scripts/final_repository_audit.py --repo-root .
```

The final audit verifies the dissertation evidence bundle, required documentation, dataset metadata, and repository hygiene without repeating the most expensive experiments.

## Reproducing the Experiments

The final complete Stage 1 comparison uses `configs/multi_scheme_grid.yaml`, which includes the
image-statistics, autoencoder, global kNN, Mahalanobis, and PatchCore configurations.

<details>
<summary>Mahalanobis CPU smoke test</summary>

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_delivery_smoke \
  --only mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv \
  --max-train-images 8 \
  --max-test-images 8 \
  --device cpu \
  --seed 42
```

</details>

See the [reproducibility runbook](docs/dissertation/reproducibility_runbook.md) for the complete
Stage 1 and Stage 2 command sequences. For Linux cluster setup, storage guidance, and SLURM
examples, use the [school server runbook](docs/dissertation/school_server_runbook.md). Generated
runs remain outside Git history.

## Local Gatekeeper Demo

After training or loading a compatible detector and threshold, launch the local drag-and-drop interface with:

```bash
python scripts/serve_gatekeeper_app.py \
  --config <experiment-config.yaml> \
  --checkpoint <model-artifact>
```

The interface returns an input-validity decision and, for PatchCore configurations, can display anomaly overlays. It does not diagnose disease or produce clinical labels.

## Threshold Policy

Two threshold concepts are kept separate:

- **ID-calibrated threshold:** selected from held-out ID validation scores and used for the prototype operating decision.
- **Research-only threshold:** selected using labelled evaluation OOD data for retrospective comparison, including FPR@95% TPR.

A research-only operating point is not a deployment threshold.

## Reproducibility and Evidence

The repository preserves versioned manifests, checksums, fixed configurations, sample-level
scores, aggregate and category metrics, threshold/robustness analyses, heatmaps, grouped Stage 2
audits, and claim-to-evidence documentation. The canonical map is the
[final evidence index](docs/dissertation/final_evidence_index.md); results are under
`reports/dissertation_results/`, `reports/stage2_grouped/`, and `reports/dissertation_final/`.

## Limitations

The reported results are an internal proof of concept, not clinical validation.

- Nominal Stage 1 training, calibration, and fallback testing use synthetic FAF-like images.
- The OOD benchmark is a curated stress test and does not estimate clinical prevalence.
- The taxonomy cannot cover every future device shift, corruption, export format, or user error.
- Stage 2 is supervised and closed-set; an unseen mechanism may be forced into a known category
  or misclassified.
- Parent grouping prevents cross-partition parent overlap, but it is not patient-, device-, site-, or protocol-independent validation.
- PatchCore maps show detector-relative feature discrepancy and are not pathology maps or segmentation masks.

External evaluation on real FAF acquired across patients, devices, sites, and protocols is required before any deployment claim can be made.

## Data, Licensing, and Safety

Dataset images are tracked through Git LFS where applicable. The benchmark combines synthetic
FAF-like images with prepared public-source OOD assets and retains manifests, content hashes,
source-bucket fields, and available derivation metadata. Imported assets remain subject to their
original source licences and conditions; this repository does not grant unrestricted
redistribution rights for every image asset. Code and data reuse should therefore be assessed
separately against the applicable repository and upstream-source terms. No new `LICENSE` claim is
made here.

Do not commit private clinical images, patient identifiers, credentials, sensitive local paths,
private model artefacts, or assets whose redistribution is not permitted.

It is not a medical device, a diagnostic system, or evidence of prospective clinical safety.

## Citation and Versioning

When referring to this work, cite the associated UCL MSc dissertation and use the
`v1.0-dissertation-submission` release as the version corresponding to the submitted project.
Later repository revisions may evolve beyond the dissertation snapshot, so the repository URL
alone is not a sufficient version identifier.

## Author and Academic Context

**Xiangyu Cui**  
MSc Scientific and Data Intensive Computing  
University College London

Dissertation supervisors: **Dr. William Woof** and **Dr. Nikos Nikolaou**  
Research advisor: **Yiu Wai Chan**

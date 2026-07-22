# Retinal FAF OOD Gatekeeper

**Unsupervised out-of-distribution detection for quality control in retinal imaging**

This repository contains the research code, benchmark package, experiment outputs, figures, and reproducibility material for the UCL MSc dissertation:

> **Unsupervised Out-of-Distribution Detection for Quality Control in Retinal Imaging**

The project develops a pre-diagnostic gatekeeper for fundus autofluorescence (FAF) imaging. Its purpose is to identify unsupported or invalid inputs before they reach a downstream analysis system.

The gatekeeper is an input-validity model, not a disease classifier. In-distribution (ID) means supported by the intended FAF input domain; it does not mean healthy. Rejection indicates insufficient support from the learned reference distribution and does not imply disease.

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

### Selected methods

- **Primary Stage 1 detector:** Mahalanobis feature distance
- **Localisation-oriented companion:** PatchCore L3
- **Stage 2 family model:** `feature_statistics_fusion`
- **Stage 2 subtype model:** non-oracle `hierarchical_classifier`

## Main Results

### Stage 1: binary input-validity detection

| Method | AUROC | AUPRC | FPR@95% TPR | OOD recall at ID-calibrated `tau_95` |
| --- | ---: | ---: | ---: | ---: |
| **Mahalanobis feature distance** | **0.9724** | **0.9975** | **0.2000** | **0.9079** |
| Global feature kNN | 0.9458 | 0.9949 | 0.3800 | 0.8024 |
| PatchCore L3 | 0.8819 | 0.9880 | 0.6333 | 0.7097 |

At the 95th-percentile ID-validation threshold, Mahalanobis rejected **90.79%** of the OOD benchmark while falsely rejecting **4.67%** of the synthetic ID fallback set.

Subtype analysis showed complementary detector behaviour:

- Mahalanobis was strongest for broad modality shifts, semantic outliers, and several diffuse corruptions.
- PatchCore L3 was more sensitive to local overlays, especially text watermarks, and produced spatial anomaly maps.
- Text watermark was the main Stage 1 weakness for globally pooled feature methods.
- Mild blur and high-quality JPEG compression were difficult for maximum-patch scoring.

### Stage 2: post-rejection reason attribution

| Task | Validation-selected method | Validation macro-F1 | Grouped-test accuracy | Grouped-test macro-F1 |
| --- | --- | ---: | ---: | ---: |
| OOD family | `feature_statistics_fusion` | 0.9925 | 1.0000 | 1.0000 |
| OOD subtype | `hierarchical_classifier` | 0.9059 | 0.9357 | 0.9176 |

The hardest Stage 2 subtype was `text_watermark`, with test F1 **0.7742**. These results describe closed-set attribution on labelled OOD data and are not end-to-end clinical performance estimates.

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

- **Stage 1 ID split:** 700 training, 150 validation, 150 fallback test images
- **Primary Stage 1 benchmark:** 150 synthetic ID fallback images and 1,650 subtype-balanced OOD images
- **Stage 2 grouped split:** 1,260 training, 420 validation, and 420 test OOD images

Stage 2 grouping uses the parent-image hash where available and the image path otherwise. This removes cross-partition parent overlap, although transformed variants remain dependent within their assigned partition.

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

For a concise understanding of the study, use the following documents:

- [Project overview](docs/dissertation/project_overview.md)
- [Final evidence index](docs/dissertation/final_evidence_index.md)
- [Reproducibility runbook](docs/dissertation/reproducibility_runbook.md)
- [Final figure shortlist](docs/dissertation/final_figure_shortlist.md)
- [Final table shortlist](docs/dissertation/final_table_shortlist.md)
- [Claims and limitations matrix](docs/dissertation/claims_and_limitations_matrix.md)

The evidence index links the principal dissertation claims to the corresponding manifests, scripts, tables, figures, and limitations.

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

The complete command sequence is documented in the [reproducibility runbook](docs/dissertation/reproducibility_runbook.md). The main entry points are summarised below.

### Stage 1 experiment grid

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/grid_full \
  --device auto \
  --seed 42 \
  --skip-existing
```

The grid runner applies a common manifest audit, seed, device policy, and output structure across method comparisons. It produces aggregate metrics, per-family and per-subtype results, threshold analyses, heatmaps, and selected failure cases.

### Final parent-grouped Stage 2 evaluation

```bash
python scripts/build_reason_attribution_manifests.py \
  --input datasets/dissertation_v1/manifests/test_ood_full.csv \
  --out-dir datasets/dissertation_v1/manifests \
  --split-mode grouped \
  --output-prefix reason_grouped \
  --seed 42

python scripts/audit_stage2_grouped.py \
  --input datasets/dissertation_v1/manifests/test_ood_full.csv \
  --train datasets/dissertation_v1/manifests/reason_grouped_train.csv \
  --val datasets/dissertation_v1/manifests/reason_grouped_val.csv \
  --test datasets/dissertation_v1/manifests/reason_grouped_test.csv \
  --out-dir reports/stage2_grouped \
  --seed 42

python scripts/generate_stage2_grouped_report.py \
  --root-dir data \
  --train-manifest datasets/dissertation_v1/manifests/reason_grouped_train.csv \
  --val-manifest datasets/dissertation_v1/manifests/reason_grouped_val.csv \
  --test-manifest datasets/dissertation_v1/manifests/reason_grouped_test.csv \
  --legacy-dir reports/dissertation_results/reason_attribution_method_comparison \
  --out-dir reports/stage2_grouped \
  --figures-dir reports/dissertation_figures/stage2_grouped \
  --seed 42
```

Manifest paths beginning with `images/...` resolve under `--root-dir data`.

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

The repository preserves:

- version-controlled manifests and checksums;
- configuration files and fixed random seeds;
- sample-level score tables;
- aggregate, family-level, and subtype-level metrics;
- threshold and bootstrap analyses;
- training-size and artefact-severity stress tests;
- failure cases and method-disagreement examples;
- PatchCore spatial anomaly maps;
- Stage 2 grouped-split audits and predictions;
- claim-to-evidence documentation for the dissertation.

Key evidence locations:

- `reports/dissertation_results/multi_scheme_comparison/`
- `reports/dissertation_results/robustness_analysis/`
- `reports/stage2_grouped/`
- `reports/dissertation_figures/`
- `reports/dissertation_final/`
- `docs/dissertation/final_evidence_index.md`

## Limitations

The reported results are an internal proof of concept, not clinical validation.

- Nominal Stage 1 training, calibration, and fallback testing use synthetic FAF-like images.
- The OOD benchmark is a curated stress test and does not estimate clinical prevalence.
- The taxonomy cannot cover every future device shift, corruption, export format, or user error.
- Stage 2 is supervised and closed-set; an unseen mechanism may be mapped to an existing category or returned as unknown.
- Parent grouping prevents cross-partition parent overlap, but it is not patient-, device-, site-, or protocol-independent validation.
- PatchCore maps show detector-relative feature discrepancy and are not pathology maps or segmentation masks.

External evaluation on real FAF acquired across patients, devices, sites, and protocols is required before any deployment claim can be made.

## Data, Licensing, and Safety

The repository uses Git LFS for benchmark assets and retains manifests, content hashes, source-bucket fields, and available derivation metadata. Imported OOD assets remain subject to their original licences and usage conditions. The packaged benchmark does not provide complete immutable per-image upstream URLs and licence records for every imported image.

Do not commit:

- private clinical images or patient identifiers;
- institutional credentials or access notes;
- unrestricted local paths containing sensitive information;
- private model weights or manifests;
- files whose redistribution is not permitted by the source licence.

This repository is a research prototype. It is not a medical device, a diagnostic system, or evidence of prospective clinical safety.

## Author and Academic Context

**Xiangyu Cui**  
MSc Scientific and Data Intensive Computing  
University College London

Dissertation supervisors: **Dr. William Woof** and **Dr. Nikos Nikolaou**  
Research advisor: **Yiu Wai Chan**

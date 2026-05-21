# Retinal FAF OOD Gatekeeper

Unsupervised out-of-distribution detection for quality control in retinal imaging.

This repository is a final-year project scaffold for building a **pre-diagnostic quality-control gatekeeper** for Fundus Autofluorescence (FAF) images. The detector is trained only on valid/normal FAF images and rejects invalid clinical inputs before they reach a downstream diagnostic model.

## What this project builds

The project builds a Python/PyTorch pipeline that:

1. Loads valid **synthetic Heidelberg FAF** images for unsupervised training.
2. Learns a normal-image feature manifold using a primary method such as **PatchCore**.
3. Scores every incoming image with a continuous anomaly score `S(x)`.
4. Applies a threshold `tau` to output either `ACCEPT: valid FAF` or `REJECT: OOD/anomaly`.
5. Evaluates on a stress-test set containing:
   - valid real FAF images,
   - wrong-modality retinal images such as colour fundus or IR,
   - FAF images with watermarks/text/annotations/composite artifacts,
   - non-retinal semantic outliers.
6. Reports clinical safety metrics: AUROC, AUPRC, FPR@95%TPR, confusion matrix at selected thresholds.
7. Produces anomaly heatmaps for interpretability.

## Repository status

This is a starter scaffold. It intentionally does **not** include medical image data, model weights, private manifests, or institutional files.

## Recommended thesis contribution

Main question:

> Can an unsupervised feature-space OOD gatekeeper trained on synthetic FAF images reject true invalid clinical inputs while not falsely rejecting valid real FAF images affected only by the synthetic-to-real domain gap?

Recommended implementation path:

- MVP: PatchCore with ResNet/WideResNet mid-level features.
- Baseline: convolutional autoencoder reconstruction error.
- Ablation: compare feature layers `layer1`, `layer2`, `layer3`, `layer4`, and `layer2+layer3`.
- Clinical evaluation: AUROC + AUPRC + FPR@95%TPR.
- Interpretability: anomaly heatmaps and example failure-case grids.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Prepare a local data folder that is **not committed** to GitHub:

```text
data/
  manifests/
    train_synthetic_faf.csv
    val_synthetic_faf.csv
    test_real_id.csv
    test_ood.csv
  images/   # ignored by git
```

Example commands once the implementation is completed:

```bash
python scripts/train_patchcore.py --config configs/patchcore_resnet50_l23.yaml
python scripts/evaluate.py --config configs/evaluation.yaml --checkpoint runs/patchcore_l23/model.joblib
python scripts/generate_report_tables.py --runs-dir runs/ --out reports/results_summary.md
```

## Manifest CSV schema

Each manifest should contain at least:

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
/path/to/image.png,0,train,synthetic_faf,id,,,,
/path/to/color_fundus.jpg,1,test,real_ood,modality_shift,,,,
```

Labels:

- `0`: in-distribution / valid FAF
- `1`: anomaly / OOD / invalid input

OOD types:

- `modality_shift`
- `sensory_artifact`
- `semantic_outlier`
- `scanner_shift`
- `unknown`

## Project documentation

See the `docs/` directory:

- `LITERATURE_REVIEW_ANALYSIS_CN.md` — Chinese explanation of what the literature review means for implementation.
- `PROJECT_SPEC_CN.md` — product and research specification.
- `DATASET_PLAN_CN.md` — dataset layout, manifests, split strategy, privacy rules.
- `EXPERIMENT_PROTOCOL_CN.md` — exact experiments, metrics, ablations, reporting.
- `CODEX_MASTER_PROMPT.md` — reusable prompt to paste into Codex tasks.
- `CODEX_TASKS.md` — staged Codex task list with ready-to-use prompts.
- `GITHUB_WORKFLOW_CN.md` — GitHub branch/issue/PR workflow.
- `FINAL_REPORT_OUTLINE_CN.md` — thesis/report writing structure.

## Privacy and safety

Never commit:

- clinical images,
- patient identifiers,
- full local paths containing patient information,
- model weights trained on private data,
- institutional credentials or data-access notes.

Use `.gitignore` and private storage for all datasets.

## Suggested final deliverables

- Working training/evaluation code.
- At least one strong feature-space model and one baseline.
- Reproducible experiment configs.
- Metrics tables and plots.
- Heatmap visualizations.
- Written analysis of sim-to-real gap and feature-layer ablation.
- GitHub repository with clean documentation and tests.

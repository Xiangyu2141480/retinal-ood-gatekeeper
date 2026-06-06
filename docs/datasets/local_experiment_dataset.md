# Local Experiment Dataset Notes

This repository is a retinal FAF OOD gatekeeper, not a disease classifier. Local images are used
only through manifest CSV files and YAML configs. Do not commit local image data, generated
manifests from private runs, model weights, heatmaps, or run outputs.

## Local Data Is Ignored

Image data should stay under ignored local paths such as:

```text
data/images/
data/raw/
data/private/
data/syntheye*/
```

The Git repository should contain code, configs, documentation, tests, and small templates only.
Manifests and configs are the source of truth for experiments; raw images are not part of the
reviewable code history.

## Current Proof-Of-Concept Dataset

The current local dataset is a proof-of-concept setup. It is not evidence that real clinical FAF
validation has been completed.

Expected local components:

```text
data/images/synthetic_faf/                 # label=0, ood_type=id
data/images/ood_artifact/                  # label=1, ood_type=sensory_artifact
data/images/ood_semantic/natural/          # label=1, ood_type=semantic_outlier
data/images/ood_modality/colour_fundus/    # label=1, ood_type=modality_shift
data/images/ood_modality/oct_screenshot/   # label=1, ood_type=modality_shift
```

`test_real_id.csv` is currently a synthetic ID fallback when it is generated from SynthEye. Do not
describe it as real clinical FAF validation unless the manifest source actually contains real,
anonymized clinical FAF images. The validator warns when a manifest named like `test_real_id.csv`
contains only synthetic sources.

## Manifest Taxonomy

Use only these `ood_type` values:

```text
id
modality_shift
sensory_artifact
semantic_outlier
```

Rules:

- `label=0` means valid FAF / ID and must use `ood_type=id`.
- `label=1` means invalid/OOD and must not use `ood_type=id`.
- Training rows must be ID-only: `split=train`, `label=0`, `ood_type=id`.
- OOD rows are for validation/testing only.
- Public/non-medical OOD images may have an empty `patient_id`.

## OOD Subtypes

Use `ood_subtype` when possible so result tables can separate visibly different OOD categories.
Recommended examples:

```text
colour_fundus
oct_screenshot
natural_cifar10
text_watermark
rectangle_annotation
arrow_annotation
composite_layout
blur_artifact
border_crop
gaussian_noise
jpeg_compression
```

Colour fundus and OCT should be reported separately even though both are `modality_shift`. CIFAR
natural images are semantic-outlier stress tests. Generated artifact images are synthetic
sensory-artifact stress tests and should be generated only from held-out `val` or `test` ID images,
not train images.

## Validation Command

Run manifest validation before training or evaluation:

```bash
python scripts/validate_manifests.py --root-dir data \
  data/manifests/train_synthetic_faf.csv \
  data/manifests/val_synthetic_faf.csv \
  data/manifests/test_ood.csv
```

Optional JSON audit:

```bash
python scripts/validate_manifests.py --root-dir data \
  --write-json reports/generated/dataset_audit.json \
  data/manifests/train_synthetic_faf.csv \
  data/manifests/val_synthetic_faf.csv \
  data/manifests/test_ood.csv
```

## Reproducibility Commands

Use the grid runner for dissertation comparisons. It validates the manifests, writes a
`manifest_audit.json`, calls the existing training/evaluation CLIs, and aggregates metrics without
including raw images, model weights, heatmaps, or per-image private metadata in Git.

Validate the complete manifest set:

```bash
python scripts/validate_manifests.py --root-dir data \
  --write-json reports/generated/manifest_audit.json \
  data/manifests/train_synthetic_faf.csv \
  data/manifests/val_synthetic_faf.csv \
  data/manifests/test_real_id.csv \
  data/manifests/test_ood.csv
```

Run a small PatchCore L2+L3 smoke test:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/grid_smoke \
  --only patchcore_layer2_layer3 \
  --max-train-images 8 \
  --max-test-images 8 \
  --device cpu \
  --seed 42
```

Run the full dissertation grid:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/grid_full \
  --device auto \
  --seed 42 \
  --skip-existing
```

Generate report tables from an existing runs directory if you need a separate summary:

```bash
python scripts/generate_report_tables.py \
  --runs-dir reports/generated/grid_full/runs \
  --out reports/generated/grid_full/experiment_summary.md \
  --csv-out reports/generated/grid_full/experiment_summary.csv \
  --per-ood-csv-out reports/generated/grid_full/per_ood_type_metrics.csv
```

Expected grid outputs:

```text
reports/generated/grid_full/
  manifest_audit.json
  metrics_summary.csv
  metrics_summary.md
  per_ood_type_metrics.csv
  per_ood_subtype_metrics.csv   # only when test_ood.csv includes ood_subtype
  runs/                         # ignored generated checkpoints and evaluation outputs
```

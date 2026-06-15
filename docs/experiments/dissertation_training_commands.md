# Dissertation Training Commands

All commands are run from the repository root on branch `codex/dissertation-training-and-figures`.

## Setup and Baseline Validation

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Observed baseline before implementation:

- `pytest` -> 140 passed, 11 warnings
- `ruff check .` -> passed

## Dry Run

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/dry_run \
  --dry-run \
  --device cpu \
  --seed 42 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv \
  --max-train-images 8 \
  --max-test-images 4
```

Result: passed; emitted train/evaluate commands for all six required experiments.

## Smoke Run

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/smoke \
  --device cpu \
  --seed 42 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv \
  --max-train-images 8 \
  --max-test-images 4
```

Result: passed for Autoencoder, PatchCore L1, L2, L3, L4, and L2+L3.

## Primary Balanced-by-Subtype Run

CPU runtime command for the prioritized model set. PatchCore runs use a uniform
10,000-patch coreset cap so the L2, L3, and L2+L3 ablation is comparable and
tractable on CPU:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/primary_balanced_by_subtype_10k \
  --device cpu \
  --seed 42 \
  --only autoencoder_baseline,patchcore_layer2,patchcore_layer3,patchcore_layer2_layer3 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  --patchcore-max-train-patches 10000
```

Observed result: completed for Autoencoder, PatchCore L2, PatchCore L3, and
PatchCore L2+L3. PatchCore L3 was the best overall PatchCore model by AUROC on
this primary run.

Extension command for the remaining PatchCore layers:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/primary_balanced_by_subtype_10k \
  --device cpu \
  --seed 42 \
  --only patchcore_layer1,patchcore_layer4 \
  --skip-existing \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  --patchcore-max-train-patches 10000
```

## Secondary Balanced-by-Type Run

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/secondary_balanced_by_type \
  --device cpu \
  --seed 42 \
  --only autoencoder_baseline,patchcore_layer2,patchcore_layer3,patchcore_layer2_layer3 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_type.csv \
  --patchcore-max-train-patches 10000
```

## Full OOD Stress Run

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/experiment_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/stress_full_ood \
  --device cpu \
  --seed 42 \
  --only autoencoder_baseline,patchcore_layer2,patchcore_layer3,patchcore_layer2_layer3 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_full.csv \
  --patchcore-max-train-patches 10000
```

## Curated Figure Suite

Generate category-targeted heatmaps for the selected PatchCore model before building the final figure suite:

```bash
python scripts/generate_selected_patchcore_heatmaps.py \
  --config reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/configs/patchcore_layer3.yaml \
  --checkpoint reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/patchcore_memory.npz \
  --scores-csv reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/evaluation/scores.csv \
  --out-dir reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/evaluation/selected_heatmaps \
  --category sensory_artifact \
  --category modality_shift \
  --per-category 3
```

Then build the curated dissertation figures:

```bash
python scripts/generate_dissertation_figure_suite.py \
  --reports-dir reports/generated/dissertation_runs/primary_balanced_by_subtype_10k \
  --out-dir reports/dissertation_figures \
  --dpi 300
```

## Final Validation Before PR

```bash
pip install -e ".[dev]"
pytest
ruff check .
git diff --check
```

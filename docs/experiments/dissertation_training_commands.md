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

The multi-scheme comparison adds three lightweight ID-only baselines:

- `image_statistics`
- `global_feature_knn`
- `mahalanobis_feature`

Primary baseline command:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/multi_scheme_primary \
  --device cpu \
  --seed 42 \
  --only image_statistics,global_feature_knn,mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv
```

PatchCore L4 was completed separately for the primary layer ablation:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/patchcore_l4_primary \
  --device cpu \
  --seed 42 \
  --only patchcore_layer4 \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  --patchcore-max-train-patches 10000
```

PatchCore L1 is listed as runtime-limited in the result package rather than imputed.

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/multi_scheme_secondary \
  --device cpu \
  --seed 42 \
  --only image_statistics,global_feature_knn,mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_type.csv
```

## Full OOD Stress Run

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/dissertation_runs/multi_scheme_stress \
  --device cpu \
  --seed 42 \
  --only image_statistics,global_feature_knn,mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_full.csv
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

Build the multi-scheme comparison package from compact generated evaluation outputs:

```bash
python scripts/generate_multi_scheme_comparison_package.py \
  --run-root balanced_by_subtype=reports/generated/dissertation_runs/primary_balanced_by_subtype_10k \
  --run-root balanced_by_subtype=reports/generated/dissertation_runs/multi_scheme_primary \
  --run-root balanced_by_subtype=reports/generated/dissertation_runs/patchcore_l4_primary \
  --run-root balanced_by_type=reports/generated/dissertation_runs/multi_scheme_secondary \
  --run-root balanced_by_type=reports/generated/dissertation_runs/selected_secondary \
  --run-root full_ood_stress=reports/generated/dissertation_runs/multi_scheme_stress \
  --run-root full_ood_stress=reports/generated/dissertation_runs/selected_stress \
  --results-dir reports/dissertation_results/multi_scheme_comparison \
  --figures-dir reports/dissertation_figures \
  --dpi 300
```

The selected secondary/stress folders are checkpoint-only evaluations for Autoencoder,
PatchCore L3, and PatchCore L2+L3 using the primary trained checkpoints. They keep the
evaluation matrix broad without retraining those models for every OOD split.

## Final Validation Before PR

```bash
pip install -e ".[dev]"
pytest
ruff check .
git diff --check
```

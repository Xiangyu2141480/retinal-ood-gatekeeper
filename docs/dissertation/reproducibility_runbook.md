# Dissertation Reproducibility Runbook

This runbook reproduces the repository state and final evidence package without adding new major experiments. Expensive full training should only be rerun when a required output is missing.

## 1. Clone and Git LFS Setup

```bash
git clone https://github.com/Xiangyu2141480/retinal-ood-gatekeeper.git
cd retinal-ood-gatekeeper
git lfs install
git lfs pull
```

Dataset images for dissertation v1 are tracked with Git LFS under `data/images/dissertation_v1/`.

## 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## One-Command Final Verification

After environment setup and `git lfs pull`, run this single shell command to rebuild the final indexes, audit the repository handoff, and validate the two core manifests:

```bash
python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final && python scripts/final_repository_audit.py && python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv
```

## 3. Dataset Verification

Schema and taxonomy validation:

```bash
python scripts/validate_manifests.py --root-dir data \
  datasets/dissertation_v1/manifests/train_id.csv \
  datasets/dissertation_v1/manifests/val_id.csv \
  datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  datasets/dissertation_v1/manifests/test_ood_full.csv
```

Image integrity audit:

```bash
python scripts/audit_dataset_images.py --root-dir data \
  --manifest datasets/dissertation_v1/manifests/train_id.csv \
  --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --manifest datasets/dissertation_v1/manifests/test_ood_full.csv \
  --out-dir reports/local_audits/dissertation_v1 \
  --write-json reports/local_audits/dissertation_v1/image_audit.json \
  --fail-on-corrupt
```

## 4. Smoke Experiment

Run a tiny CPU-only smoke experiment to verify the training/evaluation path without rerunning the full dissertation grid:

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

Generated smoke outputs are ignored by Git under `reports/generated/`.

## 5. Main Result Reproduction

The committed main-result package is generated from existing evaluation outputs with:

```bash
python scripts/generate_multi_scheme_comparison_package.py \
  --run-root balanced_by_subtype=reports/generated/dissertation_runs/multi_scheme_primary \
  --results-dir reports/dissertation_results/multi_scheme_comparison \
  --figures-dir reports/dissertation_figures
```

If the generated run root is absent on a new server clone, use the smoke command above to verify the workflow, then rerun the intended grid only if the dissertation outputs must be regenerated from scratch.

## 6. Figure Regeneration

Multi-scheme comparison figures:

```bash
python scripts/generate_multi_scheme_comparison_package.py \
  --run-root balanced_by_subtype=reports/generated/dissertation_runs/multi_scheme_primary \
  --results-dir reports/dissertation_results/multi_scheme_comparison \
  --figures-dir reports/dissertation_figures
```

Robustness figures:

```bash
python scripts/generate_dissertation_robustness_analysis.py \
  --results-dir reports/dissertation_results/robustness_analysis \
  --figures-dir reports/dissertation_figures/robustness \
  --bootstrap-samples 500 \
  --severity-sample-size 12 \
  --device cpu
```

Final figure/table/evidence indexes:

```bash
python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final
```

## 7. Common Troubleshooting

Missing Git LFS files:

- Run `git lfs install` and `git lfs pull`.
- Check `.gitattributes` contains `data/images/dissertation_v1/**/*.png filter=lfs`.

Missing dataset images:

- Verify `data/images/dissertation_v1/` exists after `git lfs pull`.
- Run manifest validation with `--no-check-files` only for schema debugging, not final reproduction.

CUDA not available:

- Use `--device cpu` for smoke tests.
- Full PatchCore grids can be slow on CPU; document hardware and avoid committing generated run folders.

Windows path issues:

- Quote paths containing spaces.
- Use PowerShell line continuations only in PowerShell; use backslashes in bash.

Long runtime:

- Use `--max-train-images`, `--max-test-images`, and `--only` for smoke checks.
- Use `--skip-existing` when rerunning grid commands.

CI-only limitations:

- CI validates code and small fixtures, not full GPU training.
- Full dissertation outputs are committed as compact reports and selected figures.

## 8. Expected Outputs

- Metrics: `reports/dissertation_results/`
- Figures: `reports/dissertation_figures/`
- Final bundle: `reports/dissertation_final/`
- Local smoke logs/results: `reports/generated/dissertation_delivery_smoke/`
- Local image audits: `reports/local_audits/`

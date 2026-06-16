#!/bin/bash
#SBATCH --job-name=retinal-ood-smoke
#SBATCH --output=reports/generated/slurm_logs/smoke_%j.out
#SBATCH --error=reports/generated/slurm_logs/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

mkdir -p reports/generated/slurm_logs
python -m pip install -e ".[dev]"
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/server_smoke \
  --only mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv \
  --max-train-images 8 \
  --max-test-images 8 \
  --device cpu \
  --seed 42

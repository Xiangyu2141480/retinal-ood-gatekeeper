#!/bin/bash
#SBATCH --job-name=retinal-ood-selected
#SBATCH --output=reports/generated/slurm_logs/selected_%j.out
#SBATCH --error=reports/generated/slurm_logs/selected_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p reports/generated/slurm_logs
python -m pip install -e ".[dev]"
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/server_selected \
  --only mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  --device cuda \
  --seed 42 \
  --skip-existing

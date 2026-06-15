# Dissertation Training and Figure Plan

## Base State

- Repository: `Xiangyu2141480/retinal-ood-gatekeeper`
- Base branch: `origin/main`
- Base decision: PR #20 was merged on 2026-06-14, so this stage starts from latest `origin/main`.
- Working branch: `codex/dissertation-training-and-figures`

## Primary Question

Can an unsupervised OOD gatekeeper trained only on valid FAF images reliably reject OOD inputs from:

- `modality_shift`
- `sensory_artifact`
- `semantic_outlier`

The gatekeeper remains binary: accept valid FAF (`label=0`, `ood_type=id`) or reject invalid/OOD input (`label=1`). This is not a disease classifier and not a supervised OOD subtype classifier.

## Dataset

Dataset package: `datasets/dissertation_v1/`

Training and validation:

- Train: `datasets/dissertation_v1/manifests/train_id.csv`
- Validation threshold calibration: `datasets/dissertation_v1/manifests/val_id.csv`

ID evaluation:

- `datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv`

Important warning: this ID test set is synthetic FAF fallback only, not real clinical FAF validation.

Primary OOD evaluation:

- `datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv`

Secondary OOD evaluations:

- `datasets/dissertation_v1/manifests/test_ood_balanced_by_type.csv`
- `datasets/dissertation_v1/manifests/test_ood_full.csv`

Smoke OOD evaluation:

- `datasets/dissertation_v1/manifests/test_ood_smoke.csv`

## Model Family and Baselines

Primary family:

- PatchCore with ResNet-50 feature extractor.

Required experiment set:

- Autoencoder baseline
- PatchCore layer1
- PatchCore layer2
- PatchCore layer3
- PatchCore layer4
- PatchCore layer2+layer3

Runtime fallback on CPU:

- Run the core set first: Autoencoder, PatchCore L2, PatchCore L3, PatchCore L2+L3.
- Extend to PatchCore L1 and L4 after the core result set is complete.
- Use the same PatchCore `max_train_patches` cap across compared PatchCore
  layers when CPU runtime requires a smaller memory bank. The primary completed
  run uses `--patchcore-max-train-patches 10000` for L2, L3, and L2+L3.

Completed primary run:

- Output: `reports/generated/dissertation_runs/primary_balanced_by_subtype_10k`
- Models: Autoencoder baseline, PatchCore L2, PatchCore L3, PatchCore L2+L3
- Selected model for heatmaps and subtype-focused figures: PatchCore L3

## Evaluation Metrics

Primary metrics:

- AUROC
- AUPRC
- FPR@95%TPR
- threshold_at_95_tpr
- ID false rejection rate
- ID and OOD counts

Per-group reporting:

- Per `ood_type`
- Per `ood_subtype`

Threshold policy:

- `research_threshold`: computed from test labels for paper/reporting metrics only.
- `deployment_threshold`: computed from validation ID score quantile only for accept/reject decisions.

## Priority Figures

The curated dissertation figure set prioritizes:

1. Overall ROC comparison
2. Overall PR comparison
3. Layer ablation figure
4. Per-OOD-type comparison
5. Per-OOD-subtype comparison
6. Score distribution with threshold
7. Sensory artifact heatmap examples
8. Modality shift heatmap examples
9. Dataset taxonomy diagram
10. System pipeline overview
11. PatchCore method diagram
12. Experiment workflow diagram

Supporting outputs include metric summary tables, threshold policy tables, case selections, caption suggestions, and a figure index.

## Output Policy

Do not commit bulky intermediate run outputs, model checkpoints, memory banks, or raw heatmap pools.

Commit:

- Reproducible scripts and docs
- Small CSV and Markdown summaries
- Curated final dissertation figures
- Caption suggestions
- Figure index

Keep local/ignored:

- `reports/generated/dissertation_runs/**`
- raw checkpoints and memory banks
- full run internals
- temporary heatmap pools

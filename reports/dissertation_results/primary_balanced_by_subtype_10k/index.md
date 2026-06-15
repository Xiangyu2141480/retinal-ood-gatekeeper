# Generated Experiment Report Index

This index is generated from aggregate metrics and score tables only. It does not include raw
medical images, model weights, generated heatmaps, or patient identifiers.

## Threshold Warning

- `research_threshold`: computed using test labels for paper evaluation only.
- `deployment_threshold`: computed from val ID score quantile only, for UI/demo decision.

Research thresholds can use test labels to report paper metrics such as FPR@95%TPR. Deployment
thresholds must be calibrated from validation ID scores only before UI/demo decisions.

## Available Tables

- [Per-category metric table](per_category_metrics.md)
- [Layer ablation table](layer_ablation_table.md)
- [AE vs PatchCore table](ae_vs_patchcore_table.md)
- [Threshold policy table](threshold_policy_table.md)
- [Heatmap index](heatmap_index.md)
- [Top TP / FP / FN / borderline case selection](case_selection.md)

## Run Summary

- experiments: 4
- ID samples total across rows: 600
- OOD samples total across rows: 6600
- heatmap note: Index of generated heatmap artifacts. Image files remain generated outputs, not Git-tracked data.

## Warnings

- No warnings recorded.

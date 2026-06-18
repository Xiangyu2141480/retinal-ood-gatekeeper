# Thesis-Ready Writing Blocks

## Overall Contribution Paragraph

This dissertation presents an upstream quality-control system for retinal Fundus Autofluorescence (FAF) images. The primary contribution is an ID-only unsupervised OOD gatekeeper that rejects invalid or out-of-distribution inputs before downstream analysis. The final repository also contains a multi-scheme Stage 1 method comparison, robustness and failure analysis, a reproducible dataset package, dissertation-ready figures and tables, and an optional Stage 2 module that explains likely rejection reasons after a rejection has already occurred.

## Results Overview Paragraph

Across the final Stage 1 comparison, Mahalanobis feature distance is the strongest quantitative gatekeeper, achieving AUROC 0.9724, AUPRC 0.9975, and FPR@95%TPR 0.2000 on the balanced-by-subtype OOD evaluation. PatchCore L3 is retained as the localization-oriented companion because it provides heatmap evidence for representative OOD and failure cases. The optional Stage 2 reason-attribution comparison identifies `linear_svm` as the selected family attribution method and the non-oracle `hierarchical_classifier` as the selected subtype attribution method.

## Stage 1 Comparison Paragraph

Stage 1 remains the core OOD gatekeeper. It is trained using ID FAF rows only and is evaluated on held-out OOD stress-test categories. The comparison shows that feature-distance methods outperform the reconstruction and low-level statistics baselines, with Mahalanobis feature distance giving the best overall quantitative trade-off. PatchCore L3 does not surpass Mahalanobis on aggregate metrics, but it remains important because its patch-level heatmaps provide localized qualitative evidence that supports the discussion of failure cases.

## Stage 2 Comparison Paragraph

Stage 2 is an optional post-rejection explanation layer. It does not change the Stage 1 gatekeeper and does not use OOD labels for Stage 1 fitting. In the Stage 2 method comparison, `linear_svm` is selected for reason-family attribution, with family test accuracy 0.9881 and macro-F1 0.9901. The non-oracle `hierarchical_classifier` is selected for subtype attribution, with subtype accuracy 0.9238 and macro-F1 0.9059. These outputs should be interpreted as likely rejection explanations rather than clinical diagnoses.

## Limitations Paragraph

The results are proof-of-concept stress-test evidence rather than clinical deployment validation. The ID test split uses synthetic FAF fallback rather than real clinical FAF validation, and the OOD sets are curated evaluation categories rather than clinical prevalence samples. Stage 2 is supervised post-hoc explanation, not unsupervised OOD detection. Its reason splits are disjoint by image path but not by `parent_image_hash` for generated sensory artifacts, so the high Stage 2 scores may be optimistic for parent-independent generalization.

## Figure Citation Blocks

Use `figure_system_pipeline_overview.png` to introduce the binary Stage 1 gatekeeper. Use `figure_dataset_taxonomy.png` to define the ID/OOD taxonomy and committed manifest sizes. Use `figure_metrics_by_scheme.png` and `figure_fpr95_by_scheme.png` to support the Stage 1 model-selection claim. Use `figure_layer_ablation_patchcore.png` and heatmap examples to justify PatchCore L3 as the localization-oriented companion. Use `figure_two_stage_updated_pipeline.png`, `figure_reason_method_family_macro_f1.png`, and the two Stage 2 confusion matrices to present the optional reason-attribution layer.

# Thesis-Ready Writing Blocks

## Overall Contribution Paragraph

This dissertation presents an upstream quality-control system for retinal Fundus Autofluorescence (FAF) images. The primary contribution is an ID-only unsupervised OOD gatekeeper that rejects invalid or out-of-distribution inputs before downstream analysis. The final repository also contains a multi-scheme Stage 1 method comparison, robustness and failure analysis, a reproducible dataset package, dissertation-ready figures and tables, and an optional Stage 2 module that explains likely rejection reasons after a rejection has already occurred.

## Results Overview Paragraph

Across the final Stage 1 comparison, Mahalanobis feature distance is the strongest quantitative gatekeeper, achieving AUROC 0.9724, AUPRC 0.9975, and FPR@95%TPR 0.2000 on the balanced-by-subtype OOD evaluation. PatchCore L3 is retained as the localization-oriented companion because it provides heatmap evidence for representative OOD and failure cases. In the optional parent-grouped Stage 2 comparison, grouped validation selects `feature_statistics_fusion` for family attribution and the non-oracle `hierarchical_classifier` for subtype attribution.

## Stage 1 Comparison Paragraph

Stage 1 remains the core OOD gatekeeper. It is trained using ID FAF rows only and is evaluated on held-out OOD stress-test categories. The comparison shows that feature-distance methods outperform the reconstruction and low-level statistics baselines, with Mahalanobis feature distance giving the best overall quantitative trade-off. PatchCore L3 does not surpass Mahalanobis on aggregate metrics, but it remains important because its patch-level heatmaps provide localized qualitative evidence that supports the discussion of failure cases.

## Stage 2 Comparison Paragraph

Stage 2 is an optional supervised post-rejection explanation layer. It does not change the Stage 1 gatekeeper and does not use OOD labels for Stage 1 fitting. Grouped validation selects `feature_statistics_fusion` for reason-family attribution (macro-F1 0.9925); retrospective grouped test accuracy and macro-F1 are both 1.0000, with no unique hardest family. Grouped validation selects the non-oracle `hierarchical_classifier` for subtype attribution (macro-F1 0.9059); grouped test accuracy is 0.9357 and macro-F1 is 0.9176. These outputs are likely rejection explanations rather than clinical diagnoses.

## Limitations Paragraph

The results are proof-of-concept stress-test evidence rather than clinical deployment validation. The ID test split uses synthetic FAF fallback rather than real clinical FAF validation, and the OOD sets are curated evaluation categories rather than clinical prevalence samples. Stage 2 is supervised post-hoc explanation, not unsupervised OOD detection. Parent grouping eliminates cross-partition image-path and group overlap, but variants from a common parent remain dependent within one split. It does not establish patient-, device-, or site-independent generalisation.

## Figure Citation Blocks

Use `figure_two_stage_updated_pipeline.png` as the canonical system figure because it shows the ID-only Stage 1 gatekeeper and the optional post-rejection Stage 2 explanation layer without blurring their boundaries. Use `figure_dataset_taxonomy.png` and `figure_manifest_split_sizes.png` to define the ID/OOD taxonomy, task-specific manifest rows, and synthetic FAF fallback caveat. Use `figure_stage1_overall_comparison_combined.png` to support the Stage 1 model-selection claim, with the full ROC/PR curves reserved for the appendix. Use `figure_patchcore_layer_ablation_combined.png` and the qualitative heatmap examples to justify PatchCore L3 as the localisation-oriented companion. Use `figure_grouped_stage2_method_comparison.png` and `figure_grouped_family_confusion_matrix.png` in the main text; place `figure_grouped_subtype_confusion_matrix.png` and `figure_legacy_vs_grouped_stage2_metrics.png` in the appendix.

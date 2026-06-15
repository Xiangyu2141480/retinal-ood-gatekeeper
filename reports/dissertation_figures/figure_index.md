# Figure Index

| File | Figure title | Purpose | Recommended dissertation section | Short caption suggestion |
| --- | --- | --- | --- | --- |
| reports/dissertation_figures/figure_roc_overall_model_comparison.png | overall ROC comparison | Compare discrimination across unsupervised gatekeeper models. | Results | Overall ROC comparison of the evaluated unsupervised gatekeeper models. |
| reports/dissertation_figures/figure_pr_overall_model_comparison.png | overall PR comparison | Compare precision-recall behavior under class imbalance. | Results | Overall precision-recall comparison on the primary OOD benchmark. |
| reports/dissertation_figures/figure_layer_ablation_patchcore.png | PatchCore layer ablation | Show which feature layer choice is strongest. | Results | PatchCore feature-layer ablation across the evaluated ResNet layers. |
| reports/dissertation_figures/figure_per_ood_type_comparison.png | per-OOD-type comparison | Identify which broad OOD category is easiest or hardest. | Results | Per-OOD-type AUROC comparison for modality, artifact, and semantic outliers. |
| reports/dissertation_figures/figure_per_ood_subtype_comparison.png | per-OOD-subtype comparison | Show detailed subtype-level strengths and weaknesses. | Results | Subtype-level performance for the selected PatchCore gatekeeper. |
| reports/dissertation_figures/figure_score_distribution_with_threshold.png | score distribution with threshold | Explain score separation and deployment thresholding. | Results / Methods | Anomaly score distributions for ID and OOD groups with the selected threshold. |
| reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png | sensory artifact heatmap examples | Show representative artifact localization behavior. | Results / Discussion | Representative sensory artifact heatmap examples from the selected PatchCore model. |
| reports/dissertation_figures/figure_heatmaps_modality_examples.png | modality shift heatmap examples | Show wrong-modality responses from the gatekeeper. | Results / Discussion | Representative modality shift heatmap examples from the selected PatchCore model. |
| reports/dissertation_figures/figure_dataset_taxonomy.png | dataset taxonomy diagram | Define the ID/OOD taxonomy without implying multiclass training. | Dataset | Dataset taxonomy used for ID-only training and OOD-only evaluation. |
| reports/dissertation_figures/figure_system_pipeline_overview.png | system pipeline overview | Explain the binary gatekeeper system role. | Introduction / Methods | System overview: input image, OOD gatekeeper, accept/reject decision. |
| reports/dissertation_figures/figure_patchcore_method.png | PatchCore method | Explain the feature-memory-bank method. | Methods | PatchCore method diagram showing feature extraction, memory bank scoring, and thresholding. |
| reports/dissertation_figures/figure_experiment_workflow.png | experiment workflow diagram | Summarize train/evaluate/aggregate/figure workflow. | Methods | Experiment workflow from dataset package through evaluation and figure generation. |

# Final Thesis Figure Shortlist

Use this as the practical dissertation figure menu. Keep the main text tight and move backup diagnostics to the appendix.

## Main Text Figures

| path | title | recommended thesis chapter | suggested caption | why included | placement | status |
|---|---|---|---|---|---|---|
| `reports/dissertation_figures/figure_system_pipeline_overview.png` | System pipeline overview | Methodology | System overview for the unsupervised binary FAF OOD gatekeeper. The model accepts valid FAF images and rejects invalid or OOD inputs; it is not a disease classifier. | Establishes the task framing. | main text | must_include |
| `reports/dissertation_figures/figure_dataset_taxonomy.png` | Dataset taxonomy | Dataset | Dataset taxonomy for ID FAF, modality shift, sensory artifact, and semantic outlier inputs. OOD labels are used for evaluation grouping only. | Makes the four input categories concrete without implying supervised class training. | main text | must_include |
| `reports/dissertation_figures/figure_roc_overall_model_comparison.png` | Overall ROC comparison | Results | Overall ROC comparison across unsupervised ID-trained OOD gatekeeper models. Mahalanobis feature distance gives the strongest ranking performance. | Main quantitative model-ranking figure. | main text | must_include |
| `reports/dissertation_figures/figure_pr_overall_model_comparison.png` | Overall PR comparison | Results | Overall precision-recall comparison across candidate gatekeepers on the OOD-heavy evaluation set. | Complements ROC under class imbalance. | main text | must_include |
| `reports/dissertation_figures/figure_layer_ablation_patchcore.png` | PatchCore layer ablation | Experiments | PatchCore feature-layer ablation showing layer-level trade-offs for the heatmap-capable model family. | Supports method-selection discussion. | main text | optional |
| `reports/dissertation_figures/figure_per_ood_type_comparison.png` | Per-OOD-type comparison | Results | Per-OOD-type performance comparison for modality shift, sensory artifact, and semantic outlier groups. | Shows where models are strong or weak. | main text | must_include |
| `reports/dissertation_figures/figure_per_ood_subtype_comparison.png` | Per-OOD-subtype comparison | Results | Per-subtype comparison identifying easy global shifts and hard subtle local artifacts. | Useful if the chapter has space for detailed subtype evidence. | main text | optional |
| `reports/dissertation_figures/figure_score_distribution_with_threshold.png` | Score distribution with threshold | Threshold Safety | ID and OOD anomaly-score distributions with the selected ID-calibrated threshold. | Makes the binary decision rule visible. | main text | must_include |
| `reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png` | Sensory artifact heatmap examples | Discussion | PatchCore heatmap examples for sensory artifact cases. Heatmaps are qualitative localization evidence, not clinical explanations. | Demonstrates interpretability/localization. | main text | optional |
| `reports/dissertation_figures/figure_heatmaps_modality_examples.png` | Modality shift heatmap examples | Discussion | PatchCore heatmap examples for modality-shift inputs. | Provides qualitative contrast with sensory artifacts. | main text | optional |
| `reports/dissertation_figures/robustness/figure_bootstrap_ci_main_metrics.png` | Bootstrap CI main metrics | Robustness Analysis | Bootstrap confidence intervals for main OOD gatekeeper metrics. Mahalanobis remains strongest for AUROC/AUPRC on the current evaluation set. | Adds uncertainty to the main result. | main text | must_include |
| `reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png` | Threshold policy trade-off | Robustness Analysis | Threshold trade-off between synthetic-ID false rejection and OOD recall. Deployment thresholds must be calibrated from validation ID scores only. | Separates research and deployment-style thresholds. | main text | must_include |
| `reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png` | Feature-space PCA by OOD type | Discussion | PCA projection of Mahalanobis feature space by OOD type. Global shifts separate clearly while subtle sensory artifacts overlap more with ID. | Explains Mahalanobis strengths and limitations. | main text | must_include |
| `reports/dissertation_figures/robustness/figure_method_disagreement_examples.png` | Method disagreement examples | Failure Analysis | Representative cases where Mahalanobis and PatchCore L3 disagree. | Shows why a localizable companion remains useful. | main text | must_include |

## Appendix Figures

| path | title | recommended thesis chapter | suggested caption | why included | placement | status |
|---|---|---|---|---|---|---|
| `reports/dissertation_figures/robustness/figure_bootstrap_ci_fpr95.png` | Bootstrap FPR@95%TPR CI | Appendix / Robustness | Bootstrap uncertainty for FPR@95%TPR. Intervals are wider and overlap, so safety claims remain cautious. | Supports cautious threshold-safety wording. | appendix | backup |
| `reports/dissertation_figures/robustness/figure_artifact_severity_reject_rate.png` | Artifact severity reject rate | Appendix / Robustness | Deployment reject rate under generated artifact severity stress tests. Responses are not universally monotonic. | Shows weak and non-monotonic artifact responses. | appendix | optional |
| `reports/dissertation_figures/robustness/figure_false_negative_ood_examples.png` | False-negative OOD examples | Appendix / Failure Analysis | False-negative OOD examples for Mahalanobis. Text watermark dominates missed OOD cases. | Supports limitations and future work. | appendix | optional |
| `reports/dissertation_figures/robustness/figure_runtime_vs_performance.png` | Runtime versus performance | Appendix / Practicality | Runtime versus AUROC for candidate gatekeepers using local smoke scoring estimates. | Practical software-prototype evidence. | appendix | optional |
| `reports/dissertation_figures/figure_experiment_workflow.png` | Experiment workflow | Appendix / Reproducibility | Workflow from manifests to ID-only fitting, OOD-only evaluation, metrics, figures, and final indexes. | Helps examiners follow reproduction steps. | appendix | backup |

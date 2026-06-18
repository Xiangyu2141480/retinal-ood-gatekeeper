# Final Thesis Table Shortlist

Use these tables for the main dissertation text and appendix. Keep the main text to 6-10 tables.

## Main Text Tables

| path | title | recommended thesis chapter | suggested caption | why included | placement | status |
|---|---|---|---|---|---|---|
| `reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.md` | Overall model metrics | Results | Overall metrics for unsupervised ID-trained OOD gatekeeper schemes on the balanced-by-subtype evaluation set. | Main quantitative evidence. | main text | must_include |
| `reports/dissertation_results/multi_scheme_comparison/model_selection_summary.md` | Model selection summary | Results | Model-selection summary combining ranking metrics, interpretability, and complexity. | Explains why Mahalanobis is primary and PatchCore is complementary. | main text | must_include |
| `reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.md` | PatchCore layer ablation | Experiments | PatchCore layer ablation table for the heatmap-capable model family. | Supports PatchCore L3 selection. | main text | optional |
| `reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.md` | Per-OOD-type metrics | Results | Per-OOD-type metrics for modality shift, sensory artifact, and semantic outlier examples. | Shows category-specific behavior. | main text | must_include |
| `reports/dissertation_results/robustness_analysis/bootstrap_ci.md` | Bootstrap confidence intervals | Robustness Analysis | Bootstrap confidence intervals for main metrics. | Adds uncertainty to claims. | main text | must_include |
| `reports/dissertation_results/robustness_analysis/threshold_policy_sweep.md` | Threshold policy sweep | Threshold Safety | Research and deployment-style threshold policies with ID false rejection and OOD recall. | Essential safety threshold table. | main text | must_include |
| `reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md` | Robustness takeaway table | Discussion | Summary table linking each robustness analysis to its implication and limitation. | Examiner-friendly evidence summary. | main text | must_include |
| `reports/dissertation_results/reason_attribution_method_comparison/best_method_summary.md` | Stage 2 reason-attribution best methods | Phase 2 Results | Selected rejected-input reason-attribution methods and held-out metrics. | Records `linear_svm` family results and non-oracle `hierarchical_classifier` subtype results. | main text | must_include |
| `reports/dissertation_results/reason_attribution_method_comparison/family_metrics_by_method.md` | Reason family method comparison | Phase 2 Results | Family-level accuracy and macro-F1 across Stage 2 methods. | Shows why `linear_svm` is selected by validation macro-F1. | main text | recommended |
| `reports/dissertation_final/final_result_summary.md` | Final result summary | Results / Discussion | Final thesis-ready summary of dataset counts, model counts, best model, threshold result, hardest subtype, and limitations. | One-stop final result table. | main text | must_include |

## Appendix Tables

| path | title | recommended thesis chapter | suggested caption | why included | placement | status |
|---|---|---|---|---|---|---|
| `reports/dissertation_results/multi_scheme_comparison/per_ood_subtype_by_scheme.md` | Per-OOD-subtype metrics | Appendix | Detailed per-subtype metrics for all schemes. | Too detailed for main text but useful for viva questions. | appendix | optional |
| `reports/dissertation_results/robustness_analysis/train_size_sensitivity.md` | Train-size sensitivity | Appendix / Robustness | ID train-size sensitivity for lightweight feature-distance methods. | Supports data-efficiency discussion. | appendix | optional |
| `reports/dissertation_results/robustness_analysis/artifact_severity_stress.md` | Artifact severity stress | Appendix / Robustness | Generated artifact severity stress-test results. | Documents non-monotonic and weak severity responses. | appendix | optional |
| `reports/dissertation_results/robustness_analysis/runtime_resource_summary.md` | Runtime/resource summary | Appendix / Practicality | Local smoke scoring estimates and artifact sizes. | Practicality evidence without overclaiming deployment readiness. | appendix | optional |
| `reports/dissertation_results/robustness_analysis/result_provenance.md` | Result provenance | Appendix / Reproducibility | Provenance for robustness outputs, including input score files, manifests, and ID-only/OOD-only guardrails. | Proves no accidental OOD training. | appendix | must_include |
| `reports/dissertation_results/reason_attribution_method_comparison/subtype_metrics_by_method.md` | Reason subtype method comparison | Appendix / Phase 2 Results | Subtype accuracy and macro-F1 across Stage 2 methods. | Documents the `hierarchical_classifier` subtype result. | appendix | recommended |
| `reports/dissertation_results/reason_attribution_method_comparison/leakage_sanity_check.md` | Reason attribution leakage sanity check | Appendix / Limitations | Feature exclusion, split disjointness, parent-hash overlap, and non-oracle hierarchy checks. | Required limitation evidence for interpreting high Stage 2 scores. | appendix | must_include |

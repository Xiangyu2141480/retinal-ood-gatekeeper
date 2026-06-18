| path | title | chapter | placement | status | why |
| --- | --- | --- | --- | --- | --- |
| reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.md | Overall model metrics | Results | main text | must_include | Main AUROC/AUPRC/FPR@95%TPR comparison. |
| reports/dissertation_results/multi_scheme_comparison/model_selection_summary.md | Model selection summary | Results | main text | must_include | Explains selected quantitative gatekeeper. |
| reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.md | PatchCore layer ablation | Experiments | main text | optional | Layer-level PatchCore comparison. |
| reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.md | Per-OOD-type metrics | Results | main text | must_include | Breaks performance down by OOD type. |
| reports/dissertation_results/robustness_analysis/bootstrap_ci.md | Bootstrap confidence intervals | Robustness Analysis | main text | must_include | Uncertainty around main metrics. |
| reports/dissertation_results/robustness_analysis/threshold_policy_sweep.md | Threshold policy sweep | Threshold Safety | main text | must_include | Research versus ID-calibrated deployment thresholds. |
| reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md | Robustness takeaway table | Discussion | main text | must_include | Concise claim-to-implication summary. |
| reports/dissertation_results/reason_attribution_method_comparison/best_method_summary.md | Stage 2 reason-attribution best methods | Phase 2 Results | main text | must_include | Summarizes selected family and subtype explanation methods. |
| reports/dissertation_results/reason_attribution_method_comparison/family_metrics_by_method.md | Reason family method comparison | Phase 2 Results | main text | recommended | Supports selecting `linear_svm` for family attribution. |
| reports/dissertation_results/robustness_analysis/runtime_resource_summary.md | Runtime/resource summary | Appendix | appendix | optional | Local smoke scoring estimates and model artifact sizes. |
| reports/dissertation_results/reason_attribution_method_comparison/subtype_metrics_by_method.md | Reason subtype method comparison | Appendix | appendix | recommended | Documents non-oracle hierarchical subtype performance. |
| reports/dissertation_results/reason_attribution_method_comparison/leakage_sanity_check.md | Reason attribution leakage sanity check | Appendix | appendix | must_include | Documents feature exclusions, split checks, parent-hash limitation, and non-oracle hierarchy. |
